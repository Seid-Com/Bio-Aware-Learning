"""
Quantitative Experimental Evaluation Suite for TA-BC-BFL.

Implements the experiments required by peer review:
  E1  Ablation study        — FedAvg / Bio-only / Trust-only / Full TA-BC-BFL
  E2  Adversarial robustness— poisoning, label-flip, Sybil (accuracy drop)
  E3  Privacy trade-off     — epsilon sweep + gradient-inversion leakage score
  E4  Scalability           — client count vs wall-clock time per round
  E5  Statistical validation— multi-seed mean/std/95% CI + paired t-test

Usage (from the simulation/ directory):
    python experiments.py                      # full run (100 rounds, 5 seeds)
    python experiments.py --quick              # smoke run (10 rounds, 2 seeds)
    python experiments.py --rounds 100 --seeds 5 --out results.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, replace, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from framework.federated_learning import (
    FederatedLearningSimulator,
    SimulationConfig,
    SimulationResult,
)
from framework.clients import LocalModelUpdate
from framework.attacks import AdversarialAttackSimulator

DEFAULT_SEEDS: List[int] = [42, 123, 456, 789, 1024, 2048, 31415, 271828, 161803, 99991]


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────
def mean_std_ci(values: Sequence[float], confidence: float = 0.95):
    """Return (mean, std, ci_low, ci_high) using Student's t distribution."""
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    mean = float(vals.mean())
    if n < 2:
        return mean, 0.0, mean, mean
    std = float(vals.std(ddof=1))
    try:
        from scipy import stats
        t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    except Exception:  # pragma: no cover - scipy always available in deps
        t_crit = 1.96
    half = t_crit * std / math.sqrt(n)
    return mean, std, mean - half, mean + half


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired t-test of runs a vs b (a = proposed method, b = baseline)."""
    from scipy import stats
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n = min(len(a_arr), len(b_arr))
    if n == 0:
        return {"t_statistic": 0.0, "p_value": 1.0,
                "mean_difference": 0.0, "n_pairs": 0}
    a_arr, b_arr = a_arr[:n], b_arr[:n]
    diff = a_arr - b_arr
    if np.std(diff) == 0.0:
        # Zero variance in differences: degenerate case scipy returns nan for
        if np.allclose(diff, 0.0):
            return {"t_statistic": 0.0, "p_value": 1.0,
                    "mean_difference": 0.0, "n_pairs": n}
        return {"t_statistic": float("inf"), "p_value": 0.0,
                "mean_difference": float(diff.mean()), "n_pairs": n}
    t_stat, p_value = stats.ttest_rel(a_arr, b_arr)
    return {"t_statistic": float(t_stat), "p_value": float(p_value),
            "mean_difference": float(diff.mean()), "n_pairs": n}


# ─────────────────────────────────────────────────────────────────────────────
#  Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentSummary:
    """Statistical summary of one experimental condition."""
    label: str
    metric: str
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    n_runs: int
    runs: List[float] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)

    @classmethod
    def from_runs(cls, label: str, runs: Sequence[float],
                  metric: str = "accuracy", extra: Optional[Dict] = None) -> "ExperimentSummary":
        mean, std, lo, hi = mean_std_ci(runs)
        return cls(label=label, metric=metric, mean=mean, std=std,
                   ci95_low=lo, ci95_high=hi, n_runs=len(list(runs)),
                   runs=[float(r) for r in runs], extra=extra or {})

    def table_row(self) -> str:
        return (f"| {self.label} | {self.mean:.2%} | {self.std:.4f} "
                f"| [{self.ci95_low:.2%}, {self.ci95_high:.2%}] | {self.n_runs} |")


@dataclass
class Report:
    """Complete experimental report."""
    meta: Dict = field(default_factory=dict)
    ablation: List[ExperimentSummary] = field(default_factory=list)
    attacks: List[ExperimentSummary] = field(default_factory=list)
    privacy: List[ExperimentSummary] = field(default_factory=list)
    leakage: Dict[str, float] = field(default_factory=dict)
    scalability: List[Dict] = field(default_factory=list)
    significance: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        def ser(obj):
            if isinstance(obj, ExperimentSummary):
                d = asdict(obj)
                return d
            if isinstance(obj, dict):
                return {str(k): ser(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [ser(v) for v in obj]
            return obj
        return ser(asdict(self))

    def to_markdown(self) -> str:
        lines: List[str] = ["# TA-BC-BFL Experimental Results", ""]

        lines += ["## E1. Ablation Study (final accuracy)", "",
                  "| Configuration | Mean | Std | 95% CI | Runs |",
                  "|---|---|---|---|---|"]
        lines += [s.table_row() for s in self.ablation]

        lines += ["", "## E2. Adversarial Robustness (method x attack)", "",
                  "| Method | Attack | Mean Acc | Std | 95% CI | Drop vs Clean | Adv. Wt (final) |",
                  "|---|---|---|---|---|---|---|"]
        for s in self.attacks:
            method, attack = s.label.split(" | ")
            drop = s.extra.get("accuracy_drop")
            drop_str = f"{drop:.2%}" if drop is not None else "—"
            adv_wt = s.extra.get("adv_weight_share_final")
            adv_str = f"{adv_wt:.2%}" if adv_wt is not None else "—"
            lines.append(f"| {method} | {attack} | {s.mean:.2%} | {s.std:.4f} "
                         f"| [{s.ci95_low:.2%}, {s.ci95_high:.2%}] | {drop_str} "
                         f"| {adv_str} |")

        lines += ["", "## E3. Privacy–Accuracy Trade-off", "",
                  "| Epsilon | Mean Accuracy | Std | 95% CI | Runs |",
                  "|---|---|---|---|---|"]
        lines += [s.table_row() for s in self.privacy]

        if self.leakage:
            lines += ["", "### Gradient-Inversion Leakage (lower = better)", "",
                      "| Epsilon | Reconstruction Quality |",
                      "|---|---|"]
            if "no_dp" in self.leakage:
                lines.append(f"| no DP (baseline) | {self.leakage['no_dp']:.4f} |")
            for eps, q in sorted(
                    ((k, v) for k, v in self.leakage.items() if k != "no_dp"),
                    key=lambda x: float(x[0])):
                lines.append(f"| {eps} | {q:.4f} |")

        if self.scalability:
            lines += ["", "## E4. Scalability", "",
                      "| Clients | Mean Time/Round (s) | Rounds |",
                      "|---|---|---|"]
            for row in self.scalability:
                lines.append(f"| {row['clients']} | {row['seconds_per_round']:.4f} "
                             f"| {row['rounds']} |")

        if self.significance:
            lines += ["", "## E5. Statistical Validation", ""]
            for name, res in self.significance.items():
                sig = "SIGNIFICANT" if res.get("p_value", 1.0) < 0.01 else \
                      "significant" if res.get("p_value", 1.0) < 0.05 else "NOT significant"
                lines.append(
                    f"- **{name}**: t={res['t_statistic']:.3f}, "
                    f"p={res['p_value']:.4f} ({sig} at alpha=0.05), "
                    f"mean diff={res['mean_difference']:+.4f}, n={res['n_pairs']}"
                )
        return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
#  Experiment runner
# ─────────────────────────────────────────────────────────────────────────────
class ExperimentRunner:
    """Runs the five peer-review experiment groups over multiple seeds."""

    def __init__(
        self,
        base_config: Optional[SimulationConfig] = None,
        seeds: Optional[Sequence[int]] = None,
        verbose: bool = True,
    ):
        # Defaults must match the CLI (main()), which passes --lr 0.001;
        # SimulationConfig.learning_rate defaults to 0.01, so state it
        # explicitly here or the programmatic and CLI paths disagree.
        self.base = base_config or SimulationConfig(
            num_clients=10, num_rounds=100, local_epochs=5,
            learning_rate=0.001,
            use_dp=True, dp_epsilon=1.0, use_he=True,
        )
        self.seeds = list(seeds or DEFAULT_SEEDS[:5])
        self.verbose = verbose

        # populated by the run_* methods (read by run_significance / run_all)
        self.ablation: List[ExperimentSummary] = []
        self.attacks: List[ExperimentSummary] = []
        self.privacy: List[ExperimentSummary] = []
        self.leakage: Dict[str, float] = {}
        self.scalability: List[Dict] = []
        self.significance: Dict = {}

    # -- helpers ----------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _cfg(self, seed: int, **overrides) -> SimulationConfig:
        return replace(self.base, seed=seed, **overrides)

    def _run_tabfl(self, config: SimulationConfig) -> SimulationResult:
        return FederatedLearningSimulator(config).run_tabfl()

    def _run_method(self, method: str, seed: int, **overrides) -> float:
        """Run one condition and return final accuracy."""
        config = self._cfg(seed, **overrides)
        if method == "tabfl":
            return self._run_tabfl(config).final_accuracy
        # every other method (fedavg, perfedavg, blockchain_fl, trust_fl)
        # runs through the simulator's baseline path
        sim = FederatedLearningSimulator(config)
        return sim.run_baseline(method).final_accuracy

    def _sweep(self, label: str, method: str, overrides_by_seed=None, **common) -> ExperimentSummary:
        runs = []
        for seed in self.seeds:
            ov = dict(common)
            if overrides_by_seed:
                ov.update(overrides_by_seed.get(seed, {}))
            runs.append(self._run_method(method, seed, **ov))
        return ExperimentSummary.from_runs(label, runs)

    # -- E1: ablation -----------------------------------------------------
    def run_ablation(self) -> List[ExperimentSummary]:
        self._log("[E1] Ablation study ...")
        no_privacy = dict(use_dp=False, use_he=False, adversarial_fraction=0.0)

        conditions = [
            ("FedAvg", "fedavg", no_privacy),
            ("Per-FedAvg", "perfedavg", no_privacy),
            ("Blockchain-FL", "blockchain_fl", no_privacy),
            ("Trust-FL", "trust_fl", no_privacy),
            ("Bio-Aware FL", "tabfl",
             dict(no_privacy, use_bio=True, use_trust=False)),
            ("Trust-Aware FL", "tabfl",
             dict(no_privacy, use_bio=False, use_trust=True)),
            ("Full TA-BC-BFL", "tabfl",
             dict(no_privacy, use_bio=True, use_trust=True)),
        ]

        summaries = []
        for label, method, ov in conditions:
            s = self._sweep(label, method, **ov)
            s.extra["accuracy_pct"] = round(s.mean * 100, 2)
            summaries.append(s)
            self._log(f"    {label:<20} acc={s.mean:.2%} ± {s.std:.4f}")
        self.ablation = summaries
        return summaries

    # -- E2: attack resistance (method x attack matrix) ---------------------
    def run_attacks(self, adversarial_fraction: float = 0.2) -> List[ExperimentSummary]:
        """
        Method x attack robustness matrix.

        Each method is evaluated clean and under three attack types; the
        degradation relative to that method's own clean accuracy quantifies
        attack resistance (smaller drop = more robust).
        """
        self._log("[E2] Adversarial robustness (method x attack matrix) ...")

        # (label, runner, config overrides applied for every attack column)
        methods = [
            ("FedAvg", "fedavg",
             dict(use_dp=False, use_he=False)),
            ("Per-FedAvg", "perfedavg",
             dict(use_dp=False, use_he=False)),
            ("Blockchain-FL", "blockchain_fl",
             dict(use_dp=False, use_he=False)),
            ("Trust-FL", "trust_fl",
             dict(use_dp=False, use_he=False)),
            ("Bio-Aware FL", "tabfl",
             dict(use_dp=False, use_he=False, use_bio=True, use_trust=False)),
            ("Trust-Aware FL", "tabfl",
             dict(use_dp=False, use_he=False, use_bio=False, use_trust=True)),
            ("Full TA-BC-BFL", "tabfl",
             dict(use_dp=True, use_he=True, use_bio=True, use_trust=True)),
        ]
        attacks = [
            ("No attack", dict(adversarial_fraction=0.0)),
            ("Model poisoning", dict(adversarial_fraction=adversarial_fraction,
                                     attack_type="model_poisoning")),
            ("Label flipping", dict(adversarial_fraction=adversarial_fraction,
                                    attack_type="label_flipping")),
            ("Sybil", dict(adversarial_fraction=adversarial_fraction,
                           attack_type="sybil")),
        ]

        summaries: List[ExperimentSummary] = []
        clean_by_method: Dict[str, float] = {}
        for m_label, m_runner, m_common in methods:
            for a_label, a_ov in attacks:
                label = f"{m_label} | {a_label}"
                runs = []
                adv_shares = []
                for seed in self.seeds:
                    ov = {**m_common, **a_ov}
                    config = self._cfg(seed, **ov)
                    if m_runner == "tabfl":
                        res = self._run_tabfl(config)
                    else:
                        sim = FederatedLearningSimulator(config)
                        res = sim.run_baseline(m_runner)
                    runs.append(res.final_accuracy)
                    if res.rounds:
                        adv_shares.append(float(
                            res.rounds[-1].aggregation_metrics.get(
                                "adversarial_weight_share", 0.0)))
                s = ExperimentSummary.from_runs(label, runs)
                if adv_shares:
                    s.extra["adv_weight_share_final"] = float(np.mean(adv_shares))
                summaries.append(s)
                if a_label == "No attack":
                    clean_by_method[m_label] = s.mean
                self._log(f"    {label:<36} acc={s.mean:.2%} ± {s.std:.4f}")

        # degradation vs each method's own clean accuracy
        for s in summaries:
            m_label = s.label.split(" | ")[0]
            clean = clean_by_method.get(m_label)
            if clean is not None:
                s.extra["accuracy_drop"] = float(clean - s.mean)
                s.extra["relative_drop_pct"] = (
                    float((clean - s.mean) * 100 / clean) if clean else 0.0)

        self.attacks = summaries
        return summaries

    # -- E3: privacy trade-off + leakage ----------------------------------
    def run_privacy(self, epsilons: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
                    leakage_iterations: int = 100) -> List[ExperimentSummary]:
        self._log("[E3] Privacy–accuracy trade-off ...")
        summaries = []
        leakage: Dict[str, float] = {}
        atk = AdversarialAttackSimulator(seed=0)

        for eps in epsilons:
            s = self._sweep(f"epsilon={eps}", "tabfl",
                            use_dp=True, use_he=True, dp_epsilon=eps,
                            adversarial_fraction=0.0)
            summaries.append(s)
            self._log(f"    eps={eps:<5} acc={s.mean:.2%} ± {s.std:.4f}")

            leakage[str(eps)] = self._leakage_score(eps, atk, leakage_iterations)

        # No-DP baseline (vulnerable reference)
        leakage["no_dp"] = self._leakage_score(None, atk, leakage_iterations)
        self._log(f"    no DP    leakage={leakage['no_dp']:.4f} (baseline)")

        self.privacy = summaries
        self.leakage = leakage
        return summaries

    def _leakage_score(self, epsilon: Optional[float], atk: AdversarialAttackSimulator,
                       iterations: int) -> float:
        """Mean gradient-inversion reconstruction quality (0 = no leakage)."""
        if epsilon is None:
            sim = FederatedLearningSimulator(
                self._cfg(self.seeds[0], use_dp=False, use_he=False,
                          adversarial_fraction=0.0))
        else:
            config = self._cfg(self.seeds[0], use_dp=True, dp_epsilon=epsilon,
                               use_he=False, adversarial_fraction=0.0)
            sim = FederatedLearningSimulator(config)
        selected = sim._select_clients()

        qualities = []
        for client in selected:
            clean_update = client.local_train(
                global_weights=sim.global_weights,
                global_bias=sim.global_bias,
                round_number=1,
                local_epochs=self.base.local_epochs,
                learning_rate=self.base.learning_rate,
                dp_epsilon=None,
            )
            reference = clean_update.gradients
            if epsilon is not None:
                # apply server-side DP noise exactly as the FL loop does
                noisy_grads, _ = sim.dp.add_noise(reference)
                observed = replace(clean_update, gradients=noisy_grads,
                                   noise_applied=True)
            else:
                observed = clean_update
            res = atk.gradient_inversion_attempt(
                observed, iterations=iterations,
                clean_gradients=reference)  # attacker aims at clean signal
            qualities.append(res["final_quality"])
        return float(np.mean(qualities))

    # -- E4: scalability ---------------------------------------------------
    def run_scalability(self, client_counts: Sequence[int] = (5, 10, 20, 40),
                        seeds: Optional[Sequence[int]] = None) -> List[Dict]:
        self._log("[E4] Scalability ...")
        rows = []
        for n in client_counts:
            times = []
            for seed in (seeds or self.seeds[:3]):
                config = self._cfg(seed, num_clients=n,
                                   use_dp=False, use_he=False,
                                   run_baselines=False, baseline_methods=[])
                sim = FederatedLearningSimulator(config)
                t0 = time.perf_counter()
                sim.run_tabfl()
                elapsed = time.perf_counter() - t0
                times.append(elapsed / self.base.num_rounds)
            mean_t = float(np.mean(times))
            rows.append({"clients": n, "seconds_per_round": round(mean_t, 6),
                         "rounds": self.base.num_rounds,
                         "speedup_vs_5": None})
            self._log(f"    {n:>3} clients -> {mean_t:.4f} s/round")

        if rows:
            base_t = rows[0]["seconds_per_round"]
            for r in rows:
                r["speedup_vs_5"] = round(base_t / r["seconds_per_round"], 3) \
                    if r["seconds_per_round"] else None
        self.scalability = rows
        return rows

    # -- E5: significance ---------------------------------------------------
    def run_significance(self) -> Dict[str, Dict]:
        """
        Paired t-tests:
          1. Clean accuracy: Full TA-BC-BFL vs each baseline (E1 runs)
          2. Robustness: per-attack accuracy Full TA-BC-BFL vs FedAvg (E2 runs)
        """
        self._log("[E5] Statistical validation ...")
        if not self.ablation:
            self.run_ablation()
        if not self.attacks:
            self.run_attacks()

        tests: Dict[str, Dict] = {}

        by_label = {s.label: s for s in self.ablation}
        proposed = by_label.get("Full TA-BC-BFL")
        if proposed:
            for label in ("FedAvg", "Bio-Aware FL", "Trust-Aware FL"):
                other = by_label.get(label)
                if other and proposed.runs and other.runs:
                    key = f"clean: Full TA-BC-BFL vs {label}"
                    tests[key] = paired_ttest(proposed.runs, other.runs)
                    self._log(f"    {key:<44} p={tests[key]['p_value']:.4f}")

        attack_by_label = {s.label: s for s in self.attacks}
        for a_label in ("Model poisoning", "Label flipping", "Sybil"):
            full = attack_by_label.get(f"Full TA-BC-BFL | {a_label}")
            fed = attack_by_label.get(f"FedAvg | {a_label}")
            if full and fed and full.runs and fed.runs:
                key = f"attack({a_label}): Full TA-BC-BFL vs FedAvg"
                tests[key] = paired_ttest(full.runs, fed.runs)
                self._log(f"    {key:<44} p={tests[key]['p_value']:.4f}")

        self.significance = tests
        return tests

    # -- orchestration ------------------------------------------------------
    def run_all(self) -> Report:
        t0 = time.perf_counter()
        report = Report()
        report.ablation = self.run_ablation()
        report.attacks = self.run_attacks()
        report.privacy = self.run_privacy()
        report.leakage = dict(self.leakage)
        report.scalability = self.run_scalability()
        report.significance = self.run_significance()
        report.meta = {
            "seeds": self.seeds,
            "num_clients": self.base.num_clients,
            "num_rounds": self.base.num_rounds,
            "local_epochs": self.base.local_epochs,
            "learning_rate": self.base.learning_rate,
            "data_distribution": self.base.data_distribution,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }
        self._log(f"[done] total {report.meta['elapsed_seconds']}s")
        return report


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TA-BC-BFL experiment suite")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out", default="experiment_results.json")
    parser.add_argument("--md", default=None, help="also write markdown report")
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: 10 rounds, 2 seeds")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    rounds = 10 if args.quick else args.rounds
    n_seeds = 2 if args.quick else args.seeds

    config = SimulationConfig(
        num_clients=args.clients,
        num_rounds=rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        data_distribution="non_iid_mild",
        use_dp=True, dp_epsilon=1.0, use_he=True,
    )

    runner = ExperimentRunner(config, seeds=DEFAULT_SEEDS[:n_seeds],
                              verbose=not args.quiet)
    report = runner.run_all()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print(f"wrote {args.out}")

    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        print(f"wrote {args.md}")
    else:
        print()
        print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
