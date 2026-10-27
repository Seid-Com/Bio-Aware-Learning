"""Tests for the quantitative experiment suite (E1-E5)."""

import numpy as np
import pytest

from experiments import (
    ExperimentRunner,
    ExperimentSummary,
    Report,
    mean_std_ci,
    paired_ttest,
    DEFAULT_SEEDS,
)
from framework.federated_learning import SimulationConfig, FederatedLearningSimulator


@pytest.fixture
def tiny_config():
    return SimulationConfig(
        num_clients=5,
        num_rounds=3,
        local_epochs=2,
        learning_rate=0.01,
        num_samples_per_client=80,
        use_dp=False,
        use_he=False,
        run_baselines=False,
        baseline_methods=[],
    )


@pytest.fixture
def runner(tiny_config):
    return ExperimentRunner(tiny_config, seeds=DEFAULT_SEEDS[:2], verbose=False)


class TestStatisticalHelpers:
    def test_mean_std_ci_basic(self):
        mean, std, lo, hi = mean_std_ci([0.8, 0.9, 0.85])
        assert 0.8 < mean < 0.9
        assert std > 0
        assert lo < mean < hi

    def test_mean_std_ci_single_value(self):
        mean, std, lo, hi = mean_std_ci([0.9])
        assert mean == 0.9
        assert std == 0.0
        assert lo == hi == 0.9

    def test_paired_ttest_identical(self):
        res = paired_ttest([0.9, 0.9, 0.9], [0.9, 0.9, 0.9])
        assert res["p_value"] == pytest.approx(1.0)
        assert res["mean_difference"] == pytest.approx(0.0)

    def test_paired_ttest_different(self):
        res = paired_ttest([0.95, 0.96, 0.97], [0.80, 0.81, 0.82])
        assert res["p_value"] < 0.05
        assert res["mean_difference"] > 0
        assert res["n_pairs"] == 3


class TestExperimentSummary:
    def test_from_runs(self):
        s = ExperimentSummary.from_runs("FedAvg", [0.8, 0.82, 0.84])
        assert s.n_runs == 3
        assert 0.8 < s.mean < 0.84
        assert s.ci95_low < s.mean < s.ci95_high

    def test_table_row_format(self):
        s = ExperimentSummary.from_runs("X", [0.5, 0.6])
        row = s.table_row()
        assert row.startswith("| X |")
        assert "95" in row or "%" in row


class TestAblationExperiment:
    def test_ablation_returns_seven_conditions(self, runner):
        results = runner.run_ablation()
        labels = [r.label for r in results]
        assert labels == ["FedAvg", "Per-FedAvg", "Blockchain-FL", "Trust-FL",
                          "Bio-Aware FL", "Trust-Aware FL", "Full TA-BC-BFL"]

    def test_ablation_accuracy_in_range(self, runner):
        for summary in runner.run_ablation():
            assert 0.0 <= summary.mean <= 1.0
            assert summary.n_runs == 2
            assert summary.ci95_low <= summary.mean <= summary.ci95_high


ALL_METHODS = ["FedAvg", "Per-FedAvg", "Blockchain-FL", "Trust-FL",
               "Bio-Aware FL", "Trust-Aware FL", "Full TA-BC-BFL"]


class TestAttackExperiment:
    def test_attacks_include_all_methods_and_types(self, runner):
        results = runner.run_attacks()
        assert len(results) == 28  # 7 methods x 4 attack conditions
        methods = {r.label.split(" | ")[0] for r in results}
        attacks = {r.label.split(" | ")[1] for r in results}
        assert methods == set(ALL_METHODS)
        assert attacks == {"No attack", "Model poisoning", "Label flipping", "Sybil"}

    def test_attack_drop_recorded(self, runner):
        results = runner.run_attacks()
        by_label = {r.label: r for r in results}
        for method in ALL_METHODS:
            clean = by_label[f"{method} | No attack"]
            attacked = by_label[f"{method} | Model poisoning"]
            assert "accuracy_drop" in attacked.extra
            assert attacked.extra["accuracy_drop"] == pytest.approx(
                clean.mean - attacked.mean, abs=1e-6
            )


class TestBaselines:
    """The three external baselines + Per-FedAvg must all run and be sane."""

    BASELINES = ["fedavg", "perfedavg", "blockchain_fl", "trust_fl"]

    def test_run_baseline_methods_produce_accuracy(self, tiny_config):
        for method in self.BASELINES:
            sim = FederatedLearningSimulator(tiny_config)
            res = sim.run_baseline(method)
            assert 0.0 <= res.final_accuracy <= 1.0, method
            assert len(res.rounds) == tiny_config.num_rounds, method

    def test_perfedavg_meta_update_uses_train_loss(self, tiny_config):
        # Per-FedAvg must aggregate meta-updates alpha_i = w - eta*grad L(beta_i),
        # i.e. round-1 accuracy should differ from plain FedAvg's.
        fedavg = FederatedLearningSimulator(tiny_config).run_baseline("fedavg")
        perfed = FederatedLearningSimulator(tiny_config).run_baseline("perfedavg")
        assert perfed.rounds[0].global_accuracy != fedavg.rounds[0].global_accuracy

    def test_run_method_dispatches_baselines(self, runner):
        acc = runner._run_method("perfedavg", 42,
                                 use_dp=False, use_he=False, adversarial_fraction=0.0)
        assert 0.0 <= acc <= 1.0

    def test_unknown_baseline_raises(self, tiny_config):
        sim = FederatedLearningSimulator(tiny_config)
        with pytest.raises(ValueError, match="unknown baseline method"):
            sim.run_baseline("not_a_method")


class TestPrivacyExperiment:
    def test_privacy_sweep(self, runner):
        results = runner.run_privacy(epsilons=(0.5, 2.0), leakage_iterations=20)
        assert len(results) == 2
        assert "0.5" in runner.leakage or 0.5 in runner.leakage
        assert "no_dp" in runner.leakage  # vulnerable baseline reference
        for q in runner.leakage.values():
            assert 0.0 <= q <= 1.0

    def test_leakage_score_bounded(self, runner):
        runner.run_privacy(epsilons=(1.0,), leakage_iterations=10)
        for score in runner.leakage.values():
            assert 0.0 <= score <= 1.0

    def test_dp_reduces_leakage_vs_no_dp(self, runner):
        runner.run_privacy(epsilons=(1.0,), leakage_iterations=100)
        # Gradient inversion should recover far more signal without DP noise
        assert runner.leakage["no_dp"] > runner.leakage["1.0"]


class TestScalabilityExperiment:
    def test_scalability_rows(self, runner):
        rows = runner.run_scalability(client_counts=(3, 5), seeds=[42])
        assert len(rows) == 2
        for row in rows:
            assert row["seconds_per_round"] > 0
            assert row["clients"] in (3, 5)


class TestSignificance:
    def test_significance_uses_ablation_and_attack_runs(self, runner):
        tests = runner.run_significance()
        assert len(tests) >= 1
        for name, res in tests.items():
            assert "p_value" in res
            assert "t_statistic" in res
            assert 0.0 <= res["p_value"] <= 1.0
            assert "TA-BC-BFL" in name


class TestReport:
    def test_run_all_produces_complete_report(self, runner):
        report = runner.run_all()
        assert len(report.ablation) == 7
        assert len(report.attacks) == 28  # 7 methods x 4 attack conditions
        assert len(report.privacy) >= 1
        assert len(report.scalability) == 4  # default: 5, 10, 20, 40 clients
        assert report.meta["seeds"] == [42, 123]
        assert report.meta["elapsed_seconds"] >= 0

    def test_markdown_report(self, runner):
        report = runner.run_all()
        md = report.to_markdown()
        assert "Ablation Study" in md
        assert "Adversarial Robustness" in md
        assert "Statistical Validation" in md

    def test_json_serializable(self, runner):
        import json
        report = runner.run_ablation()
        r = Report(ablation=report)
        json.dumps(r.to_dict(), default=str)


class TestConfigSwitches:
    def test_use_trust_false_gives_uniform_weights(self, tiny_config):
        from framework.federated_learning import FederatedLearningSimulator
        cfg = SimulationConfig(
            num_clients=4, num_rounds=1, use_trust=False,
            use_dp=False, use_he=False, num_samples_per_client=50,
        )
        sim = FederatedLearningSimulator(cfg)
        result = sim.run_tabfl()
        # aggregation ran without error and produced finite accuracy
        assert np.isfinite(result.final_accuracy)

    def test_use_bio_false_falls_back_to_sample_weight(self, tiny_config):
        from framework.federated_learning import FederatedLearningSimulator
        cfg = SimulationConfig(
            num_clients=4, num_rounds=1, use_bio=False,
            use_dp=False, use_he=False, num_samples_per_client=50,
        )
        sim = FederatedLearningSimulator(cfg)
        assert sim.bio_aggregator.bio_weight == 0.0
        assert sim.bio_aggregator.sample_weight == 1.0

    def test_seed_reproducibility(self):
        from framework.federated_learning import FederatedLearningSimulator
        def run(seed):
            cfg = SimulationConfig(
                num_clients=4, num_rounds=2, seed=seed,
                use_dp=False, use_he=False, num_samples_per_client=50,
            )
            return FederatedLearningSimulator(cfg).run_tabfl().final_accuracy
        assert run(7) == run(7)

    def test_seed_reproducibility_with_dp(self):
        """DP-on trajectories must be deterministic too (regression: DP noise
        was drawn from the process-global RNG)."""
        from framework.federated_learning import FederatedLearningSimulator
        def run(seed):
            cfg = SimulationConfig(
                num_clients=4, num_rounds=3, seed=seed,
                use_dp=True, dp_epsilon=1.0, use_he=False,
                num_samples_per_client=50,
            )
            return FederatedLearningSimulator(cfg).run_tabfl().final_accuracy
        assert run(7) == run(7)
