"""
Federated Learning Simulator Module
Implements the core TA-BC-BFL federated learning loop alongside
baseline methods (FedAvg, Blockchain-FL, Trust-FL) for comparison.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

from .clients import FederatedClient, ClientConfig, LocalModelUpdate, OMICS_TYPES
from .bio_aware import BioAwareAggregator, adversarial_weight_bound
from .blockchain import BlockchainTrustManager
from .privacy import DifferentialPrivacy, HomomorphicEncryption
from .attacks import AdversarialAttackSimulator


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@dataclass
class SimulationConfig:
    num_clients: int = 10
    num_rounds: int = 20
    local_epochs: int = 5
    learning_rate: float = 0.01
    fraction_clients: float = 1.0
    global_model_dim: int = 64

    # Bio-aware settings
    bio_weight: float = 0.4
    sample_weight: float = 0.3
    quality_weight: float = 0.3
    bio_temperature: float = 1.0

    # Blockchain settings
    initial_trust: float = 0.7
    trust_learning_rate: float = 0.15
    trust_decay: float = 0.95
    anomaly_threshold: float = 2.5
    mining_difficulty: int = 2

    # Privacy settings
    use_dp: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    use_he: bool = True
    he_key_size: int = 2048

    # Attack settings
    adversarial_fraction: float = 0.0
    attack_type: str = "model_poisoning"
    attack_scale: float = 3.0

    # Data settings
    num_samples_per_client: int = 500
    data_distribution: str = "non_iid_mild"

    # Reproducibility (multi-seed statistical validation)
    seed: int = 42

    # Ablation switches (isolate component contributions)
    use_bio: bool = True
    use_trust: bool = True

    # Baselines to compare
    run_baselines: bool = True
    baseline_methods: List[str] = field(default_factory=lambda: ["fedavg", "perfedavg", "blockchain_fl", "trust_fl"])


@dataclass
class RoundResult:
    round_number: int
    method: str
    global_accuracy: float
    global_loss: float
    participating_clients: int
    trust_scores: Optional[np.ndarray]
    bio_weights: Optional[np.ndarray]
    aggregation_metrics: Dict
    privacy_metrics: Dict
    flagged_clients: List[int]
    block_hash: Optional[str]
    communication_cost_mb: float


@dataclass
class SimulationResult:
    method: str
    config: SimulationConfig
    rounds: List[RoundResult]
    final_accuracy: float
    final_loss: float
    total_time_simulated_s: float
    privacy_report: Dict
    blockchain_integrity: bool
    attack_summary: Dict
    client_summaries: Dict


class FederatedLearningSimulator:
    """
    Full simulation of the TA-BC-BFL framework with baseline comparisons.

    The simulation proceeds as:
    1. Initialize global model and all clients
    2. Each round:
       a. Sample participating clients
       b. Clients perform local training (with optional DP noise)
       c. Server collects updates (optionally via HE secure aggregation)
       d. Blockchain records and updates trust scores
       e. Bio-aware aggregation computes weighted global model
    3. Evaluate global model accuracy after each round
    """

    def __init__(self, config: SimulationConfig, progress_callback: Optional[Callable] = None):
        self._validate_config(config)
        self.config = config
        self.progress_callback = progress_callback
        self.rng = np.random.RandomState(config.seed)

        self._setup_clients()
        self._setup_components()

    @staticmethod
    def _validate_config(config: SimulationConfig) -> None:
        """Validate simulation configuration parameters."""
        if config.num_clients < 1:
            raise ValueError("num_clients must be >= 1")
        if config.num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 <= config.fraction_clients <= 1.0:
            raise ValueError("fraction_clients must be in [0.0, 1.0]")
        if not 0.0 <= config.bio_weight <= 1.0:
            raise ValueError("bio_weight must be in [0.0, 1.0]")
        if not 0.0 <= config.sample_weight <= 1.0:
            raise ValueError("sample_weight must be in [0.0, 1.0]")
        if not 0.0 <= config.quality_weight <= 1.0:
            raise ValueError("quality_weight must be in [0.0, 1.0]")
        weight_sum = config.bio_weight + config.sample_weight + config.quality_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"bio_weight + sample_weight + quality_weight must sum to 1.0, got {weight_sum}")
        if config.dp_epsilon <= 0:
            raise ValueError("dp_epsilon must be > 0")
        if config.dp_delta <= 0 or config.dp_delta >= 1:
            raise ValueError("dp_delta must be in (0, 1)")
        if config.adversarial_fraction < 0 or config.adversarial_fraction > 0.5:
            raise ValueError("adversarial_fraction must be in [0.0, 0.5]")

    def _setup_clients(self):
        """Initialize federated clients with heterogeneous data."""
        n = self.config.num_clients
        n_adversarial = int(n * self.config.adversarial_fraction)

        omics_cycle = OMICS_TYPES * (n // len(OMICS_TYPES) + 1)
        self.clients: List[FederatedClient] = []

        for i in range(n):
            is_adversarial = i < n_adversarial
            config = ClientConfig(
                client_id=i,
                omics_type=omics_cycle[i],
                data_distribution=self.config.data_distribution,
                num_samples=max(10, self.config.num_samples_per_client
                                + self.rng.randint(-100, 100)),
                attack_type=self.config.attack_type if is_adversarial else "none",
                poison_rate=0.3 if is_adversarial else 0.0,
                seed=i * 137,
            )
            self.clients.append(FederatedClient(config, self.config.global_model_dim))

    def _setup_components(self):
        """Initialize TA-BC-BFL components (ablation switches applied here)."""
        # Ablation: with bio-awareness disabled, fall back to pure sample weighting
        if self.config.use_bio:
            bio_weight = self.config.bio_weight
            sample_weight = self.config.sample_weight
            quality_weight = self.config.quality_weight
        else:
            bio_weight = 0.0
            sample_weight = 1.0
            quality_weight = 0.0

        self.bio_aggregator = BioAwareAggregator(
            bio_weight=bio_weight,
            sample_weight=sample_weight,
            quality_weight=quality_weight,
            temperature=self.config.bio_temperature,
        )

        self.trust_manager = BlockchainTrustManager(
            num_clients=self.config.num_clients,
            initial_trust=self.config.initial_trust,
            learning_rate=self.config.trust_learning_rate,
            decay_factor=self.config.trust_decay,
            anomaly_threshold=self.config.anomaly_threshold,
            mining_difficulty=self.config.mining_difficulty,
        )

        if self.config.use_dp:
            self.dp = DifferentialPrivacy(
                epsilon=self.config.dp_epsilon,
                delta=self.config.dp_delta,
                max_grad_norm=1.0,
                seed=self.config.seed,
            )
        else:
            self.dp = None

        if self.config.use_he:
            self.he = HomomorphicEncryption(key_size_bits=self.config.he_key_size)
        else:
            self.he = None

        self.attack_simulator = AdversarialAttackSimulator()

        # Initialize global model
        self.global_weights = self.rng.randn(self.config.global_model_dim) * 0.01
        self.global_bias = 0.0

    def _select_clients(self) -> List[FederatedClient]:
        """Select a fraction of clients to participate in this round."""
        n_select = max(1, int(self.config.num_clients * self.config.fraction_clients))
        indices = self.rng.choice(self.config.num_clients, n_select, replace=False)
        return [self.clients[i] for i in sorted(indices)]

    def _evaluate_global_model(self, clients: List[FederatedClient]) -> tuple[float, float]:
        """Evaluate global model accuracy and loss across all participating clients."""
        all_accuracies = []
        all_losses = []

        for client in clients:
            feature_dim = client.features.shape[1]
            w = self.global_weights[:feature_dim] if len(self.global_weights) >= feature_dim else \
                np.pad(self.global_weights, (0, feature_dim - len(self.global_weights)))

            logits = client.features @ w + self.global_bias
            preds_prob = _sigmoid(logits)
            preds = (preds_prob >= 0.5).astype(int)
            acc = (preds == client.labels).mean()

            eps = 1e-12
            loss = -np.mean(
                client.labels * np.log(preds_prob + eps) +
                (1 - client.labels) * np.log(1 - preds_prob + eps)
            )
            all_accuracies.append(float(acc))
            all_losses.append(float(loss))

        return float(np.mean(all_accuracies)), float(np.mean(all_losses))

    def _run_round_tabfl(self, round_num: int, selected_clients: List[FederatedClient]) -> RoundResult:
        """Execute one round of TA-BC-BFL."""
        updates: List[LocalModelUpdate] = []

        # Local training — DP noise is applied at server level only (not per-client)
        # to avoid double-noising gradients
        for client in selected_clients:
            update = client.local_train(
                global_weights=self.global_weights,
                global_bias=self.global_bias,
                round_number=round_num,
                local_epochs=self.config.local_epochs,
                learning_rate=self.config.learning_rate,
                dp_epsilon=None,
            )
            updates.append(update)

        # Blockchain trust update
        trust_metrics = self.trust_manager.update_trust(updates, round_num)
        trust_weights = trust_metrics["trust_weights"]

        # Ablation: with trust-awareness disabled, all clients get equal trust
        if not self.config.use_trust:
            trust_weights = np.ones(len(trust_weights), dtype=float)
            trust_weights /= trust_weights.sum()

        # Privacy: central DP — server adds calibrated noise to the aggregated
        # model update (the quantity actually released). Sensitivity of the
        # sample-weighted mean of clipped updates is 2*C/n_total, so noise
        # shrinks as more clients/samples contribute (standard DP-FedAvg).
        privacy_metrics: Dict = {}
        he_metrics: Dict = {}

        # Homomorphic encryption secure aggregation
        if self.he is not None:
            grads_list = [u.gradients for u in updates]
            _, he_meta = self.he.secure_aggregate(grads_list, trust_weights)
            he_metrics = he_meta

        # Bio-aware aggregation
        new_weights, agg_metrics = self.bio_aggregator.aggregate(updates, trust_weights)

        # Adversarial influence metrics (Proposition 1, Appendix A.2):
        # realized share of the global update carried by adversarial clients,
        # plus the analytic bound computed from that round's alpha/trust.
        client_ids = [u.client_id for u in updates]
        gamma = agg_metrics["combined_weights"]
        alpha = agg_metrics["final_weights"]
        trust_scores = trust_metrics["trust_scores"]
        flagged_ids = set(trust_metrics["flagged_clients"])

        adv_pos = [i for i, cid in enumerate(client_ids)
                   if self.clients[cid].is_adversarial]
        if adv_pos:
            # Effective per-client trust actually applied this round
            if self.config.use_trust:
                beta = np.array([trust_scores[cid] for cid in client_ids])
            else:
                beta = np.ones(len(client_ids), dtype=float)
            adversarial_weight_share = float(gamma[adv_pos].sum())
            adversarial_bio_share = float(alpha[adv_pos].sum())
            beta_adv_max = float(beta[adv_pos].max())
            honest_pos = [i for i, cid in enumerate(client_ids)
                          if not self.clients[cid].is_adversarial]
            beta_honest_min = float(beta[honest_pos].min()) if honest_pos else 1.0
            weight_bound = adversarial_weight_bound(
                adversarial_bio_share, beta_adv_max, beta_honest_min)
        else:
            adversarial_weight_share = 0.0
            adversarial_bio_share = 0.0
            weight_bound = 0.0

        influence_metrics = {
            "adversarial_weight_share": adversarial_weight_share,
            "adversarial_bio_share": adversarial_bio_share,
            "adversarial_weight_bound": weight_bound,
            "flagged_this_round": len(flagged_ids),
        }

        if self.dp is not None:
            n_total = max(int(sum(u.num_samples for u in updates)), 1)
            sensitivity = 2.0 * self.dp.max_grad_norm / n_total
            new_weights, noise_info = self.dp.add_noise(new_weights, sensitivity=sensitivity)
            privacy_metrics["dp"] = self.dp.get_privacy_report()
            privacy_metrics["avg_snr_db"] = float(noise_info["snr_db"])
            privacy_metrics["noise_scale"] = float(noise_info["noise_scale"])

        # Update global model (with momentum-style smoothing)
        momentum = 0.1
        self.global_weights = (1 - momentum) * new_weights + momentum * self.global_weights
        self.global_bias = float(np.mean([u.weights.mean() for u in updates]) * 0.01)

        # Evaluate
        accuracy, loss = self._evaluate_global_model(selected_clients)

        # Communication cost estimate
        grad_bytes = sum(u.gradients.nbytes for u in updates)
        weight_bytes = sum(u.weights.nbytes for u in updates)
        comm_mb = (grad_bytes + weight_bytes) / (1024 * 1024)

        return RoundResult(
            round_number=round_num,
            method="TA-BC-BFL",
            global_accuracy=accuracy,
            global_loss=loss,
            participating_clients=len(selected_clients),
            trust_scores=self.trust_manager.trust_scores.copy(),
            bio_weights=agg_metrics.get("final_weights"),
            aggregation_metrics={**agg_metrics, **he_metrics, **influence_metrics},
            privacy_metrics=privacy_metrics,
            flagged_clients=trust_metrics["flagged_clients"],
            block_hash=trust_metrics["block_hash"],
            communication_cost_mb=comm_mb,
        )

    def run_tabfl(self, progress_start: float = 0.0, progress_end: float = 1.0) -> SimulationResult:
        """Run the full TA-BC-BFL simulation."""
        results: List[RoundResult] = []
        progress_range = progress_end - progress_start

        for r in range(1, self.config.num_rounds + 1):
            selected = self._select_clients()
            round_result = self._run_round_tabfl(r, selected)
            results.append(round_result)

            if self.progress_callback:
                progress = progress_start + (r / self.config.num_rounds) * progress_range
                self.progress_callback(progress, f"TA-BC-BFL Round {r}/{self.config.num_rounds}: acc={round_result.global_accuracy:.3f}")

        privacy_report = self.dp.get_privacy_report() if self.dp else {}

        return SimulationResult(
            method="TA-BC-BFL",
            config=self.config,
            rounds=results,
            final_accuracy=results[-1].global_accuracy if results else 0.0,
            final_loss=results[-1].global_loss if results else 0.0,
            total_time_simulated_s=float(self.config.num_rounds * self.config.local_epochs * 0.5),
            privacy_report=privacy_report,
            blockchain_integrity=self.trust_manager.verify_chain_integrity(),
            attack_summary=self.attack_simulator.get_attack_summary(),
            client_summaries=self.trust_manager.get_client_trust_summary(),
        )

    def run_baseline(self, method: str, progress_start: float = 0.0, progress_end: float = 1.0) -> SimulationResult:
        """Run a baseline federated learning method for comparison."""
        # Reset global model for fair comparison (same seed => paired runs)
        rng = np.random.RandomState(self.config.seed)
        global_weights = rng.randn(self.config.global_model_dim) * 0.01
        global_bias = 0.0
        results: List[RoundResult] = []
        progress_range = progress_end - progress_start

        for r in range(1, self.config.num_rounds + 1):
            selected = self._select_clients()
            updates = []

            for client in selected:
                update = client.local_train(
                    global_weights=global_weights,
                    global_bias=global_bias,
                    round_number=r,
                    local_epochs=self.config.local_epochs,
                    learning_rate=self.config.learning_rate,
                    dp_epsilon=None,
                )
                updates.append(update)

            # Compute aggregation based on method
            if method == "fedavg":
                # Standard FedAvg: weighted by sample count only
                total_samples = sum(u.num_samples for u in updates)
                agg_weights = np.array([u.num_samples / total_samples for u in updates])
                trust_scores = np.ones(len(selected)) / len(selected)
                block_hash = None

            elif method == "perfedavg":
                # First-order Per-FedAvg (Fallah et al., NeurIPS 2021).
                # local_train yields the personalized model beta_i; each
                # client then evaluates the meta-gradient of its training
                # loss at beta_i and returns the meta-update
                #   alpha_i = w_t - eta_loc * grad L_i(beta_i),
                # which the server averages sample-proportionally (FedAvg).
                # eta_loc is the inner-loop budget (lr x local_epochs x
                # ceil(n/64) batches): beta_i already moved that far, so the
                # meta-step is given the same step size as one inner epoch
                # sweep, otherwise Per-FedAvg under-converges vs FedAvg.
                dim = self.config.global_model_dim
                alphas = []
                for client, u in zip(selected, updates):
                    beta_w = u.weights[:dim]
                    beta_b = global_bias + float(u.gradients[-1])  # bias delta
                    _, grad_w, grad_b = client._compute_loss_and_grad(
                        client.features, client.labels, beta_w, beta_b)
                    batches = int(np.ceil(u.num_samples / 64)) or 1
                    eta_loc = (self.config.learning_rate
                               * self.config.local_epochs * batches)
                    aw = global_weights.copy()
                    n = min(len(grad_w), len(aw))
                    aw[:n] = (global_weights[:n] - eta_loc * grad_w[:n])
                    alphas.append((aw, global_bias - eta_loc * grad_b))
                total_samples = sum(u.num_samples for u in updates)
                agg_weights = np.array([u.num_samples / total_samples for u in updates])
                trust_scores = np.ones(len(selected)) / len(selected)
                block_hash = None

            elif method == "blockchain_fl":
                # Blockchain-FL: blockchain coordination but no bio-awareness
                # Simulate basic trust scoring without bio component
                grad_norms = np.array([np.linalg.norm(u.gradients) for u in updates])
                trust_scores = np.exp(-np.abs(grad_norms - np.median(grad_norms)))
                trust_scores /= trust_scores.sum()
                agg_weights = trust_scores
                block_hash = "sim_" + str(hash(str(r)))[:12]

            elif method == "trust_fl":
                # Trust-FL: trust-based but without blockchain or bio-awareness
                accuracies = np.array([u.accuracy for u in updates])
                trust_scores = accuracies / accuracies.sum()
                agg_weights = trust_scores
                block_hash = None

            else:
                raise ValueError(f"unknown baseline method: {method!r}")

            # Adversarial influence (Proposition 1): realized share of the
            # global update carried by adversarial clients under this baseline
            adv_pos = [i for i, c in enumerate(selected) if c.is_adversarial]
            adv_share = float(sum(agg_weights[i] for i in adv_pos)) if adv_pos else 0.0

            # Aggregate (Per-FedAvg aggregates the meta-updates alpha_i, not
            # the personalized models beta_i)
            if method == "perfedavg":
                agg_payload = [a[0] for a in alphas]
            else:
                agg_payload = [u.weights for u in updates]
            new_weights = np.zeros(self.config.global_model_dim)
            for i, payload in enumerate(agg_payload):
                new_weights += agg_weights[i] * payload

            global_weights = 0.9 * new_weights + 0.1 * global_weights
            if method == "perfedavg":
                global_bias = float(np.mean([a[1] for a in alphas]))
            else:
                global_bias = float(np.mean([u.weights.mean() for u in updates]) * 0.01)

            # Evaluate global model properly (same as TA-BC-BFL)
            all_accuracies = []
            all_losses = []
            for client in selected:
                feature_dim = client.features.shape[1]
                w = global_weights[:feature_dim] if len(global_weights) >= feature_dim else \
                    np.pad(global_weights, (0, feature_dim - len(global_weights)))

                logits = client.features @ w + global_bias
                preds_prob = _sigmoid(logits)
                preds = (preds_prob >= 0.5).astype(int)
                acc = (preds == client.labels).mean()

                eps = 1e-12
                loss = -np.mean(
                    client.labels * np.log(preds_prob + eps) +
                    (1 - client.labels) * np.log(1 - preds_prob + eps)
                )
                all_accuracies.append(float(acc))
                all_losses.append(float(loss))

            accuracy = float(np.mean(all_accuracies))
            loss = float(np.mean(all_losses))

            result = RoundResult(
                round_number=r,
                method=method.upper().replace("_", "-"),
                global_accuracy=accuracy,
                global_loss=loss,
                participating_clients=len(selected),
                trust_scores=trust_scores,
                bio_weights=None,
                aggregation_metrics={"adversarial_weight_share": adv_share},
                privacy_metrics={},
                flagged_clients=[],
                block_hash=block_hash,
                communication_cost_mb=float(sum(u.weights.nbytes for u in updates)) / 1e6,
            )
            results.append(result)

            if self.progress_callback:
                progress = progress_start + (r / self.config.num_rounds) * progress_range
                self.progress_callback(progress, f"{method.upper()} Round {r}/{self.config.num_rounds}")

        return SimulationResult(
            method=method.upper().replace("_", "-"),
            config=self.config,
            rounds=results,
            final_accuracy=results[-1].global_accuracy if results else 0.0,
            final_loss=results[-1].global_loss if results else 0.0,
            total_time_simulated_s=float(self.config.num_rounds * self.config.local_epochs * 0.5),
            privacy_report={},
            blockchain_integrity=method in ["blockchain_fl"],
            attack_summary={},
            client_summaries={},
        )

    def run_all(self) -> Dict[str, SimulationResult]:
        """Run TA-BC-BFL and all baselines."""
        results = {}
        n_methods = 1 + len(self.config.baseline_methods) if self.config.run_baselines else 1
        slot = 1.0 / n_methods

        results["TA-BC-BFL"] = self.run_tabfl(
            progress_start=0.0,
            progress_end=slot,
        )

        if self.config.run_baselines:
            for idx, method in enumerate(self.config.baseline_methods):
                start = (idx + 1) * slot
                end = start + slot
                results[method.upper().replace("_", "-")] = self.run_baseline(
                    method,
                    progress_start=start,
                    progress_end=min(end, 1.0),
                )

        return results
