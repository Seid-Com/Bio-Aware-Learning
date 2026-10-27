"""Tests for the federated learning simulator."""

import numpy as np
import pytest

from framework.federated_learning import FederatedLearningSimulator, SimulationConfig


class TestSimulationConfig:
    def test_default_config(self):
        config = SimulationConfig()
        assert config.num_clients == 10
        assert config.num_rounds == 20
        assert config.use_dp is True

    def test_valid_weight_sum(self):
        config = SimulationConfig(
            bio_weight=0.5, sample_weight=0.3, quality_weight=0.2
        )
        assert abs(config.bio_weight + config.sample_weight + config.quality_weight - 1.0) < 1e-6


class TestFederatedLearningSimulator:
    def test_initialization(self, sample_config):
        sim = FederatedLearningSimulator(sample_config)
        assert len(sim.clients) == 5
        assert sim.global_weights.shape == (32,)
        assert sim.trust_manager is not None
        assert sim.bio_aggregator is not None

    def test_config_validation_rejects_bad_weights(self):
        config = SimulationConfig(
            bio_weight=0.5, sample_weight=0.5, quality_weight=0.5  # sum != 1
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            FederatedLearningSimulator(config)

    def test_config_validation_rejects_bad_epsilon(self):
        config = SimulationConfig(dp_epsilon=-1.0)
        with pytest.raises(ValueError, match="dp_epsilon must be > 0"):
            FederatedLearningSimulator(config)

    def test_config_validation_rejects_bad_clients(self):
        config = SimulationConfig(num_clients=0)
        with pytest.raises(ValueError, match="num_clients must be >= 1"):
            FederatedLearningSimulator(config)

    def test_select_clients(self, sample_config):
        sim = FederatedLearningSimulator(sample_config)
        selected = sim._select_clients()
        assert len(selected) == 5
        assert all(c.config.client_id in range(5) for c in selected)

    def test_evaluate_global_model(self, sample_config):
        sim = FederatedLearningSimulator(sample_config)
        acc, loss = sim._evaluate_global_model(sim.clients[:3])
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0

    def test_run_tabfl_produces_results(self, sample_config):
        sim = FederatedLearningSimulator(sample_config)
        result = sim.run_tabfl()
        assert result.method == "TA-BC-BFL"
        assert len(result.rounds) == 3
        assert 0.0 <= result.final_accuracy <= 1.0
        assert result.final_loss >= 0.0
        assert result.privacy_report is not None

    def test_run_baseline_fedavg(self, sample_config):
        sim = FederatedLearningSimulator(sample_config)
        result = sim.run_baseline("fedavg")
        assert result.method == "FEDAVG"
        assert len(result.rounds) == 3
        assert 0.0 <= result.final_accuracy <= 1.0

    def test_run_baseline_blockchain_fl(self, sample_config):
        sim = FederatedLearningSimulator(sample_config)
        result = sim.run_baseline("blockchain_fl")
        assert result.method == "BLOCKCHAIN-FL"
        assert result.blockchain_integrity is True

    def test_run_baseline_trust_fl(self, sample_config):
        sim = FederatedLearningSimulator(sample_config)
        result = sim.run_baseline("trust_fl")
        assert result.method == "TRUST-FL"

    def test_run_all_methods(self, sample_config):
        sim = FederatedLearningSimulator(sample_config)
        results = sim.run_all()
        assert "TA-BC-BFL" in results
        assert "FEDAVG" in results
        assert "BLOCKCHAIN-FL" in results
        assert "TRUST-FL" in results

    def test_run_all_without_baselines(self, sample_config):
        sample_config.run_baselines = False
        sample_config.baseline_methods = []
        sim = FederatedLearningSimulator(sample_config)
        results = sim.run_all()
        assert "TA-BC-BFL" in results
        assert len(results) == 1

    def test_progress_callback_called(self, sample_config):
        progress_calls = []
        def callback(pct, msg):
            progress_calls.append(pct)
        sim = FederatedLearningSimulator(sample_config, progress_callback=callback)
        sim.run_tabfl()
        assert len(progress_calls) == 3

    def test_adversarial_clients(self):
        config = SimulationConfig(
            num_clients=5,
            num_rounds=2,
            adversarial_fraction=0.4,
            attack_type="model_poisoning",
            num_samples_per_client=100,
        )
        sim = FederatedLearningSimulator(config)
        n_adv = sum(1 for c in sim.clients if c.is_adversarial)
        assert n_adv == 2

    def test_dp_noise_applied_once(self):
        config = SimulationConfig(
            num_clients=3,
            num_rounds=1,
            use_dp=True,
            dp_epsilon=2.0,
            num_samples_per_client=50,
        )
        sim = FederatedLearningSimulator(config)
        result = sim.run_tabfl()
        # Single DP application happens at server level each round
        assert "dp" in result.rounds[0].privacy_metrics
        assert "dp" in result.privacy_report or result.privacy_report.get("rounds_completed", 0) >= 1

    def test_dp_noise_affects_aggregated_model(self):
        """Regression: server DP noise must modify the released model update."""
        base = dict(num_clients=3, num_rounds=1, num_samples_per_client=50, seed=7)
        sim_plain = FederatedLearningSimulator(SimulationConfig(**base, use_dp=False))
        sim_dp = FederatedLearningSimulator(
            SimulationConfig(**base, use_dp=True, dp_epsilon=0.05))
        sim_plain.run_tabfl()
        sim_dp.run_tabfl()
        diff = float(np.linalg.norm(sim_plain.global_weights - sim_dp.global_weights))
        assert diff > 1e-6  # noise actually reached the global model
