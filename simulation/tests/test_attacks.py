"""Tests for the adversarial attack simulation module."""

from dataclasses import replace

import numpy as np
import pytest

from framework.attacks import AdversarialAttackSimulator
from framework.clients import LocalModelUpdate


@pytest.fixture
def attack_sim():
    return AdversarialAttackSimulator(seed=42)


@pytest.fixture
def clean_update():
    rng = np.random.RandomState(42)
    return LocalModelUpdate(
        client_id=0,
        weights=rng.randn(32) * 0.01,
        gradients=rng.randn(33) * 0.01,
        loss=0.5,
        accuracy=0.75,
        num_samples=200,
        omics_type="genomic",
        bio_features=rng.randn(16),
        round_number=1,
    )


class TestAdversarialAttackSimulator:
    def test_model_poisoning_sign_flip(self, attack_sim, clean_update):
        poisoned = attack_sim.model_poisoning_attack(clean_update, strategy="sign_flip", scale=3.0)
        np.testing.assert_array_almost_equal(poisoned.gradients, -clean_update.gradients * 3.0)
        assert poisoned.is_malicious is True

    def test_model_poisoning_scale_up(self, attack_sim, clean_update):
        poisoned = attack_sim.model_poisoning_attack(clean_update, strategy="scale_up", scale=3.0)
        assert np.linalg.norm(poisoned.gradients) > np.linalg.norm(clean_update.gradients)
        assert poisoned.is_malicious is True

    def test_model_poisoning_noise(self, attack_sim, clean_update):
        poisoned = attack_sim.model_poisoning_attack(clean_update, strategy="noise", scale=5.0)
        assert poisoned.is_malicious is True
        assert not np.array_equal(poisoned.gradients, clean_update.gradients)

    def test_sybil_attack(self, attack_sim, clean_update):
        sybils = attack_sim.sybil_attack(clean_update, num_sybils=3)
        assert len(sybils) == 3
        for s in sybils:
            assert s.is_malicious is True
            assert s.client_id != clean_update.client_id

    def test_gradient_inversion(self, attack_sim, clean_update):
        result = attack_sim.gradient_inversion_attempt(clean_update, iterations=50)
        assert "final_quality" in result
        assert "reconstruction_quality_history" in result
        assert "data_leakage_risk" in result
        assert result["data_leakage_risk"] in ["High", "Medium", "Low"]
        assert 0.0 <= result["final_quality"] <= 1.0

    def test_gradient_inversion_noise_reduces_leakage(self, attack_sim, clean_update):
        noisy = clean_update.gradients + np.random.RandomState(7).randn(
            *clean_update.gradients.shape) * 5.0
        protected = replace(clean_update, gradients=noisy)
        clean_res = attack_sim.gradient_inversion_attempt(
            clean_update, iterations=50, clean_gradients=clean_update.gradients)
        noisy_res = attack_sim.gradient_inversion_attempt(
            protected, iterations=50, clean_gradients=clean_update.gradients)
        assert noisy_res["final_quality"] < clean_res["final_quality"]

    def test_backdoor_attack(self, attack_sim, clean_update):
        backdoored = attack_sim.backdoor_attack(clean_update, trigger_strength=0.5)
        assert backdoored.is_malicious is True
        assert not np.array_equal(backdoored.weights, clean_update.weights)

    def test_measure_attack_success(self, attack_sim):
        clean = np.ones(32)
        attacked = clean + 0.5 * np.ones(32)
        result = attack_sim.measure_attack_success(clean, attacked, baseline_accuracy=0.8)
        assert "accuracy_drop" in result
        assert "attack_success_rate" in result
        assert result["accuracy_drop"] > 0

    def test_attack_summary_empty(self, attack_sim):
        summary = attack_sim.get_attack_summary()
        assert summary["total_attacks"] == 0

    def test_attack_summary_after_attacks(self, attack_sim, clean_update):
        attack_sim.model_poisoning_attack(clean_update)
        attack_sim.sybil_attack(clean_update, num_sybils=2)
        attack_sim.gradient_inversion_attempt(clean_update)
        summary = attack_sim.get_attack_summary()
        assert summary["total_attacks"] == 3
        assert "model_poisoning" in summary["by_type"]
        assert "sybil" in summary["by_type"]
