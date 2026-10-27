"""Tests for the privacy module (DP + HE)."""

import numpy as np
import pytest

from framework.privacy import (
    DifferentialPrivacy,
    HomomorphicEncryption,
    PrivacyAccountant,
    advanced_composition_total,
    rdp_total_epsilon,
)


class TestDifferentialPrivacy:
    @pytest.fixture
    def dp(self):
        return DifferentialPrivacy(epsilon=2.0, delta=1e-5, max_grad_norm=1.0)

    def test_noise_multiplier_calibration(self, dp):
        assert dp.noise_multiplier > 0
        # For epsilon=2.0, delta=1e-5: sigma ≈ sqrt(2*ln(1.25/1e-5)) / 2.0
        expected = np.sqrt(2 * np.log(1.25 / 1e-5)) / 2.0
        assert abs(dp.noise_multiplier - expected) < 0.01

    def test_clip_gradients_reduces_norm(self, dp):
        grads = np.ones(32) * 10.0
        clipped, norm = dp.clip_gradients(grads)
        assert norm == 1.0
        assert np.linalg.norm(clipped) <= 1.0 + 1e-8

    def test_clip_gradients_no_change_when_small(self, dp):
        grads = np.ones(32) * 0.01
        clipped, norm = dp.clip_gradients(grads)
        assert norm == np.linalg.norm(grads)
        np.testing.assert_array_almost_equal(clipped, grads)

    def test_add_noise_changes_gradients(self, dp):
        grads = np.ones(32) * 0.5
        noisy, info = dp.add_noise(grads)
        assert not np.array_equal(noisy, grads)
        assert "noise_scale" in info
        assert "snr_db" in info
        assert "gradient_norm_before" in info
        assert "gradient_norm_after" in info

    def test_noise_stream_reproducible_across_instances(self):
        """Same seed => bit-identical noise (regression: was drawn from the
        process-global np.random, making DP runs irreproducible)."""
        grads = np.linspace(-0.5, 0.5, 65)
        dp1 = DifferentialPrivacy(epsilon=1.0, seed=7)
        dp2 = DifferentialPrivacy(epsilon=1.0, seed=7)
        n1, _ = dp1.add_noise(grads.copy(), sensitivity=0.01)
        n2, _ = dp2.add_noise(grads.copy(), sensitivity=0.01)
        assert np.array_equal(n1, n2)

    def test_noise_stream_depends_on_seed(self):
        grads = np.linspace(-0.5, 0.5, 65)
        dp1 = DifferentialPrivacy(epsilon=1.0, seed=7)
        dp2 = DifferentialPrivacy(epsilon=1.0, seed=8)
        n1, _ = dp1.add_noise(grads.copy(), sensitivity=0.01)
        n2, _ = dp2.add_noise(grads.copy(), sensitivity=0.01)
        assert not np.array_equal(n1, n2)

    def test_add_noise_custom_sensitivity(self, dp):
        """Central-DP mode: sensitivity=2C/n yields proportionally smaller noise."""
        grads = np.ones(32) * 0.01
        np.random.seed(0)
        _, info_default = dp.add_noise(grads)
        np.random.seed(0)
        _, info_central = dp.add_noise(grads, sensitivity=2.0 * 1.0 / 5000)
        assert info_central["noise_scale"] < info_default["noise_scale"] / 100
        # epsilon accounting is identical (bound cancels)
        assert dp.accountant.rounds == 2

    def test_privacy_accountant_tracks_rounds(self, dp):
        grads = np.ones(32) * 0.5
        dp.add_noise(grads)
        dp.add_noise(grads)
        assert dp.accountant.rounds == 2
        assert dp.accountant.spent_epsilon > 0

    def test_budget_exhaustion(self):
        dp = DifferentialPrivacy(epsilon=0.01, delta=1e-5, max_grad_norm=1.0)
        for _ in range(100):
            dp.add_noise(np.ones(32))
        assert dp.accountant.is_budget_exhausted

    def test_privacy_report_structure(self, dp):
        dp.add_noise(np.ones(32))
        report = dp.get_privacy_report()
        assert "epsilon" in report
        assert "delta" in report
        assert "spent_epsilon" in report
        assert "rounds_completed" in report
        assert "budget_used_fraction" in report

    def test_zero_epsilon_returns_large_noise(self):
        dp = DifferentialPrivacy(epsilon=0.0, delta=1e-5)
        assert dp.noise_multiplier == 10.0


class TestHomomorphicEncryption:
    @pytest.fixture
    def he(self):
        return HomomorphicEncryption(key_size_bits=2048)

    def test_encrypt_decrypt_roundtrip(self, he):
        plaintext = np.array([1.0, 2.0, 3.0, 4.0])
        ciphertext, meta = he.encrypt(plaintext)
        decrypted = he.decrypt(ciphertext)
        np.testing.assert_array_almost_equal(decrypted, plaintext, decimal=2)

    def test_encryption_ops_counted(self, he):
        assert he.encryption_ops == 0
        he.encrypt(np.ones(10))
        assert he.encryption_ops == 1

    def test_aggregate_ciphertexts(self, he):
        cts = [np.ones(10) * i for i in range(3)]
        weights = np.array([0.5, 0.3, 0.2])
        result = he.aggregate_ciphertexts(cts, weights)
        expected = 0.5 * 0 + 0.3 * 1 + 0.2 * 2
        np.testing.assert_array_almost_equal(result, np.ones(10) * expected)

    def test_secure_aggregate_pipeline(self, he):
        grads = [np.ones(10) * i for i in range(3)]
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        result, metrics = he.secure_aggregate(grads, weights)
        assert result.shape == (10,)
        assert "encryption_overhead_ratio" in metrics
        assert metrics["encryption_ops"] == 3

    def test_security_level(self):
        he_4096 = HomomorphicEncryption(key_size_bits=4096)
        assert "256-bit" in he_4096.get_security_level()

        he_2048 = HomomorphicEncryption(key_size_bits=2048)
        assert "112-bit" in he_2048.get_security_level()

        he_1024 = HomomorphicEncryption(key_size_bits=1024)
        assert "80-bit" in he_1024.get_security_level()


class TestPrivacyAccountant:
    def test_advanced_composition(self):
        acct = PrivacyAccountant(epsilon=5.0, delta=1e-5)
        acct.update(1.0, 1e-5)
        assert acct.spent_epsilon > 0
        assert acct.spent_delta > 0

    def test_budget_remaining(self):
        acct = PrivacyAccountant(epsilon=10.0, delta=1e-5)
        acct.update(1.0, 1e-5)
        assert acct.budget_remaining > 0
        assert acct.budget_remaining < 10.0

    def test_fraction_used(self):
        acct = PrivacyAccountant(epsilon=10.0, delta=1e-5)
        acct.update(1.0, 1e-5)
        assert 0.0 < acct.budget_fraction_used < 1.0


class TestCompositionAccounting:
    """Total privacy budget after k composed releases (paper Appendix A.5)."""

    def test_advanced_composition_matches_formula(self):
        eps_r, delta_r, k = 1.0, 1e-5, 100
        eps, delta = advanced_composition_total(eps_r, delta_r, k)
        expected_eps = eps_r * np.sqrt(2 * k * np.log(1 / delta_r)) + k * eps_r * (
            np.exp(eps_r) - 1
        )
        expected_delta = k * delta_r * (np.exp(eps_r) - 1) + k * delta_r
        assert eps == pytest.approx(expected_eps)
        assert delta == pytest.approx(expected_delta)

    def test_advanced_composition_deployment_value(self):
        """Deployed per-round (1.0, 1e-5) over T=100 rounds -> (219.81, 2.72e-3)."""
        eps, delta = advanced_composition_total(1.0, 1e-5, 100)
        assert eps == pytest.approx(219.81, abs=0.05)
        assert delta == pytest.approx(2.72e-3, rel=0.02)

    def test_advanced_composition_monotone_in_rounds(self):
        prev = -1.0
        for k in (1, 10, 50, 100, 500):
            eps, _ = advanced_composition_total(1.0, 1e-5, k)
            assert eps > prev
            prev = eps

    def test_rdp_deployment_value(self):
        """sigma = sqrt(2 ln(1.25/delta))/1 ~ 4.845, k=100, delta=1e-5 -> ~12.0."""
        sigma = np.sqrt(2 * np.log(1.25 / 1e-5)) / 1.0
        eps, alpha = rdp_total_epsilon(sigma, 100, 1e-5)
        assert sigma == pytest.approx(4.8447, abs=1e-3)
        assert eps == pytest.approx(12.04, abs=0.1)
        assert alpha > 1

    def test_rdp_tighter_than_advanced_composition(self):
        sigma = np.sqrt(2 * np.log(1.25 / 1e-5))
        rdp_eps, _ = rdp_total_epsilon(sigma, 100, 1e-5)
        adv_eps, _ = advanced_composition_total(1.0, 1e-5, 100)
        assert rdp_eps < adv_eps

    def test_rdp_monotone_in_rounds_and_noise(self):
        sigma = np.sqrt(2 * np.log(1.25 / 1e-5))
        e1, _ = rdp_total_epsilon(sigma, 10, 1e-5)
        e2, _ = rdp_total_epsilon(sigma, 100, 1e-5)
        assert e2 > e1
        louder, _ = rdp_total_epsilon(2 * sigma, 100, 1e-5)
        assert louder < e2

    def test_privacy_loss_reports_both_accountants(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        loss = dp.compute_privacy_loss(100)
        assert loss["total_epsilon"] == pytest.approx(219.81, abs=0.05)
        assert loss["total_delta"] == pytest.approx(2.72e-3, rel=0.02)
        assert loss["total_epsilon_rdp"] == pytest.approx(12.04, abs=0.1)
        # Regression: the old implementation returned min(composed, spent),
        # understating the budget actually consumed.
        assert loss["total_epsilon"] >= loss["rounds"] * 0.5

    def test_privacy_report_exposes_rdp_total(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        for _ in range(3):
            dp.add_noise(np.ones(32))
        report = dp.get_privacy_report()
        assert report["spent_epsilon_rdp"] > 0
        assert report["spent_epsilon_rdp"] < report["spent_epsilon"]
        assert report["epsilon_per_round"] == pytest.approx(1.0)
