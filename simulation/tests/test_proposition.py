"""
Tests for Proposition 1 (bounded adversarial influence, Appendix A.2):
closed-form bound, per-round metric recording, and empirical collapse of
adversarial aggregation weight under the trust layer.
"""

import numpy as np
import pytest

from framework.bio_aware import adversarial_weight_bound
from framework.federated_learning import FederatedLearningSimulator, SimulationConfig


class TestAdversarialWeightBound:
    """Closed-form properties of the Proposition 1 bound."""

    def test_conservative_closed_form(self):
        # s=0.2, beta_A=0.05 (trust floor), beta_H=0.7 (initial trust)
        # -> 0.01 / (0.01 + 0.56) ~ 1.754%
        assert adversarial_weight_bound(0.2, 0.05, 0.7) == pytest.approx(
            0.01 / 0.57, rel=1e-12)

    def test_asymptotic_closed_form(self):
        # honest trust converged to max_trust=1.0 -> 0.01 / 0.81 ~ 1.235%
        assert adversarial_weight_bound(0.2, 0.05, 1.0) == pytest.approx(
            0.01 / 0.81, rel=1e-12)

    def test_without_trust_layer_bound_equals_share(self):
        # beta_A == beta_H cancels trust entirely: bound collapses to s
        assert adversarial_weight_bound(0.3, 0.7, 0.7) == pytest.approx(0.3)

    def test_zero_share_gives_zero_bound(self):
        assert adversarial_weight_bound(0.0, 0.05, 0.7) == 0.0

    def test_monotonicity(self):
        b = adversarial_weight_bound
        assert b(0.2, 0.05, 1.0) < b(0.2, 0.05, 0.7)   # stronger honest trust
        assert b(0.2, 0.1, 0.7) > b(0.2, 0.05, 0.7)     # weaker adv. trust
        assert b(0.3, 0.05, 0.7) > b(0.2, 0.05, 0.7)    # more adv. bio weight

    def test_bound_is_at_most_one(self):
        assert adversarial_weight_bound(1.0, 1.0, 0.0) <= 1.0


def _poisoning_config(**overrides) -> SimulationConfig:
    base = dict(
        num_clients=10,
        num_rounds=25,
        local_epochs=3,
        adversarial_fraction=0.2,
        attack_type="model_poisoning",
        use_dp=False,
        use_he=False,
        seed=42,
    )
    base.update(overrides)
    return SimulationConfig(**base)


@pytest.fixture(scope="module")
def poisoning_result():
    sim = FederatedLearningSimulator(_poisoning_config())
    return sim.run_tabfl()


class TestInfluenceMetric:
    """Per-round weight-share metric and its respect of the bound."""

    def test_metric_recorded_every_round(self, poisoning_result):
        for r in poisoning_result.rounds:
            m = r.aggregation_metrics
            assert "adversarial_weight_share" in m
            assert "adversarial_weight_bound" in m
            assert "adversarial_bio_share" in m
            assert 0.0 <= m["adversarial_weight_share"] <= 1.0

    def test_every_round_respects_bound(self, poisoning_result):
        for r in poisoning_result.rounds:
            m = r.aggregation_metrics
            assert m["adversarial_weight_share"] <= m["adversarial_weight_bound"] + 1e-9, (
                f"round {r.round_number}: share "
                f"{m['adversarial_weight_share']:.4f} exceeds bound "
                f"{m['adversarial_weight_bound']:.4f}"
            )

    def test_share_starts_near_participation_and_collapses(self, poisoning_result):
        first = poisoning_result.rounds[0].aggregation_metrics
        last = poisoning_result.rounds[-1].aggregation_metrics
        # round 1: all trusts equal -> adversaries carry ~their participation share
        assert first["adversarial_weight_share"] == pytest.approx(0.2, abs=0.08)
        # after detection + decay to the trust floor: well below half of it
        assert last["adversarial_weight_share"] < 0.03
        assert last["adversarial_weight_share"] < 0.5 * first["adversarial_weight_share"]

    def test_bound_tightens_after_detection(self, poisoning_result):
        first = poisoning_result.rounds[0].aggregation_metrics
        last = poisoning_result.rounds[-1].aggregation_metrics
        assert last["adversarial_weight_bound"] < 0.05
        assert first["adversarial_weight_bound"] > last["adversarial_weight_bound"]

    def test_clean_run_has_zero_adversarial_share(self):
        sim = FederatedLearningSimulator(_poisoning_config(adversarial_fraction=0.0))
        res = sim.run_tabfl()
        assert all(
            r.aggregation_metrics["adversarial_weight_share"] == 0.0
            for r in res.rounds
        )

    def test_fedavg_carries_participation_share(self):
        sim = FederatedLearningSimulator(_poisoning_config())
        res = sim.run_baseline("fedavg")
        shares = [r.aggregation_metrics["adversarial_weight_share"]
                  for r in res.rounds]
        # no trust attenuation: adversaries keep ~their sample-proportional share
        assert np.mean(shares) == pytest.approx(0.2, abs=0.08)

    def test_bio_only_ablation_has_no_attenuation(self):
        sim = FederatedLearningSimulator(
            _poisoning_config(use_bio=True, use_trust=False))
        res = sim.run_tabfl()
        last = res.rounds[-1].aggregation_metrics
        # without the trust layer gamma == alpha: share equals bio share
        assert last["adversarial_weight_share"] == pytest.approx(
            last["adversarial_bio_share"], rel=1e-9)
        assert last["adversarial_weight_share"] > 0.10
