"""Tests for the bio-aware aggregation module."""

import numpy as np
import pytest

from framework.bio_aware import BioAwareAggregator, OMICS_COMPATIBILITY_MATRIX
from framework.clients import LocalModelUpdate


class TestBioAwareAggregator:
    @pytest.fixture
    def aggregator(self):
        return BioAwareAggregator(
            bio_weight=0.4, sample_weight=0.3, quality_weight=0.3, temperature=1.0
        )

    @pytest.fixture
    def sample_updates(self):
        rng = np.random.RandomState(42)
        updates = []
        omics = ["genomic", "proteomic", "clinical"]
        for i in range(3):
            updates.append(LocalModelUpdate(
                client_id=i,
                weights=rng.randn(32) * 0.01,
                gradients=rng.randn(33) * 0.01,
                loss=0.5 + rng.randn() * 0.1,
                accuracy=0.7 + rng.randn() * 0.05,
                num_samples=200,
                omics_type=omics[i],
                bio_features=rng.randn(16),
                round_number=1,
            ))
        return updates

    def test_pathway_similarity_symmetric(self, aggregator, sample_updates):
        features = [u.bio_features for u in sample_updates]
        sim = aggregator.compute_pathway_similarity(features)
        assert sim.shape == (3, 3)
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)
        np.testing.assert_array_almost_equal(np.diag(sim), np.ones(3))

    def test_pathway_similarity_range(self, aggregator, sample_updates):
        features = [u.bio_features for u in sample_updates]
        sim = aggregator.compute_pathway_similarity(features)
        assert sim.min() >= -1.0
        assert sim.max() <= 1.0

    def test_omics_compatibility(self, aggregator):
        types = ["genomic", "proteomic", "genomic"]
        compat = aggregator.compute_omics_compatibility(types)
        assert compat.shape == (3, 3)
        assert compat[0, 0] == 1.0
        assert compat[0, 2] == 1.0  # same type
        assert 0.0 <= compat[0, 1] <= 1.0

    def test_quality_scores_range(self, aggregator, sample_updates):
        scores = aggregator.compute_quality_scores(sample_updates)
        assert scores.shape == (3,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_compute_bio_weights_sum_to_one(self, aggregator, sample_updates):
        weights, metrics = aggregator.compute_bio_weights(sample_updates)
        assert weights.shape == (3,)
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)
        assert "bio_similarity_matrix" in metrics
        assert "sample_weights" in metrics
        assert "quality_scores" in metrics

    def test_aggregate_returns_global_weights(self, aggregator, sample_updates):
        trust = np.ones(3) / 3
        weights, agg_metrics = aggregator.aggregate(sample_updates, trust)
        assert weights.shape == (32,)
        assert "combined_weights" in agg_metrics

    def test_aggregate_with_zero_trust(self, aggregator, sample_updates):
        trust = np.zeros(3)
        weights, agg_metrics = aggregator.aggregate(sample_updates, trust)
        assert weights.shape == (32,)
        # Combined weights should still be valid after normalization
        assert agg_metrics["combined_weights"].sum() >= 0

    def test_similarity_history_stored(self, aggregator, sample_updates):
        aggregator.aggregate(sample_updates, np.ones(3) / 3)
        aggregator.aggregate(sample_updates, np.ones(3) / 3)
        assert len(aggregator.similarity_history) == 2

    def test_omics_compatibility_matrix_shape(self):
        assert OMICS_COMPATIBILITY_MATRIX.shape == (5, 5)
        np.testing.assert_array_almost_equal(np.diag(OMICS_COMPATIBILITY_MATRIX), np.ones(5))
