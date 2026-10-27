"""Tests for the federated client module."""

import numpy as np
import pytest

from framework.clients import FederatedClient, ClientConfig, OMICS_TYPES


class TestFederatedClient:
    def test_initialization(self, client):
        assert client.global_model_dim == 32
        assert client.features.shape == (200, 32)
        assert client.labels.shape == (200,)
        assert client.bio_signature.shape == (16,)

    def test_all_omics_types_supported(self):
        for omics in OMICS_TYPES:
            config = ClientConfig(client_id=0, omics_type=omics, num_samples=50)
            c = FederatedClient(config, global_model_dim=16)
            assert c.features.shape == (50, 16)
            assert c.config.omics_type == omics

    def test_local_train_returns_update(self, client):
        update = client.local_train(
            global_weights=np.zeros(32),
            global_bias=0.0,
            round_number=1,
            local_epochs=2,
        )
        assert update.client_id == 0
        assert update.weights.shape == (32,)
        assert update.gradients.shape[0] == 32 or update.gradients.shape[0] == 33
        assert 0.0 <= update.accuracy <= 1.0
        assert update.loss >= 0.0
        assert update.num_samples == 200

    def test_training_history_recorded(self, client):
        client.local_train(np.zeros(32), 0.0, 1, 2)
        assert len(client.training_history) == 1
        assert client.training_history[0]["round"] == 1

    def test_label_flipping_attack(self):
        config = ClientConfig(
            client_id=0,
            omics_type="genomic",
            num_samples=100,
            attack_type="label_flipping",
            poison_rate=0.5,
            seed=42,
        )
        c = FederatedClient(config, global_model_dim=16)
        # With 50% poison rate, roughly half labels should be flipped
        assert c.labels is not None

    def test_model_poisoning_attack(self):
        config = ClientConfig(
            client_id=0,
            omics_type="genomic",
            num_samples=100,
            attack_type="model_poisoning",
            poison_rate=0.3,
            seed=42,
        )
        c = FederatedClient(config, global_model_dim=16)
        update = c.local_train(np.zeros(16), 0.0, 1, 2)
        assert update.is_malicious is True

    def test_is_adversarial_property(self):
        benign = ClientConfig(client_id=0, omics_type="genomic", attack_type="none")
        c_benign = FederatedClient(benign, 16)
        assert c_benign.is_adversarial is False

        adversarial = ClientConfig(client_id=1, omics_type="genomic", attack_type="model_poisoning")
        c_adv = FederatedClient(adversarial, 16)
        assert c_adv.is_adversarial is True

    def test_iid_data_distribution(self):
        config = ClientConfig(
            client_id=0, omics_type="genomic",
            num_samples=500, data_distribution="iid", seed=42,
        )
        c = FederatedClient(config, 16)
        class_ratio = c.labels.mean()
        assert 0.3 < class_ratio < 0.7

    def test_non_iid_severe_distribution(self):
        config = ClientConfig(
            client_id=0, omics_type="transcriptomic",
            num_samples=500, data_distribution="non_iid_severe", seed=42,
        )
        c = FederatedClient(config, 16)
        # Severe Non-IID should have more skewed class distribution
        class_ratio = c.labels.mean()
        assert 0.0 < class_ratio < 1.0
