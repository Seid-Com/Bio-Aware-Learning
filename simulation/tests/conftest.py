"""Shared fixtures for simulation tests."""

import pytest
import numpy as np

from framework.clients import FederatedClient, ClientConfig, OMICS_TYPES
from framework.bio_aware import BioAwareAggregator, OMICS_COMPATIBILITY_MATRIX
from framework.blockchain import BlockchainTrustManager
from framework.privacy import DifferentialPrivacy, HomomorphicEncryption, PrivacyAccountant
from framework.attacks import AdversarialAttackSimulator
from framework.federated_learning import FederatedLearningSimulator, SimulationConfig


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_config():
    return SimulationConfig(
        num_clients=5,
        num_rounds=3,
        local_epochs=2,
        learning_rate=0.01,
        fraction_clients=1.0,
        global_model_dim=32,
        bio_weight=0.4,
        sample_weight=0.3,
        quality_weight=0.3,
        use_dp=True,
        dp_epsilon=2.0,
        dp_delta=1e-5,
        use_he=False,
        adversarial_fraction=0.0,
        num_samples_per_client=100,
        data_distribution="non_iid_mild",
    )


@pytest.fixture
def client_config():
    return ClientConfig(
        client_id=0,
        omics_type="genomic",
        data_distribution="non_iid_mild",
        num_samples=200,
        attack_type="none",
        seed=42,
    )


@pytest.fixture
def client(client_config):
    return FederatedClient(client_config, global_model_dim=32)


@pytest.fixture
def clients():
    configs = []
    for i in range(5):
        configs.append(ClientConfig(
            client_id=i,
            omics_type=OMICS_TYPES[i % len(OMICS_TYPES)],
            num_samples=200,
            seed=i * 42,
        ))
    return [FederatedClient(c, global_model_dim=32) for c in configs]


@pytest.fixture
def sample_update(client):
    client.local_train(
        global_weights=np.zeros(32),
        global_bias=0.0,
        round_number=1,
        local_epochs=2,
    )
    return client.training_history[-1]
