"""
Federated Client Simulation Module
Simulates healthcare institution clients with heterogeneous multi-omics data.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


OMICS_TYPES = ["genomic", "proteomic", "clinical", "epigenomic", "transcriptomic"]

DATA_DISTRIBUTIONS = {
    "iid": "IID (Uniform)",
    "non_iid_mild": "Non-IID (Mild)",
    "non_iid_severe": "Non-IID (Severe)",
}

ATTACK_TYPES = {
    "none": "No Attack",
    "model_poisoning": "Model Poisoning",
    "label_flipping": "Label Flipping",
    "sybil": "Sybil Attack",
    "gradient_inversion": "Gradient Inversion",
}


@dataclass
class ClientConfig:
    client_id: int
    omics_type: str
    data_distribution: str = "non_iid_mild"
    num_samples: int = 500
    attack_type: str = "none"
    poison_rate: float = 0.0
    seed: Optional[int] = None


@dataclass
class LocalModelUpdate:
    client_id: int
    weights: np.ndarray
    gradients: np.ndarray
    loss: float
    accuracy: float
    num_samples: int
    omics_type: str
    bio_features: np.ndarray
    round_number: int
    is_malicious: bool = False
    noise_applied: bool = False


class FederatedClient:
    """
    Simulates a federated learning client representing a healthcare institution.
    Each client holds a local dataset with multi-omics features and trains
    a local model without sharing raw data.
    """

    def __init__(self, config: ClientConfig, global_model_dim: int = 64):
        self.config = config
        self.global_model_dim = global_model_dim
        self.rng = np.random.RandomState(config.seed or config.client_id * 42)

        self._generate_local_dataset()
        self._init_local_model()
        self.training_history: list[dict] = []

    def _generate_local_dataset(self):
        """Generate synthetic multi-omics healthcare dataset."""
        n = self.config.num_samples
        omics_idx = OMICS_TYPES.index(self.config.omics_type) if self.config.omics_type in OMICS_TYPES else 0

        # Raw feature dimensionality varies by omics type (mirrors real-world scale)
        feature_dims = {"genomic": 128, "proteomic": 96, "clinical": 32, "epigenomic": 80, "transcriptomic": 112}
        raw_dim = feature_dims.get(self.config.omics_type, 64)

        # Generate class distribution based on data heterogeneity setting
        if self.config.data_distribution == "iid":
            class_probs = [0.5, 0.5]
        elif self.config.data_distribution == "non_iid_mild":
            skew = 0.2 * (omics_idx / max(len(OMICS_TYPES) - 1, 1))
            class_probs = [0.5 + skew, 0.5 - skew]
        else:  # non_iid_severe
            skew = 0.4 * (omics_idx / max(len(OMICS_TYPES) - 1, 1))
            class_probs = [0.5 + skew, 0.5 - skew]

        self.labels = self.rng.choice([0, 1], size=n, p=class_probs)

        # Generate raw features with class-conditional distributions
        raw_features = np.zeros((n, raw_dim))
        for cls in [0, 1]:
            mask = self.labels == cls
            mean = self.rng.randn(raw_dim) * 0.5 + cls * 0.8
            cov_scale = 0.3 + 0.2 * omics_idx / max(len(OMICS_TYPES) - 1, 1)
            raw_features[mask] = self.rng.randn(mask.sum(), raw_dim) * cov_scale + mean

        # Normalize raw features
        raw_features = (raw_features - raw_features.mean(0)) / (raw_features.std(0) + 1e-8)

        # Project features to global model dimension using a fixed random projection
        # (simulates a shared model backbone — all clients have same input dim to global model)
        if raw_dim != self.global_model_dim:
            proj = self.rng.randn(raw_dim, self.global_model_dim) / np.sqrt(raw_dim)
            self.features = raw_features @ proj
        else:
            self.features = raw_features

        # Re-normalize projected features
        self.features = (self.features - self.features.mean(0)) / (self.features.std(0) + 1e-8)

        # Biological feature signature (used for bio-aware similarity) — from raw features
        self.bio_signature = self._compute_bio_signature(raw_features)

        # Apply label flipping attack
        if self.config.attack_type == "label_flipping" and self.config.poison_rate > 0:
            n_poison = int(n * self.config.poison_rate)
            poison_idx = self.rng.choice(n, n_poison, replace=False)
            self.labels[poison_idx] = 1 - self.labels[poison_idx]

    def _compute_bio_signature(self, raw_features: np.ndarray) -> np.ndarray:
        """Compute a biological feature signature from raw omics features."""
        pathway_dim = 16
        base_sig = self.rng.randn(pathway_dim) * 0.3
        # Omics-type-specific bias encodes domain knowledge about pathway relevance
        omics_bias = np.array([
            [0.9, 0.1, 0.1, 0.3, 0.2],   # genomic
            [0.1, 0.9, 0.2, 0.1, 0.3],   # proteomic
            [0.2, 0.2, 0.9, 0.1, 0.1],   # clinical
            [0.3, 0.1, 0.1, 0.9, 0.4],   # epigenomic
            [0.2, 0.3, 0.1, 0.4, 0.9],   # transcriptomic
        ], dtype=float)

        omics_idx = OMICS_TYPES.index(self.config.omics_type)
        bio_basis = np.tile(omics_bias[omics_idx], pathway_dim // len(OMICS_TYPES) + 1)[:pathway_dim]
        # Add a data-driven component: mean activation of first pathway_dim features
        n_feat = min(pathway_dim, raw_features.shape[1])
        data_sig = raw_features[:, :n_feat].mean(axis=0)
        if len(data_sig) < pathway_dim:
            data_sig = np.pad(data_sig, (0, pathway_dim - len(data_sig)))
        return base_sig + bio_basis + 0.2 * data_sig

    def _init_local_model(self):
        """Initialize local logistic regression-style model weights."""
        feature_dim = self.features.shape[1]
        self.weights = self.rng.randn(feature_dim) * 0.01
        self.bias = 0.0

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _compute_loss_and_grad(self, features, labels, weights, bias, l2_reg=0.01):
        """Binary cross-entropy loss with L2 regularization."""
        logits = features @ weights + bias
        preds = self._sigmoid(logits)
        eps = 1e-12
        loss = -np.mean(labels * np.log(preds + eps) + (1 - labels) * np.log(1 - preds + eps))
        loss += 0.5 * l2_reg * np.dot(weights, weights)

        error = preds - labels
        grad_w = features.T @ error / len(labels) + l2_reg * weights
        grad_b = error.mean()
        return loss, grad_w, grad_b

    def local_train(
        self,
        global_weights: np.ndarray,
        global_bias: float,
        round_number: int,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        dp_epsilon: Optional[float] = None,
    ) -> LocalModelUpdate:
        """
        Perform local training on private data using global model as starting point.
        Features are already projected to global_model_dim so shapes always match.
        Optionally apply differential privacy noise to gradients.
        """
        # Initialize from global model (features and weights share same dim)
        weights = global_weights.copy()  # shape: (global_model_dim,)
        bias = global_bias

        # Mini-batch SGD
        batch_size = min(64, len(self.labels))
        for _ in range(local_epochs):
            idx = self.rng.permutation(len(self.labels))
            for start in range(0, len(self.labels), batch_size):
                batch_idx = idx[start:start + batch_size]
                X_batch = self.features[batch_idx]
                y_batch = self.labels[batch_idx]
                loss, grad_w, grad_b = self._compute_loss_and_grad(X_batch, y_batch, weights, bias)
                weights -= learning_rate * grad_w
                bias -= learning_rate * grad_b

        # Compute final metrics on full local dataset
        final_loss, _, _ = self._compute_loss_and_grad(self.features, self.labels, weights, bias)
        logits = self.features @ weights + bias
        preds = (self._sigmoid(logits) >= 0.5).astype(int)
        accuracy = float((preds == self.labels).mean())

        # Apply model poisoning attack (corrupt weights before sending)
        is_malicious = False
        if self.config.attack_type == "model_poisoning" and self.config.poison_rate > 0:
            poison_scale = 3.0 + self.config.poison_rate * 5.0
            weights += self.rng.randn(*weights.shape) * poison_scale
            is_malicious = True

        # Sybil attack: fake identities submit amplified erratic updates
        if self.config.attack_type == "sybil" and self.config.poison_rate > 0:
            sybil_scale = 2.0 + self.config.poison_rate * 4.0
            weights += self.rng.randn(*weights.shape) * sybil_scale
            is_malicious = True

        # Gradients = weight delta from global (same dimension, no projection needed)
        gradients = np.append(weights - global_weights, bias - global_bias)

        # Apply differential privacy noise (Gaussian mechanism)
        noise_applied = False
        if dp_epsilon is not None:
            sensitivity = 2.0 / len(self.labels)
            dp_noise_scale = sensitivity / dp_epsilon
            gradients += self.rng.randn(*gradients.shape) * dp_noise_scale
            weights += self.rng.randn(*weights.shape) * (dp_noise_scale * 0.5)
            noise_applied = True

        update = LocalModelUpdate(
            client_id=self.config.client_id,
            weights=weights,                  # shape: (global_model_dim,)
            gradients=gradients,              # shape: (global_model_dim + 1,)
            loss=float(final_loss),
            accuracy=accuracy,
            num_samples=len(self.labels),
            omics_type=self.config.omics_type,
            bio_features=self.bio_signature,
            round_number=round_number,
            is_malicious=is_malicious,
            noise_applied=noise_applied,
        )

        self.training_history.append({
            "round": round_number,
            "loss": float(final_loss),
            "accuracy": accuracy,
        })

        return update

    @property
    def is_adversarial(self) -> bool:
        return self.config.attack_type != "none"
