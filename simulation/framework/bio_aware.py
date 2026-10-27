"""
Bio-Aware Aggregation Module
Computes multi-omics biological similarity between federated clients
and uses it to weight model updates during aggregation.
"""

import numpy as np
from typing import List, Dict, Tuple
from .clients import LocalModelUpdate, OMICS_TYPES


OMICS_COMPATIBILITY_MATRIX = np.array([
    # genomic  proteomic  clinical  epigenomic  transcriptomic
    [1.00,     0.65,      0.45,     0.80,       0.75],  # genomic
    [0.65,     1.00,      0.50,     0.60,       0.70],  # proteomic
    [0.45,     0.50,      1.00,     0.40,       0.45],  # clinical
    [0.80,     0.60,      0.40,     1.00,       0.85],  # epigenomic
    [0.75,     0.70,      0.45,     0.85,       1.00],  # transcriptomic
], dtype=float)


def adversarial_weight_bound(
    adversarial_bio_share: float,
    adversarial_trust_max: float,
    honest_trust_min: float,
) -> float:
    """
    Exact upper bound on the combined aggregation weight that adversarial
    clients can obtain in a single round (Proposition 1, Appendix A.2).

    With combined weights gamma_i = alpha_i * beta_i / sum_j(alpha_j * beta_j),
    let s = sum_{i in Adv} alpha_i be the adversaries' total bio-aware weight,
    suppose every adversarial trust score is at most beta_A, and every honest
    trust score is at least beta_H.  Writing the aggregation as
    W_Adv = N / (N + U) with N = sum_Adv alpha_i * beta_i and
    U = sum_Hon alpha_j * beta_j, we have N <= s * beta_A (monotone
    increasing) and U >= (1 - s) * beta_H (monotone decreasing), hence

        W_Adv <= s * beta_A / (s * beta_A + (1 - s) * beta_H).

    With the deployed parameters (s ~ f = 0.2, beta_A = min_trust = 0.05
    once flagged adversaries reach the trust floor, beta_H = initial trust
    0.7 for unflagged honest clients) this evaluates to
    0.01 / (0.01 + 0.56) ~ 1.75%, and to 0.01 / (0.01 + 0.80) ~ 1.23%
    once honest trust converges to max_trust = 1.0 — versus an
    unattenuated participation share of 20%.

    Parameters
    ----------
    adversarial_bio_share : float
        s = total alpha weight of the adversarial participants this round.
    adversarial_trust_max : float
        Upper bound beta_A on adversarial trust scores this round.
    honest_trust_min : float
        Lower bound beta_H on honest trust scores this round.

    Returns
    -------
    float
        Upper bound in [0, 1] on the adversaries' combined aggregation weight.
    """
    s = float(np.clip(adversarial_bio_share, 0.0, 1.0))
    numerator = s * float(adversarial_trust_max)
    denominator = numerator + (1.0 - s) * float(honest_trust_min)
    if denominator <= 0.0:
        return 0.0
    return float(min(numerator / denominator, 1.0))


class BioAwareAggregator:
    """
    Implements the Bio-Aware Federated Aggregation mechanism from the TA-BC-BFL paper.

    The aggregation incorporates:
    1. Multi-omics pathway similarity (cosine similarity of biological signatures)
    2. Omics-type compatibility scores (domain knowledge matrix)
    3. Data quality scores (based on loss & gradient norms)
    4. Hybrid weighting combining bio-similarity with sample count
    """

    def __init__(
        self,
        bio_weight: float = 0.4,
        sample_weight: float = 0.3,
        quality_weight: float = 0.3,
        temperature: float = 1.0,
    ):
        self.bio_weight = bio_weight
        self.sample_weight = sample_weight
        self.quality_weight = quality_weight
        self.temperature = temperature
        self.similarity_history: List[np.ndarray] = []

    def compute_pathway_similarity(self, bio_features: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise cosine similarity of biological pathway signatures.
        Higher similarity → clients share more biological context.
        """
        n = len(bio_features)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                fi, fj = bio_features[i], bio_features[j]
                norm_i = np.linalg.norm(fi)
                norm_j = np.linalg.norm(fj)
                if norm_i > 0 and norm_j > 0:
                    sim_matrix[i, j] = np.dot(fi, fj) / (norm_i * norm_j)
                else:
                    sim_matrix[i, j] = 0.0
        return sim_matrix

    def compute_omics_compatibility(self, omics_types: List[str]) -> np.ndarray:
        """
        Use domain knowledge matrix to score compatibility between omics modalities.
        """
        n = len(omics_types)
        compat_matrix = np.zeros((n, n))
        for i, t1 in enumerate(omics_types):
            for j, t2 in enumerate(omics_types):
                idx1 = OMICS_TYPES.index(t1) if t1 in OMICS_TYPES else 0
                idx2 = OMICS_TYPES.index(t2) if t2 in OMICS_TYPES else 0
                compat_matrix[i, j] = OMICS_COMPATIBILITY_MATRIX[idx1, idx2]
        return compat_matrix

    def compute_quality_scores(self, updates: List[LocalModelUpdate]) -> np.ndarray:
        """
        Score model quality based on local loss, accuracy, and gradient norm.
        Lower loss + higher accuracy + reasonable gradient norm → higher quality.
        """
        losses = np.array([u.loss for u in updates])
        accuracies = np.array([u.accuracy for u in updates])
        grad_norms = np.array([np.linalg.norm(u.gradients) for u in updates])

        # Normalize each component to [0, 1]
        loss_scores = 1.0 - (losses - losses.min()) / ((losses.max() - losses.min()) + 1e-8)
        acc_scores = (accuracies - accuracies.min()) / ((accuracies.max() - accuracies.min()) + 1e-8)

        # Prefer moderate gradient norms (too large may indicate poisoning)
        grad_median = np.median(grad_norms)
        grad_deviation = np.abs(grad_norms - grad_median) / (grad_median + 1e-8)
        grad_scores = np.exp(-grad_deviation)

        quality = 0.4 * loss_scores + 0.4 * acc_scores + 0.2 * grad_scores
        return quality

    def compute_bio_weights(
        self, updates: List[LocalModelUpdate]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute final aggregation weights incorporating biological similarity,
        sample size, and model quality.

        Returns:
            weights: normalized weight for each client update
            metrics: detailed breakdown of weight components
        """
        n = len(updates)
        bio_features = [u.bio_features for u in updates]
        omics_types = [u.omics_type for u in updates]
        sample_counts = np.array([u.num_samples for u in updates], dtype=float)

        # Component 1: Pathway similarity (average similarity to all peers)
        pathway_sim = self.compute_pathway_similarity(bio_features)
        omics_compat = self.compute_omics_compatibility(omics_types)
        bio_sim_matrix = 0.6 * pathway_sim + 0.4 * omics_compat
        avg_bio_sim = bio_sim_matrix.mean(axis=1)

        # Store for visualization
        self.similarity_history.append(bio_sim_matrix)

        # Component 2: Sample-based weights (standard FedAvg component)
        sample_weights = sample_counts / sample_counts.sum()

        # Component 3: Model quality scores
        quality_scores = self.compute_quality_scores(updates)

        # Combine components
        raw_weights = (
            self.bio_weight * avg_bio_sim
            + self.sample_weight * sample_weights
            + self.quality_weight * quality_scores
        )

        # Apply temperature softmax for smooth weighting
        raw_weights /= self.temperature
        weights = np.exp(raw_weights - raw_weights.max())
        weights /= weights.sum()

        metrics = {
            "bio_similarity_matrix": bio_sim_matrix,
            "pathway_similarity": pathway_sim,
            "omics_compatibility": omics_compat,
            "avg_bio_similarity": avg_bio_sim,
            "sample_weights": sample_weights,
            "quality_scores": quality_scores,
            "final_weights": weights,
        }

        return weights, metrics

    def aggregate(
        self, updates: List[LocalModelUpdate], trust_scores: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform bio-aware federated aggregation.

        Combines bio-aware weights with blockchain trust scores to produce
        the final aggregated global model.
        """
        bio_weights, bio_metrics = self.compute_bio_weights(updates)

        # Combine bio-awareness with trust scores
        combined_weights = bio_weights * trust_scores
        if combined_weights.sum() > 0:
            combined_weights /= combined_weights.sum()

        # Weighted average of model weights
        model_dim = updates[0].weights.shape[0]
        global_weights = np.zeros(model_dim)
        for i, update in enumerate(updates):
            global_weights += combined_weights[i] * update.weights

        combined_sq_sum = float((combined_weights**2).sum())
        aggregation_metrics = {
            **bio_metrics,
            "trust_scores": trust_scores,
            "combined_weights": combined_weights,
            # participation diversity; guard against zero-weight (all trust=0) case
            "effective_clients": float(1.0 / combined_sq_sum) if combined_sq_sum > 0 else 0.0,
        }

        return global_weights, aggregation_metrics

    def get_omics_type_distribution(self, updates: List[LocalModelUpdate]) -> Dict[str, int]:
        """Count clients by omics type."""
        dist: Dict[str, int] = {}
        for u in updates:
            dist[u.omics_type] = dist.get(u.omics_type, 0) + 1
        return dist
