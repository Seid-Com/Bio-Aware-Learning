"""
Adversarial Attack Simulation Module
Simulates various attacks on the federated learning system and measures
the resilience of the TA-BC-BFL framework vs. baselines.
"""

import numpy as np
from typing import List, Dict, Optional
from .clients import LocalModelUpdate


class AdversarialAttackSimulator:
    """
    Simulates adversarial attacks on federated learning systems.
    Tests model poisoning, Sybil attacks, inference attacks, and backdoors.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.attack_log: List[Dict] = []

    def model_poisoning_attack(
        self,
        update: LocalModelUpdate,
        scale: float = 5.0,
        strategy: str = "sign_flip",
    ) -> LocalModelUpdate:
        """
        Model poisoning: malicious client sends corrupted gradients.
        Strategies:
        - sign_flip: flip gradient signs
        - scale_up: amplify gradients by large factor
        - noise: replace with random noise
        """
        poisoned = LocalModelUpdate(
            client_id=update.client_id,
            weights=update.weights.copy(),
            gradients=update.gradients.copy(),
            loss=update.loss,
            accuracy=update.accuracy,
            num_samples=update.num_samples,
            omics_type=update.omics_type,
            bio_features=update.bio_features.copy(),
            round_number=update.round_number,
            is_malicious=True,
        )

        if strategy == "sign_flip":
            poisoned.gradients = -update.gradients * scale
            poisoned.weights = -update.weights * scale
        elif strategy == "scale_up":
            poisoned.gradients = update.gradients * scale * 10
            poisoned.weights = update.weights + self.rng.randn(*update.weights.shape) * scale
        else:  # noise
            poisoned.gradients = self.rng.randn(*update.gradients.shape) * scale
            poisoned.weights = self.rng.randn(*update.weights.shape) * scale

        self.attack_log.append({
            "type": "model_poisoning",
            "client_id": update.client_id,
            "round": update.round_number,
            "strategy": strategy,
            "scale": scale,
        })
        return poisoned

    def sybil_attack(
        self,
        legitimate_update: LocalModelUpdate,
        num_sybils: int = 3,
    ) -> List[LocalModelUpdate]:
        """
        Sybil attack: one malicious party creates multiple fake identities
        to gain disproportionate influence.
        """
        sybils = []
        for i in range(num_sybils):
            sybil_id = legitimate_update.client_id * 100 + i + 1
            noise = self.rng.randn(*legitimate_update.weights.shape) * 0.3
            sybil = LocalModelUpdate(
                client_id=sybil_id,
                weights=legitimate_update.weights + noise,
                gradients=legitimate_update.gradients + self.rng.randn(*legitimate_update.gradients.shape) * 0.2,
                loss=legitimate_update.loss * (1 + self.rng.randn() * 0.1),
                accuracy=legitimate_update.accuracy * (0.95 + self.rng.randn() * 0.02),
                num_samples=legitimate_update.num_samples,
                omics_type=legitimate_update.omics_type,
                bio_features=legitimate_update.bio_features + self.rng.randn(*legitimate_update.bio_features.shape) * 0.1,
                round_number=legitimate_update.round_number,
                is_malicious=True,
            )
            sybils.append(sybil)

        self.attack_log.append({
            "type": "sybil",
            "client_id": legitimate_update.client_id,
            "round": legitimate_update.round_number,
            "num_sybils": num_sybils,
        })
        return sybils

    def gradient_inversion_attempt(
        self,
        update: LocalModelUpdate,
        iterations: int = 100,
        clean_gradients: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Simulate a gradient inversion (model inversion) attack.
        Attempts to reconstruct private training data from shared gradients.
        Returns estimated data leakage metrics.

        The attacker converges toward the *observed* (possibly noisy) gradient,
        but reconstruction quality is measured against the clean pre-noise
        gradient (what the attacker actually wants to recover), so DP noise
        lowers the score. Without DP the score approaches 1.0.
        """
        # Simulate iterative reconstruction quality
        reconstruction_quality = []
        target_grad = update.gradients
        reference = target_grad if clean_gradients is None else clean_gradients
        reconstructed = self.rng.randn(*target_grad.shape)

        lr = 0.1
        for it in range(iterations):
            grad_diff = reconstructed - target_grad
            reconstructed -= lr * grad_diff
            # Quality = 1 - normalized distance to the CLEAN gradient
            dist = np.linalg.norm(reconstructed - reference) / (np.linalg.norm(reference) + 1e-8)
            reconstruction_quality.append(1.0 - min(dist, 1.0))

        final_quality = float(reconstruction_quality[-1])

        self.attack_log.append({
            "type": "gradient_inversion",
            "client_id": update.client_id,
            "round": update.round_number,
            "reconstruction_quality": final_quality,
        })

        return {
            "reconstruction_quality_history": reconstruction_quality,
            "final_quality": final_quality,
            "data_leakage_risk": "High" if final_quality > 0.8 else "Medium" if final_quality > 0.5 else "Low",
            "effective_with_dp": final_quality < 0.4,
        }

    def backdoor_attack(
        self,
        update: LocalModelUpdate,
        trigger_strength: float = 0.5,
    ) -> LocalModelUpdate:
        """
        Backdoor attack: embed a trigger pattern that causes
        misclassification on triggered inputs while normal accuracy is maintained.
        """
        backdoored = LocalModelUpdate(
            client_id=update.client_id,
            weights=update.weights.copy(),
            gradients=update.gradients.copy(),
            loss=update.loss,
            accuracy=update.accuracy,
            num_samples=update.num_samples,
            omics_type=update.omics_type,
            bio_features=update.bio_features.copy(),
            round_number=update.round_number,
            is_malicious=True,
        )

        # Add subtle directional bias to weights
        trigger_direction = self.rng.randn(*update.weights.shape)
        trigger_direction /= np.linalg.norm(trigger_direction) + 1e-8
        backdoored.weights = update.weights + trigger_strength * trigger_direction

        self.attack_log.append({
            "type": "backdoor",
            "client_id": update.client_id,
            "round": update.round_number,
            "trigger_strength": trigger_strength,
        })
        return backdoored

    def measure_attack_success(
        self,
        clean_global_weights: np.ndarray,
        attacked_global_weights: np.ndarray,
        baseline_accuracy: float,
    ) -> Dict:
        """
        Measure the effectiveness of an attack by comparing
        clean vs. attacked global model parameters.
        """
        weight_deviation = np.linalg.norm(attacked_global_weights - clean_global_weights)
        relative_deviation = weight_deviation / (np.linalg.norm(clean_global_weights) + 1e-8)

        # Estimate accuracy degradation from weight deviation
        # (simplified linear model)
        estimated_acc_drop = min(relative_deviation * 0.3, baseline_accuracy * 0.5)
        attacked_accuracy = max(0.0, baseline_accuracy - estimated_acc_drop)

        return {
            "weight_deviation": float(weight_deviation),
            "relative_deviation": float(relative_deviation),
            "baseline_accuracy": float(baseline_accuracy),
            "attacked_accuracy": float(attacked_accuracy),
            "accuracy_drop": float(estimated_acc_drop),
            "attack_success_rate": float(min(relative_deviation * 2, 1.0)),
        }

    def get_attack_summary(self) -> Dict:
        """Summarize all attacks launched in this simulation."""
        if not self.attack_log:
            return {"total_attacks": 0, "by_type": {}}

        by_type: Dict[str, int] = {}
        for entry in self.attack_log:
            atype = entry["type"]
            by_type[atype] = by_type.get(atype, 0) + 1

        return {
            "total_attacks": len(self.attack_log),
            "by_type": by_type,
            "affected_rounds": list(set(e["round"] for e in self.attack_log)),
            "affected_clients": list(set(e["client_id"] for e in self.attack_log)),
        }
