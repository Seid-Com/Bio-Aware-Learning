"""
Privacy-Preserving Mechanisms Module
Implements Differential Privacy (Gaussian mechanism) and a simulation of
Homomorphic Encryption for the TA-BC-BFL framework.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


def advanced_composition_total(
    round_epsilon: float, round_delta: float, rounds: int
) -> Tuple[float, float]:
    """Total (epsilon, delta) after `rounds` releases under advanced composition.

    For a sequence of k mechanisms that are each (eps_r, delta_r)-DP, for any
    delta' > 0 the composition is (eps_tot, delta_tot)-DP with

        eps_tot  = eps_r * sqrt(2 k ln(1/delta')) + k eps_r (exp(eps_r) - 1)
        delta_tot = k delta_r + k delta_r (exp(eps_r) - 1)      (delta' = delta_r)

    This is the accounting used by :class:`PrivacyAccountant`.
    """
    k = int(rounds)
    if k <= 0:
        return 0.0, 0.0
    delta_prime = max(float(round_delta), 1e-300)
    eps = float(round_epsilon)
    eps_tot = eps * np.sqrt(2.0 * k * np.log(1.0 / delta_prime)) + k * eps * (
        np.exp(eps) - 1.0
    )
    delta_tot = k * delta_prime + k * delta_prime * (np.exp(eps) - 1.0)
    if not np.isfinite(eps_tot):
        eps_tot = float(np.finfo(float).max)
    return float(eps_tot), float(min(delta_tot, 1.0))


def rdp_total_epsilon(
    noise_multiplier: float, rounds: int, delta: float, alpha_grid: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Tight-ish total epsilon for `rounds` Gaussian-mechanism releases via RDP.

    The Gaussian mechanism with noise/sensitivity ratio sigma is
    (alpha, alpha / (2 sigma^2))-RDP for every order alpha > 1.  RDP composes
    additively over adaptive releases, so after k rounds the RDP guarantee is
    (alpha, k * alpha / (2 sigma^2)); converting to (epsilon, delta)-DP gives

        eps(delta) = min_{alpha > 1} [ k alpha / (2 sigma^2) + ln(1/delta)/(alpha-1) ]

    which is markedly tighter than elementary/advanced composition for
    noise multipliers of the size used in this work (sigma ~= 4.85).

    Returns (epsilon_total, alpha_optimal).
    """
    sigma2 = float(noise_multiplier) ** 2
    k = int(rounds)
    if k <= 0 or sigma2 <= 0:
        return 0.0, 1.0
    c = k / (2.0 * sigma2)           # eps_alpha = c * alpha
    if alpha_grid is None:
        # Interior optimum sits at alpha* = 1 + sqrt(ln(1/delta) / c); search
        # a generous window around it (log-spaced, plus a dense local band).
        alpha0 = 1.0 + np.sqrt(np.log(1.0 / max(delta, 1e-300)) / max(c, 1e-12))
        alpha_grid = np.unique(
            np.concatenate(
                [
                    np.linspace(1.0 + 1e-6, max(4.0 * alpha0, 4.0), 4000),
                    np.linspace(max(alpha0 - 50.0, 1.0 + 1e-6), alpha0 + 50.0, 4000),
                ]
            )
        )
    eps = c * alpha_grid + np.log(1.0 / max(delta, 1e-300)) / (alpha_grid - 1.0)
    i = int(np.argmin(eps))
    return float(eps[i]), float(alpha_grid[i])


@dataclass
class PrivacyAccountant:
    """Tracks the cumulative privacy budget spent over rounds using advanced composition."""
    epsilon: float
    delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    rounds: int = 0

    def update(self, round_epsilon: float, round_delta: float) -> None:
        """Update privacy budget using advanced composition theorem.

        Uses the advanced composition theorem which provides tighter bounds
        than simple additive composition for (epsilon, delta)-DP:
        - Simple: eps_total = k * eps
        - Advanced: eps_total = eps * sqrt(2k * ln(1/delta')) + k * eps * (e^eps - 1)
          with delta' = round_delta. The second term is required by the
          theorem; dropping it understates the composed budget by
          k * eps * (e^eps - 1) (171.8 for the deployed k=100, eps=1).
        """
        self.rounds += 1
        advanced_epsilon, advanced_delta = advanced_composition_total(
            round_epsilon, round_delta, self.rounds
        )

        self.spent_epsilon = advanced_epsilon
        self.spent_delta = advanced_delta

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.epsilon - self.spent_epsilon)

    @property
    def budget_fraction_used(self) -> float:
        return min(1.0, self.spent_epsilon / self.epsilon)

    @property
    def is_budget_exhausted(self) -> bool:
        return self.spent_epsilon >= self.epsilon


class DifferentialPrivacy:
    """
    Differentially Private Gradient Perturbation (Gaussian Mechanism).

    Provides (ε, δ)-differential privacy guarantees by adding calibrated
    Gaussian noise to gradients before transmission.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        seed: int = 42,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.accountant = PrivacyAccountant(epsilon=epsilon, delta=delta)
        # Seeded noise stream: the global np.random state is seeded from OS
        # entropy per process, which made DP runs irreproducible across
        # invocations. An instance-seeded RandomState keeps runs bit-stable.
        self._rng = np.random.RandomState(seed)

        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            # Calibrate noise multiplier from (ε, δ) — simplified formula
            self.noise_multiplier = self._calibrate_noise(epsilon, delta)

        self.noise_history: List[Dict] = []

    def _calibrate_noise(self, epsilon: float, delta: float) -> float:
        """Approximate noise multiplier calibration from (ε, δ) parameters."""
        # Rough analytic calibration: σ ≈ sqrt(2 * ln(1.25/δ)) / ε
        if epsilon <= 0:
            return 10.0
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return max(0.01, sigma)

    def clip_gradients(self, gradients: np.ndarray) -> Tuple[np.ndarray, float]:
        """Clip gradients to bounded L2 norm (sensitivity bounding)."""
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > self.max_grad_norm:
            gradients = gradients * (self.max_grad_norm / grad_norm)
        return gradients, min(float(grad_norm), self.max_grad_norm)

    def add_noise(self, gradients: np.ndarray,
                  sensitivity: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply the Gaussian mechanism: clip + add noise.

        noise_scale = sensitivity * noise_multiplier, where sensitivity
        defaults to max_grad_norm (per-client release). For server-level
        central DP on an averaged update, pass sensitivity = 2*C/n so the
        noise scales correctly with the number of aggregated samples.
        """
        clipped_grads, clipped_norm = self.clip_gradients(gradients)

        bound = self.max_grad_norm if sensitivity is None else float(sensitivity)
        noise_scale = bound * self.noise_multiplier
        noise = self._rng.randn(*gradients.shape) * noise_scale
        noisy_grads = clipped_grads + noise

        # Per-round epsilon from the Gaussian mechanism (independent of bound:
        # bound cancels between sensitivity and noise scale)
        round_epsilon = np.sqrt(2 * np.log(1.25 / self.delta)) / self.noise_multiplier
        self.accountant.update(float(round_epsilon), self.delta)

        noise_metrics: Dict = {
            "noise_scale": float(noise_scale),
            "noise_norm": float(np.linalg.norm(noise)),
            "gradient_norm_before": float(np.linalg.norm(gradients)),
            "gradient_norm_after": float(np.linalg.norm(noisy_grads)),
            "clipping_applied": bool(np.linalg.norm(gradients) > self.max_grad_norm),
            "snr_db": float(20 * np.log10(clipped_norm / (float(np.linalg.norm(noise)) + 1e-8))),
            "round_epsilon": float(round_epsilon),
        }
        self.noise_history.append(noise_metrics)

        return noisy_grads, noise_metrics

    def compute_privacy_loss(self, n_compositions: int) -> Dict:
        """Total privacy loss for n composed rounds, under two accountants.

        Returns both the advanced-composition bound (the elementary bound
        tracked by :class:`PrivacyAccountant`) and the tighter RDP conversion
        for the same per-round noise multiplier.  ``total_epsilon`` /
        ``total_delta`` are the advanced-composition figures; use
        ``total_epsilon_rdp`` for the tight (epsilon, delta) statement at the
        stated *total* delta.
        """
        adv_eps, adv_delta = advanced_composition_total(
            self.accountant_round_epsilon, self.delta, n_compositions
        )
        rdp_eps, alpha_opt = rdp_total_epsilon(
            self.noise_multiplier, n_compositions, self.delta
        )
        # Legacy, strictly tighter-than-nothing guard: never report less
        # privacy loss than the accountant has actually recorded.
        adv_eps = max(adv_eps, self.accountant.spent_epsilon)
        adv_delta = max(adv_delta, self.accountant.spent_delta)

        return {
            "total_epsilon": float(adv_eps),
            "total_delta": float(adv_delta),
            "total_epsilon_rdp": float(rdp_eps),
            "rdp_alpha_optimal": float(alpha_opt),
            "rounds": n_compositions,
            "budget_used_fraction": self.accountant.budget_fraction_used,
        }

    @property
    def accountant_round_epsilon(self) -> float:
        """Per-round epsilon implied by the deployed noise multiplier."""
        return float(
            np.sqrt(2 * np.log(1.25 / self.delta)) / max(self.noise_multiplier, 1e-12)
        )

    def get_privacy_report(self) -> Dict:
        """Full privacy accounting report.

        ``epsilon``/``delta`` are the *per-round* release parameters;
        ``spent_epsilon``/``spent_delta`` are the advanced-composition totals
        after ``rounds_completed`` releases, and ``spent_epsilon_rdp`` the
        tighter RDP total at the same *total* delta.
        """
        rounds = self.accountant.rounds
        composed = self.compute_privacy_loss(rounds) if rounds else {}
        return {
            "epsilon": self.epsilon,
            "epsilon_per_round": self.accountant_round_epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "spent_epsilon": self.accountant.spent_epsilon,
            "spent_delta": self.accountant.spent_delta,
            "spent_epsilon_rdp": composed.get("total_epsilon_rdp", 0.0),
            "rounds_completed": rounds,
            "budget_remaining": self.accountant.budget_remaining,
            "budget_used_fraction": self.accountant.budget_fraction_used,
            "is_exhausted": self.accountant.is_budget_exhausted,
        }


class HomomorphicEncryption:
    """
    Simulated Homomorphic Encryption for Secure Aggregation.

    In the real TA-BC-BFL framework, gradients are encrypted with a 
    partially homomorphic scheme (e.g., Paillier) allowing the server 
    to aggregate ciphertexts without decrypting individual contributions.

    This simulation captures the computational overhead and privacy 
    properties without actual cryptographic operations.
    """

    def __init__(self, key_size_bits: int = 2048, precision_bits: int = 24):
        self.key_size_bits = key_size_bits
        self.precision_bits = precision_bits
        self.scale = 2 ** precision_bits
        self._rng = np.random.RandomState(42)

        # Simulate key generation overhead
        self._public_key_seed = self._rng.randint(0, 2**31)
        self._private_key_seed = self._rng.randint(0, 2**31)

        self.encryption_ops: int = 0
        self.decryption_ops: int = 0
        self.aggregation_ops: int = 0
        self.total_ciphertext_size_bytes: int = 0

    def encrypt(self, plaintext: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Simulate encryption of gradient vector.
        Returns noisy ciphertext and encryption metadata.
        """
        # Quantize to fixed-point representation
        quantized = np.round(plaintext * self.scale).astype(np.int64)

        # Add simulated encryption noise (LWE-style)
        noise_bound = self.scale // (2 ** 10)
        encryption_noise = self._rng.randint(-noise_bound, noise_bound, size=plaintext.shape)
        ciphertext = (quantized + encryption_noise).astype(np.float64)

        ciphertext_size = ciphertext.nbytes + self.key_size_bits // 8
        self.total_ciphertext_size_bytes += ciphertext_size
        self.encryption_ops += 1

        metadata = {
            "ciphertext_size_bytes": ciphertext_size,
            "quantization_error": float(np.max(np.abs(quantized / self.scale - plaintext))),
            "key_size_bits": self.key_size_bits,
        }
        return ciphertext, metadata

    def decrypt(self, ciphertext: np.ndarray) -> np.ndarray:
        """Simulate decryption of aggregated ciphertext."""
        self.decryption_ops += 1
        return ciphertext / self.scale

    def aggregate_ciphertexts(self, ciphertexts: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """
        Simulate homomorphic weighted aggregation.
        In real HE: Σ (weight_i ⊙ ciphertext_i) — no decryption needed.
        """
        self.aggregation_ops += 1
        aggregated = np.zeros_like(ciphertexts[0], dtype=np.float64)
        for ct, w in zip(ciphertexts, weights):
            aggregated += w * ct
        return aggregated

    def secure_aggregate(
        self, gradients_list: List[np.ndarray], weights: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full secure aggregation pipeline:
        encrypt → aggregate (on ciphertexts) → decrypt
        """
        ciphertexts = []
        enc_metadatas = []
        for grads in gradients_list:
            ct, meta = self.encrypt(grads)
            ciphertexts.append(ct)
            enc_metadatas.append(meta)

        # Aggregate in encrypted domain (no raw gradient seen by server)
        aggregated_ct = self.aggregate_ciphertexts(ciphertexts, weights)
        result = self.decrypt(aggregated_ct)

        total_comm_overhead = sum(m["ciphertext_size_bytes"] for m in enc_metadatas)
        plaintext_size = sum(g.nbytes for g in gradients_list)

        metrics = {
            "encryption_overhead_ratio": total_comm_overhead / max(plaintext_size, 1),
            "total_ciphertext_bytes": total_comm_overhead,
            "plaintext_bytes": plaintext_size,
            "encryption_ops": self.encryption_ops,
            "decryption_ops": self.decryption_ops,
            "aggregation_ops": self.aggregation_ops,
            "max_quantization_error": float(max(m["quantization_error"] for m in enc_metadatas)),
            "key_size_bits": self.key_size_bits,
        }
        return result, metrics

    def get_security_level(self) -> str:
        """Estimate the AES-equivalent security level."""
        if self.key_size_bits >= 4096:
            return "256-bit (Post-Quantum Ready)"
        elif self.key_size_bits >= 2048:
            return "112-bit (NIST Level 3)"
        elif self.key_size_bits >= 1024:
            return "80-bit (Legacy)"
        return "Below Recommended"
