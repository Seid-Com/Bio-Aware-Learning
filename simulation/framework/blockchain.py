"""
Blockchain Trust Management Module
Simulates a blockchain-based adaptive trust scoring model for federated learning.
Each communication round creates a new block recording client contributions
and updating trust scores based on gradient quality and historical reliability.
"""

import numpy as np
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .clients import LocalModelUpdate


@dataclass
class TrustRecord:
    """A single client trust record stored on the blockchain."""
    client_id: int
    round_number: int
    loss: float
    accuracy: float
    gradient_norm: float
    trust_delta: float
    is_flagged: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class Block:
    """A single block in the trust blockchain."""
    index: int
    round_number: int
    trust_records: List[TrustRecord]
    previous_hash: str
    nonce: int = 0
    timestamp: float = field(default_factory=time.time)
    hash: str = ""

    def compute_hash(self) -> str:
        block_data = {
            "index": self.index,
            "round": self.round_number,
            "records": [
                {
                    "client": r.client_id,
                    "loss": round(r.loss, 6),
                    "accuracy": round(r.accuracy, 6),
                    "grad_norm": round(r.gradient_norm, 6),
                    "flagged": r.is_flagged,
                }
                for r in self.trust_records
            ],
            "prev_hash": self.previous_hash,
            "nonce": self.nonce,
            "timestamp": round(self.timestamp, 3),
        }
        block_str = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_str.encode()).hexdigest()

    def mine(self, difficulty: int = 2):
        """Proof-of-work mining (lightweight simulation)."""
        target = "0" * difficulty
        while True:
            self.hash = self.compute_hash()
            if self.hash.startswith(target):
                break
            self.nonce += 1

    def is_valid(self) -> bool:
        return self.hash == self.compute_hash()


class BlockchainTrustManager:
    """
    Implements the Blockchain-Enabled Adaptive Trust Scoring model from TA-BC-BFL.

    Key mechanisms:
    1. Gradient consistency check — penalizes anomalous gradient norms
    2. Historical reliability — trust accumulates over rounds
    3. Performance tracking — accuracy and loss contribute to trust
    4. Anomaly detection — detects potential poisoning via statistical outliers
    5. Sybil resistance — trust is bounded and requires consistent history
    """

    def __init__(
        self,
        num_clients: int,
        initial_trust: float = 0.7,
        learning_rate: float = 0.15,
        decay_factor: float = 0.95,
        anomaly_threshold: float = 2.5,
        min_trust: float = 0.05,
        max_trust: float = 1.0,
        mining_difficulty: int = 2,
    ):
        self.num_clients = num_clients
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.anomaly_threshold = anomaly_threshold
        self.min_trust = min_trust
        self.max_trust = max_trust
        self.mining_difficulty = mining_difficulty

        # Initialize trust scores
        self.trust_scores = np.full(num_clients, initial_trust)
        self.trust_history: List[np.ndarray] = [self.trust_scores.copy()]

        # Blockchain
        self.chain: List[Block] = []
        self.block_hashes: List[str] = []

        # Analytics
        self.anomalies_detected: List[Dict] = []

        # Performance histories for adaptive scoring
        self.loss_history: Dict[int, List[float]] = {i: [] for i in range(num_clients)}
        self.accuracy_history: Dict[int, List[float]] = {i: [] for i in range(num_clients)}
        self.grad_norm_history: Dict[int, List[float]] = {i: [] for i in range(num_clients)}
        self.flag_counts: Dict[int, int] = {i: 0 for i in range(num_clients)}

        self._create_genesis_block(num_clients, initial_trust)

    def _create_genesis_block(self, num_clients: int, initial_trust: float):
        genesis_records = [
            TrustRecord(
                client_id=i,
                round_number=0,
                loss=0.0,
                accuracy=0.5,
                gradient_norm=0.0,
                trust_delta=0.0,
                is_flagged=False,
            )
            for i in range(num_clients)
        ]
        genesis = Block(
            index=0,
            round_number=0,
            trust_records=genesis_records,
            previous_hash="0" * 64,
        )
        genesis.mine(self.mining_difficulty)
        self.chain.append(genesis)
        self.block_hashes.append(genesis.hash)

    def _detect_anomalies(self, updates: List[LocalModelUpdate]) -> np.ndarray:
        """
        Statistical anomaly detection for gradient norms.
        Uses z-score and historical context to flag suspicious updates.
        """
        grad_norms = np.array([np.linalg.norm(u.gradients) for u in updates])
        flagged = np.zeros(len(updates), dtype=bool)

        if len(updates) >= 3:
            median = np.median(grad_norms)
            mad = np.median(np.abs(grad_norms - median))
            if mad > 1e-8:
                z_scores = np.abs(grad_norms - median) / (1.4826 * mad)
                flagged = z_scores > self.anomaly_threshold

        # Check accuracy anomalies (too low could indicate poisoning)
        accuracies = np.array([u.accuracy for u in updates])
        mean_acc = accuracies.mean()
        std_acc = accuracies.std() + 1e-8
        acc_z = (mean_acc - accuracies) / std_acc
        flagged |= acc_z > self.anomaly_threshold

        return flagged

    def _compute_trust_delta(
        self,
        client_id: int,
        update: LocalModelUpdate,
        is_flagged: bool,
        all_updates: List[LocalModelUpdate],
    ) -> float:
        """
        Compute the trust score adjustment for a client after a round.

        Positive factors: high accuracy, low loss, consistent gradients, history
        Negative factors: anomalous gradients, low accuracy, flagged behavior
        """
        # Normalize performance against peers
        all_losses = [u.loss for u in all_updates]
        all_accuracies = [u.accuracy for u in all_updates]

        loss_rank = sorted(all_losses).index(update.loss) / max(len(all_losses) - 1, 1)
        acc_rank = sorted(all_accuracies, reverse=True).index(update.accuracy) / max(len(all_accuracies) - 1, 1)

        # Higher rank = better performance
        performance_score = 0.5 * (1 - loss_rank) + 0.5 * acc_rank

        # Historical consistency bonus
        history_len = len(self.accuracy_history[client_id])
        if history_len >= 2:
            recent_accuracies = self.accuracy_history[client_id][-3:]
            consistency = 1.0 - np.std(recent_accuracies)
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 0.5

        # Compute delta
        if is_flagged:
            # Escalating penalty for repeatedly flagged (anomalous) clients
            delta = -self.learning_rate * (0.5 + self.flag_counts[client_id] * 0.1)
        else:
            # Reliability accrual: always non-negative so honest clients
            # converge toward max_trust (uniform weights in clean settings),
            # while better-performing clients accrue slightly faster.
            delta = self.learning_rate * (0.3 + 0.25 * performance_score + 0.15 * consistency)

        return float(delta)

    def update_trust(
        self, updates: List[LocalModelUpdate], round_number: int
    ) -> Dict:
        """
        Update trust scores after a federated round and record on blockchain.

        Returns dict with trust metrics and block information.
        """
        flagged = self._detect_anomalies(updates)
        trust_records = []
        trust_deltas = []

        for i, update in enumerate(updates):
            cid = update.client_id
            is_flagged = bool(flagged[i])
            grad_norm = float(np.linalg.norm(update.gradients))

            # Update histories
            self.loss_history[cid].append(update.loss)
            self.accuracy_history[cid].append(update.accuracy)
            self.grad_norm_history[cid].append(grad_norm)

            if is_flagged:
                self.flag_counts[cid] += 1
                self.anomalies_detected.append({
                    "round": round_number,
                    "client_id": cid,
                    "grad_norm": grad_norm,
                    "accuracy": update.accuracy,
                })

            # Compute trust delta
            delta = self._compute_trust_delta(cid, update, is_flagged, updates)
            trust_deltas.append(delta)

            # Apply decay + delta, then clip
            new_trust = self.trust_scores[cid] * self.decay_factor + delta
            new_trust = np.clip(new_trust, self.min_trust, self.max_trust)
            self.trust_scores[cid] = new_trust

            record = TrustRecord(
                client_id=cid,
                round_number=round_number,
                loss=update.loss,
                accuracy=update.accuracy,
                gradient_norm=grad_norm,
                trust_delta=delta,
                is_flagged=is_flagged,
            )
            trust_records.append(record)

        # Create new blockchain block
        prev_hash = self.chain[-1].hash
        new_block = Block(
            index=len(self.chain),
            round_number=round_number,
            trust_records=trust_records,
            previous_hash=prev_hash,
        )
        new_block.mine(self.mining_difficulty)
        self.chain.append(new_block)
        self.block_hashes.append(new_block.hash)

        self.trust_history.append(self.trust_scores.copy())

        # Return indices mapped to updates (in case clients are a subset)
        client_indices = [u.client_id for u in updates]
        trust_subset = np.array([self.trust_scores[cid] for cid in client_indices])
        # Normalize to sum to 1 for use as weights
        trust_weights = trust_subset / trust_subset.sum()

        return {
            "trust_scores": self.trust_scores.copy(),
            "trust_weights": trust_weights,
            "flagged_clients": [u.client_id for i, u in enumerate(updates) if flagged[i]],
            "trust_deltas": dict(zip(client_indices, trust_deltas)),
            "block_hash": new_block.hash,
            "block_index": new_block.index,
            "chain_length": len(self.chain),
            "anomalies": len(self.anomalies_detected),
        }

    def verify_chain_integrity(self) -> bool:
        """Verify the entire blockchain is intact and unmodified."""
        for i in range(1, len(self.chain)):
            block = self.chain[i]
            prev_block = self.chain[i - 1]
            if not block.is_valid():
                return False
            if block.previous_hash != prev_block.hash:
                return False
        return True

    def get_trust_history_matrix(self) -> np.ndarray:
        """Return trust scores over time as (rounds+1, num_clients) matrix."""
        return np.array(self.trust_history)

    def get_client_trust_summary(self) -> Dict:
        """Summary statistics for each client's trust trajectory."""
        summary = {}
        for cid in range(self.num_clients):
            history = [h[cid] for h in self.trust_history]
            summary[cid] = {
                "current": float(self.trust_scores[cid]),
                "min": float(min(history)),
                "max": float(max(history)),
                "trend": float(history[-1] - history[0]) if len(history) > 1 else 0.0,
                "flags": self.flag_counts[cid],
            }
        return summary
