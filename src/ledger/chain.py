"""
In-memory, append-only hash-chain ledger for EARN (design doc Sections 4.2 and 5).

This is the Week 3 stub and the design doc's third cut line ("a simulated hash-chain that still
enforces the trust-step rule"). The Solidity contract in ledger/contracts/EarnLedger.sol
enforces the same two rules on a real (local Hardhat) chain:

  1. Append-only. Round r can only be committed after round r-1; every block stores the hash of
     the previous block, so rewriting any past round breaks every later hash.
  2. Trust-step rule. No trust value may rise by more than max_step from the previous round.
     Decreases of any size are allowed ("slow up, fast down"). A violating commit is rejected,
     so even the coordinator cannot silently boost a hospital.

Trust is stored in basis points (0..10000) - exactly what the contract stores - so the Python
and on-chain checks agree bit for bit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

BPS = 10_000
ROUNDING_BPS = 1   # float -> basis-point rounding slack; the contract uses the same value


class LedgerError(ValueError):
    """A commit broke the ledger's rules and was rejected."""


def to_bps(trust: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(np.asarray(trust, dtype=np.float64), 0.0, 1.0) * BPS).astype(np.int64)


def hash_arrays(*arrays) -> str:
    """SHA-256 over the raw bytes of the arrays (float64 / int64), in order."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Block:
    round: int
    trust_bps: tuple[int, ...]      # K x C, row-major
    coverage: tuple[int, ...]       # C
    history_hash: str
    prev_hash: str
    block_hash: str

    def payload(self) -> dict:
        return {"round": self.round, "trust_bps": list(self.trust_bps),
                "coverage": list(self.coverage), "history_hash": self.history_hash,
                "prev_hash": self.prev_hash}


def _block_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class HashChainLedger:
    GENESIS = "0" * 64

    def __init__(self, n_clients: int, n_classes: int, max_step: float = 0.1):
        self.K, self.C = n_clients, n_classes
        self.max_step_bps = int(round(max_step * BPS))
        self.blocks: list[Block] = []

    @property
    def last_trust_bps(self) -> np.ndarray:
        if not self.blocks:
            return np.zeros((self.K, self.C), dtype=np.int64)
        return np.asarray(self.blocks[-1].trust_bps, dtype=np.int64).reshape(self.K, self.C)

    def commit(self, round_num: int, trust: np.ndarray, coverage: np.ndarray,
               history_hash: str) -> Block:
        expected = len(self.blocks) + 1
        if round_num != expected:
            raise LedgerError(f"round {round_num} out of order; next must be {expected}")
        t = to_bps(trust)
        if t.shape != (self.K, self.C):
            raise LedgerError(f"trust shape {t.shape} != {(self.K, self.C)}")
        rise = t - self.last_trust_bps
        if (rise > self.max_step_bps + ROUNDING_BPS).any():
            k, c = np.argwhere(rise > self.max_step_bps + ROUNDING_BPS)[0]
            raise LedgerError(f"trust of client {k} on class {c} rose by {rise[k, c]} bps "
                              f"> max step {self.max_step_bps} bps")
        prev = self.blocks[-1].block_hash if self.blocks else self.GENESIS
        payload = {"round": round_num, "trust_bps": t.ravel().tolist(),
                   "coverage": np.asarray(coverage, dtype=np.int64).tolist(),
                   "history_hash": history_hash, "prev_hash": prev}
        block = Block(round_num, tuple(payload["trust_bps"]), tuple(payload["coverage"]),
                      history_hash, prev, _block_hash(payload))
        self.blocks.append(block)
        return block

    def verify(self) -> bool:
        """True if every block's hash and back-link is intact and the step rule holds."""
        prev, last = self.GENESIS, np.zeros(self.K * self.C, dtype=np.int64)
        for i, b in enumerate(self.blocks, 1):
            if b.round != i or b.prev_hash != prev or _block_hash(b.payload()) != b.block_hash:
                return False
            t = np.asarray(b.trust_bps, dtype=np.int64)
            if (t - last > self.max_step_bps + ROUNDING_BPS).any():
                return False
            prev, last = b.block_hash, t
        return True

    def export(self) -> list[dict]:
        return [dict(b.payload(), block_hash=b.block_hash) for b in self.blocks]
