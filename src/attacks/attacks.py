"""
Attacks A1-A3 (EARN Project Design v2, Section 6.4), as client hooks for the Tier A loop.

Each attack is a ClientHook (src/federated/tier_a.py): it replaces one client's local training
and returns that client's state_dict for the round. Constants are fixed here, before any
attack result exists (docs/DEVIATIONS.md D6).

    A1 SuddenExpert   Holds little or no rare data. Trains on its data with rare-class labels
                      flipped to the target class, then rewrites its rare-class head rows so
                      they look like a big holder's: large, and pointing so as to push the
                      rare classes' scores down (toward the target class).
    A2 Sleeper        Honest for SLEEPER_HONEST_ROUNDS rounds, then behaves exactly like A1.
    A3 Scaling        Trains with flipped labels and multiplies its whole update by SCALE,
                      to dominate an unclipped average. Tests the clipping step.

"Look like a holder" is made concrete as: each attacked rare row's weight change has norm
INFLATE x the largest row norm of the attacker's own honest update this round, along -u, where
u is the unit mean feature vector of the attacker's local data (DenseNet features are
non-negative, so -u lowers that class's score for essentially every image); the bias of each
attacked row moves by -INFLATE x the attacker's largest honest bias change.

Every hook also exposes `active(round_num)` and `claims_holder` so an experiment can model the
attacker spoofing an evidence reading (the oracle-evidence runs, D6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.federated.interface import DEFAULT_HEAD_PREFIX, State

TARGET_CLASS = 1             # Melanocytic nevus - the design doc's example target
INFLATE = 3.0                # A1/A2 rare-row norm, x the attacker's largest honest row norm
SCALE = 10.0                 # A3 update multiplier (docs/aggregation_interface.md example)
SLEEPER_HONEST_ROUNDS = 15   # A2: honest for rounds 1..15, attacks from round 16


def flip_labels(y: torch.Tensor, rare_ids, target: int = TARGET_CLASS) -> torch.Tensor:
    out = y.clone()
    for c in rare_ids:
        out[y == c] = target
    return out


def craft_expert_rows(state: State, global_state: State, x: torch.Tensor, rare_ids,
                      inflate: float = INFLATE, prefix: str = DEFAULT_HEAD_PREFIX) -> State:
    """Rewrite the rare head rows of `state` (a trained client state) as A1 does."""
    wk, bk = f"{prefix}.weight", f"{prefix}.bias"
    dw = (state[wk] - global_state[wk]).clone()
    db = (state[bk] - global_state[bk]).clone()
    row_norm = float(dw.norm(dim=1).max())
    bias_mag = float(db.abs().max())
    u = x.float().mean(dim=0)
    u = u / u.norm().clamp_min(1e-12)
    for c in rare_ids:
        dw[c] = -inflate * row_norm * u.to(dw.dtype)
        db[c] = -inflate * bias_mag
    out = {k: v.detach().clone() for k, v in state.items()}
    out[wk] = global_state[wk] + dw
    out[bk] = global_state[bk] + db
    return out


@dataclass
class SuddenExpert:
    """A1. Active from `start_round` (1 = every round)."""
    rare_ids: tuple[int, ...]
    target: int = TARGET_CLASS
    inflate: float = INFLATE
    start_round: int = 1
    name: str = "A1"
    claims_holder: bool = True

    def active(self, round_num: int) -> bool:
        return round_num >= self.start_round

    def __call__(self, *, round_num, client_id, global_state, data, honest_train) -> State:
        if not self.active(round_num):
            return honest_train()
        state = honest_train(data.x, flip_labels(data.y, self.rare_ids, self.target))
        return craft_expert_rows(state, global_state, data.x, self.rare_ids, self.inflate)


@dataclass
class Sleeper(SuddenExpert):
    """A2. Honest for SLEEPER_HONEST_ROUNDS rounds, then A1."""
    start_round: int = SLEEPER_HONEST_ROUNDS + 1
    name: str = "A2"


@dataclass
class Scaling:
    """A3. Label-flipped update, multiplied by `scale`."""
    rare_ids: tuple[int, ...]
    target: int = TARGET_CLASS
    scale: float = SCALE
    name: str = "A3"
    claims_holder: bool = False
    start_round: int = 1

    def active(self, round_num: int) -> bool:
        return round_num >= self.start_round

    def __call__(self, *, round_num, client_id, global_state, data, honest_train) -> State:
        state = honest_train(data.x, flip_labels(data.y, self.rare_ids, self.target))
        return {k: (global_state[k] + self.scale * (v - global_state[k])
                    if v.is_floating_point() else v.clone())
                for k, v in state.items()}


@dataclass(frozen=True)
class AttackSpec:
    """Which attack, on which client. Fixed per experiment in docs/DEVIATIONS.md D6."""
    name: str                 # "none" | "A1" | "A2" | "A3"
    attacker: int | None = None
    kwargs: dict = field(default_factory=dict)

    def hooks(self, rare_ids) -> dict:
        if self.name == "none":
            return {}
        cls = {"A1": SuddenExpert, "A2": Sleeper, "A3": Scaling}[self.name]
        return {self.attacker: cls(rare_ids=tuple(rare_ids), **self.kwargs)}


# Attacker per attack (D6): A1 and A3 - centre 4, which holds no image of either rare class on
# S1 or S2, so any rare-class influence it gains is unearned. A2 - centre 1, a genuine holder of
# both rare classes on S1 (23 / 45 training images), so it can build an honest record first.
DEFAULT_ATTACKS = {
    "none": AttackSpec("none"),
    "A1": AttackSpec("A1", attacker=4),
    "A2": AttackSpec("A2", attacker=1),
    "A3": AttackSpec("A3", attacker=4),
    # D7 follow-up: the specialist itself turns after 15 honest rounds (sole rare holder on S2)
    "A2s": AttackSpec("A2", attacker=2),
}
