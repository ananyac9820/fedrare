"""Tests for EARN (one per step of the design doc's round), the hash-chain ledger and attacks."""

import numpy as np
import pytest
import torch

from src.attacks.attacks import (DEFAULT_ATTACKS, Scaling, Sleeper, SuddenExpert,
                                 craft_expert_rows, flip_labels)
from src.federated import baselines
from src.federated.earn import EARN, EARNConfig, NEUTRAL
from src.federated.interface import validate_round_info
from src.federated.primitives import cap_weights
from src.federated.tier_a import ClientData, TierAConfig, TierAHead, run_federated
from src.ledger.chain import HashChainLedger, LedgerError

K, C, D = 6, 8, 16
SIZES = [9930, 3163, 2691, 1807, 655, 351]


def states_from_rows(row_updates: np.ndarray, seed: int = 0):
    """Global state + client states whose head updates are exactly row_updates (K, C, D+1)."""
    g = torch.Generator().manual_seed(seed)
    base = {"classifier.weight": torch.randn(C, D, generator=g, dtype=torch.float64),
            "classifier.bias": torch.randn(C, generator=g, dtype=torch.float64)}
    clients = []
    for k in range(K):
        u = torch.as_tensor(row_updates[k])
        clients.append({"classifier.weight": base["classifier.weight"] + u[:, :D],
                        "classifier.bias": base["classifier.bias"] + u[:, D]})
    return clients, base


def aligned_rows(rng, flip=(), scale=1.0):
    """Every client pushes each row the same way (plus noise); clients in flip push opposite."""
    direction = rng.normal(size=(C, D + 1))
    rows = np.stack([direction + 0.05 * rng.normal(size=(C, D + 1)) for _ in range(K)]) * scale
    for k in flip:
        rows[k] = -rows[k]
    return rows


def evidence_from(matrix):
    return lambda updates, layout, r: np.asarray(matrix, dtype=np.float64)


# ------------------------------------------------------------------ contract and reduction

def test_earn_meets_the_interface_and_starts_as_fedavg():
    rng = np.random.default_rng(0)
    clients, g = states_from_rows(aligned_rows(rng))
    earn = EARN(EARNConfig(tau=0.5), evidence_fn=evidence_from(np.ones((K, C))))
    new, info = earn(clients, SIZES, g, 1, n_classes=C)
    validate_round_info(info, K, C)
    # With no holders, trust stays 0 and EARN is clipped FedAvg with the 50% row cap.
    earn0 = EARN(EARNConfig(tau=1e9), evidence_fn=evidence_from(np.ones((K, C))))
    new0, info0 = earn0(clients, SIZES, g, 1, n_classes=C)
    assert (info0["trust"] == 0).all()
    capped = cap_weights(np.asarray(SIZES, dtype=float), 0.5)
    assert np.allclose(info0["head_row_weights"], capped[:, None])
    ref, _ = baselines.camp_a(clients, SIZES, g, 1, lam=0.0)    # size share only, clipped, capped
    for k in g:
        assert torch.allclose(new0[k].double(), ref[k].double(), atol=1e-10)


# ------------------------------------------------------------------ step by step

def test_coverage_and_peer_confidence():
    ev = np.zeros((K, C))
    for c in range(C):
        ev[: min(c + 1, K), c] = 1.0             # class c has min(c+1, 6) holders
    rng = np.random.default_rng(1)
    clients, g = states_from_rows(aligned_rows(rng))
    _, info = EARN(EARNConfig(tau=0.5), evidence_fn=evidence_from(ev))(clients, SIZES, g, 1)
    assert info["coverage"].tolist() == [1, 2, 3, 4, 5, 6, 6, 6]
    assert np.allclose(info["peer_confidence"], [0, 0.25, 0.5, 0.75, 1, 1, 1, 1])
    assert (info["holders"] == (ev > 0.5)).all()


def test_slow_up_fast_down_and_non_holders_untouched():
    rng = np.random.default_rng(2)
    ev = np.ones((K, C))
    ev[5] = 0.0                                  # client 5 holds nothing
    earn = EARN(EARNConfig(tau=0.5), evidence_fn=evidence_from(ev))
    g = None
    for r in range(1, 4):                        # three agreeing rounds: +0.1 each
        clients, g = states_from_rows(aligned_rows(rng), seed=r)
        _, info = earn(clients, SIZES, g, r)
    assert np.allclose(info["trust"][:5], 0.3)
    assert (info["trust"][5] == 0).all()
    clients, g = states_from_rows(aligned_rows(rng, flip=(0,)), seed=9)   # client 0 turns
    _, info = earn(clients, SIZES, g, 4)
    assert np.allclose(info["trust"][0], 0.15)   # halved
    assert np.allclose(info["trust"][1:5], 0.4)
    assert np.isnan(info["agreement"][5]).all()


def test_single_holder_is_judged_on_history_only():
    rng = np.random.default_rng(3)
    ev = np.zeros((K, C))
    ev[2] = 1.0                                  # coverage 1 everywhere: p = 0
    earn = EARN(EARNConfig(tau=0.5), evidence_fn=evidence_from(ev))
    clients, g = states_from_rows(aligned_rows(rng), seed=1)
    _, info = earn(clients, SIZES, g, 1)
    assert np.allclose(info["agreement"][2], NEUTRAL)       # no history, no peers
    rows = aligned_rows(rng)
    clients, g = states_from_rows(rows, seed=2)
    _, info = earn(clients, SIZES, g, 2)
    rows[2] = -rows[2]                                      # contradicts its own past
    clients, g = states_from_rows(rows, seed=3)
    _, info3 = earn(clients, SIZES, g, 3)
    assert (info3["agreement"][2] < 0).all()
    assert np.allclose(info3["trust"][2], info["trust"][2] * 0.5)


def test_weighting_uses_trust_times_evidence_and_caps_at_half():
    rng = np.random.default_rng(4)
    ev = np.ones((K, C))
    ev[4] = 100.0                                # client 4 claims overwhelming evidence
    earn = EARN(EARNConfig(tau=0.5, slow_ramp=False, lam=5.0), evidence_fn=evidence_from(ev))
    clients, g = states_from_rows(aligned_rows(rng))
    _, info = earn(clients, SIZES, g, 1)
    assert (info["head_row_weights"] <= 0.5 + 1e-9).all()
    assert np.allclose(info["head_row_weights"][4], 0.5)    # trust 1 (no ramp) -> capped


def test_untrusted_evidence_buys_nothing():
    rng = np.random.default_rng(5)
    ev = np.ones((K, C))
    ev[4] = 100.0
    earn = EARN(EARNConfig(tau=0.5), evidence_fn=evidence_from(ev))
    clients, g = states_from_rows(aligned_rows(rng, flip=(4,)))   # and it disagrees
    _, info = earn(clients, SIZES, g, 1)
    assert (info["trust"][4] == 0).all()
    # others earned 0.1 trust, client 4 none: it gets no more than its zero-trust (capped) share
    zero_trust = cap_weights(np.asarray(SIZES, dtype=float), 0.5)
    assert (info["head_row_weights"][4] < zero_trust[4]).all()


def test_ledger_ablation_lets_an_attacker_pass_the_history_check():
    rng = np.random.default_rng(6)
    ev = np.zeros((K, C))
    ev[3] = 1.0
    rows = aligned_rows(rng)
    for editable, expect_rise in (((), False), ((3,), True)):
        earn = EARN(EARNConfig(tau=0.5, ledger=not editable, editable_history=editable),
                    evidence_fn=evidence_from(ev))
        for r in range(1, 3):
            clients, g = states_from_rows(rows, seed=r)
            earn(clients, SIZES, g, r)
        flipped = rows.copy()
        flipped[3] = -flipped[3]
        clients, g = states_from_rows(flipped, seed=3)
        _, info = earn(clients, SIZES, g, 3)
        assert (info["trust"][3] > info["trust_prev"][3]).all() == expect_rise
        assert ("block_hash" in info) == (not editable)


def test_earn_commits_every_round_to_a_valid_chain():
    rng = np.random.default_rng(7)
    earn = EARN(EARNConfig(tau=0.5), evidence_fn=evidence_from(np.ones((K, C))))
    for r in range(1, 6):
        clients, g = states_from_rows(aligned_rows(rng), seed=r)
        earn(clients, SIZES, g, r)
    assert len(earn.chain.blocks) == 5 and earn.chain.verify()


# ------------------------------------------------------------------ ledger

def test_ledger_rejects_fast_rises_out_of_order_rounds_and_detects_tampering():
    led = HashChainLedger(2, 2, max_step=0.1)
    led.commit(1, np.array([[0.1, 0.0], [0.1, 0.1]]), [2, 1], "h1")
    with pytest.raises(LedgerError, match="rose by"):
        led.commit(2, np.array([[0.3, 0.0], [0.1, 0.1]]), [2, 1], "h2")
    with pytest.raises(LedgerError, match="out of order"):
        led.commit(3, np.zeros((2, 2)), [2, 1], "h3")
    led.commit(2, np.array([[0.2, 0.0], [0.05, 0.2]]), [2, 1], "h2")   # halving is fine
    assert led.verify()
    led.blocks[0] = led.blocks[0].__class__(1, (9000, 0, 1000, 1000), (2, 1), "h1",
                                            led.blocks[0].prev_hash, led.blocks[0].block_hash)
    assert not led.verify()


# ------------------------------------------------------------------ attacks

def tiny_data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ClientData(torch.rand(n, D, generator=g), torch.randint(0, C, (n,), generator=g))


def test_flip_labels_moves_only_rare_classes():
    y = torch.tensor([0, 5, 6, 1, 5])
    assert flip_labels(y, [5, 6], 1).tolist() == [0, 1, 1, 1, 1]


def test_expert_rows_are_inflated_and_push_rare_scores_down():
    data = tiny_data()
    g = {k: v.detach().clone() for k, v in TierAHead(D, C).state_dict().items()}
    trained = {k: v + 0.01 * torch.randn_like(v) for k, v in g.items()}
    crafted = craft_expert_rows(trained, g, data.x, [5, 6], inflate=3.0)
    dw = crafted["classifier.weight"] - g["classifier.weight"]
    honest = (trained["classifier.weight"] - g["classifier.weight"]).norm(dim=1).max()
    assert torch.allclose(dw[5].norm(), 3 * honest, rtol=1e-5)
    delta_logits = data.x @ dw[5] + (crafted["classifier.bias"] - g["classifier.bias"])[5]
    assert (delta_logits < 0).all()
    assert torch.equal(crafted["classifier.weight"][0], trained["classifier.weight"][0])


def test_attack_schedules_and_scaling():
    assert SuddenExpert(rare_ids=(5, 6)).active(1)
    s = Sleeper(rare_ids=(5, 6))
    assert not s.active(15) and s.active(16)
    data = tiny_data()
    g = {k: v.detach().clone() for k, v in TierAHead(D, C).state_dict().items()}
    honest = {k: v + 0.01 for k, v in g.items()}
    out = Scaling(rare_ids=(5, 6), scale=10)(round_num=1, client_id=0, global_state=g, data=data,
                                             honest_train=lambda x=None, y=None: honest)
    assert torch.allclose(out["classifier.bias"] - g["classifier.bias"],
                          torch.full((C,), 0.1), atol=1e-6)
    assert set(DEFAULT_ATTACKS) == {"none", "A1", "A2", "A3"}


def test_attacks_and_earn_run_in_the_tier_a_loop():
    g = torch.Generator().manual_seed(0)
    clients = [ClientData(torch.rand(40, D, generator=g), torch.randint(0, C, (40,), generator=g))
               for _ in range(K)]
    cfg = TierAConfig(rounds=3, local_steps=3, batch_size=8, feature_dim=D, n_classes=C)
    hooks = DEFAULT_ATTACKS["A1"].hooks([5, 6])
    res = run_federated(lambda: EARN(EARNConfig(tau=0.0)), clients, torch.rand(20, D),
                        np.random.default_rng(0).integers(0, C, 20), cfg, 0, [5, 6],
                        client_hooks=hooks)
    assert res["rule"] == "earn" and len(res["round_infos"]) == 3
    assert "confusion" in res["final"]
