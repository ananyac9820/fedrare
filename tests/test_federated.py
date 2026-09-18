"""Tests for the aggregation interface, shared primitives, baseline rules and Tier A loop."""

import numpy as np
import pytest
import torch

from src.federated import baselines
from src.federated.interface import (INTERFACE_VERSION, flatten, state_delta,
                                     validate_round_info)
from src.federated.primitives import (FlatLayout, cap_weights, clip_to_median_norm,
                                      evidence_share, head_evidence)
from src.federated.tier_a import (ClientData, TierAConfig, TierAHead, local_train,
                                  run_federated)

K, C, D = 6, 8, 16


def make_states(seed=0, body=False, outlier=None):
    g = torch.Generator().manual_seed(seed)
    base = {"classifier.weight": torch.randn(C, D, generator=g),
            "classifier.bias": torch.randn(C, generator=g)}
    if body:
        base = {"body.weight": torch.randn(D, D, generator=g),
                "body.num_batches_tracked": torch.tensor(7), **base}
    clients = []
    for k in range(K):
        s = {n: (v + 0.1 * torch.randn(v.shape, generator=g) if v.is_floating_point() else v.clone())
             for n, v in base.items()}
        if k == outlier:
            s = {n: (v + 50.0 if v.is_floating_point() else v) for n, v in s.items()}
        clients.append(s)
    return clients, base


SIZES = [9930, 3163, 2691, 1807, 655, 351]


# ------------------------------------------------------------------ interface

@pytest.mark.parametrize("rule", sorted(baselines.RULES))
@pytest.mark.parametrize("body", [False, True])
def test_every_rule_meets_the_contract(rule, body):
    clients, g = make_states(body=body)
    new, info = baselines.RULES[rule](clients, SIZES, g, 3, n_classes=C, unused_kwarg="ignored")
    validate_round_info(info, K, C)
    assert info["rule"] == rule and info["round"] == 3
    assert info["interface_version"] == INTERFACE_VERSION
    assert set(new) == set(g)
    for k, v in g.items():
        assert new[k].shape == v.shape and new[k].dtype == v.dtype
        assert new[k].data_ptr() != v.data_ptr()
    if body:  # non-float entries copied unchanged
        assert int(new["body.num_batches_tracked"]) == 7


def test_validate_round_info_rejects_bad_shapes():
    clients, g = make_states()
    _, info = baselines.fedavg(clients, SIZES, g, 1)
    bad = dict(info, head_row_weights=info["head_row_weights"][:, :3])
    with pytest.raises(ValueError, match="head_row_weights"):
        validate_round_info(bad, K, C)
    with pytest.raises(ValueError, match="missing"):
        validate_round_info({k: v for k, v in info.items() if k != "clip_scale"}, K, C)


# ------------------------------------------------------------------ primitives

def test_flat_layout_rows_cover_head_exactly():
    clients, g = make_states(body=True)
    layout = FlatLayout.from_state(g)
    all_head = np.concatenate(layout.head_cols)
    assert len(all_head) == C * D + C and len(np.unique(all_head)) == len(all_head)
    assert len(layout.body_cols) == D * D
    assert set(all_head).isdisjoint(layout.body_cols)
    # row c = weight row c + bias c, in that order within the flat vector
    vec = flatten({k: g[k].double() for k in layout.keys}, layout.keys)
    c = 5
    assert torch.equal(vec[layout.head_weight_cols[c]], g["classifier.weight"][c].double())
    assert torch.equal(vec[layout.head_cols[c]],
                       torch.cat([g["classifier.weight"][c], g["classifier.bias"][c:c + 1]]).double())


def test_clip_to_median_norm():
    u = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 60.0], dtype=torch.float64))
    clipped, scale = clip_to_median_norm(u)
    median = 3.5
    norms = clipped.norm(dim=1)
    assert torch.allclose(norms, torch.tensor([1, 2, 3, median, median, median], dtype=torch.float64))
    assert np.allclose(scale[:3], 1.0) and np.all(scale[3:] < 1)


def test_head_evidence_and_share():
    clients, g = make_states()
    layout = FlatLayout.from_state(g)
    u = torch.stack([flatten(state_delta(s, g, layout.keys), layout.keys) for s in clients])
    e = head_evidence(u, layout)
    manual = np.array([[float((s["classifier.weight"][c] - g["classifier.weight"][c]).double().norm())
                        for c in range(C)] for s in clients])
    assert np.allclose(e, manual)
    share = evidence_share(e)
    assert np.allclose(share.sum(0), 1)
    assert np.allclose(evidence_share(np.zeros((K, C))), 0)


def test_cap_weights():
    w = cap_weights([10, 1, 1, 1, 1, 1], cap=0.5)
    assert np.isclose(w.sum(), 1) and w.max() <= 0.5 + 1e-12 and np.isclose(w[0], 0.5)
    assert np.allclose(w[1:], 0.1)                         # excess spread proportionally
    assert np.allclose(cap_weights([1, 1, 1, 1, 1, 1]), 1 / 6)   # nothing over cap: unchanged
    w = cap_weights([1, 0, 0, 0, 0, 0], cap=0.5)           # all others zero: spread equally
    assert np.isclose(w[0], 0.5) and np.allclose(w[1:], 0.1)
    w = cap_weights([6, 5, 0.1, 0.1, 0.1, 0.1], cap=0.4)   # cascading caps
    assert np.isclose(w.sum(), 1) and w.max() <= 0.4 + 1e-12
    with pytest.raises(ValueError):
        cap_weights([1, 1], cap=0.3)


# ------------------------------------------------------------------ rules

def test_fedavg_is_size_weighted_mean_of_states():
    clients, g = make_states()
    new, info = baselines.fedavg(clients, SIZES, g, 1)
    w = np.array(SIZES) / sum(SIZES)
    for key in g:
        expected = sum(w[k] * clients[k][key].double() for k in range(K))
        assert torch.allclose(new[key].double(), expected, atol=1e-5)
    assert np.allclose(info["head_row_weights"], w[:, None])


def test_camp_a_reduces_to_fedavg_without_evidence_bonus():
    clients, g = make_states(body=True)
    fa, _ = baselines.fedavg(clients, SIZES, g, 1)
    ca, info = baselines.camp_a(clients, SIZES, g, 1, lam=0.0, clip=False, cap=1.0)
    for key in g:
        assert torch.allclose(fa[key].double(), ca[key].double(), atol=1e-6)
    assert "evidence" in info and "evidence_share" in info


def test_camp_a_rewards_evidence_and_respects_cap():
    clients, g = make_states()
    # client 5 (smallest) makes a large change to row 6 only
    clients[5]["classifier.weight"][6] += 5.0
    _, info = baselines.camp_a(clients, SIZES, g, 1, clip=False)
    fedavg_w = np.array(SIZES) / sum(SIZES)
    assert info["head_row_weights"][5, 6] > 5 * fedavg_w[5]
    assert info["head_row_weights"].max() <= 0.5 + 1e-9
    assert np.argmax(info["evidence"][:, 6]) == 5


def test_krum_drops_outlier_and_trimmed_weights_are_per_row():
    clients, g = make_states(outlier=4)
    _, info = baselines.krum(clients, SIZES, g, 1)
    assert not info["selected"][4]
    _, info = baselines.multi_krum(clients, SIZES, g, 1)
    assert not info["selected"][4] and info["selected"].sum() == 5
    _, info = baselines.trimmed_mean(clients, SIZES, g, 1)
    assert np.allclose(info["head_row_weights"][4], 0)      # outlier trimmed in every row


# ------------------------------------------------------------------ Tier A loop

def synthetic_clients(n_per=(60, 40, 30, 20, 20, 10), d=12, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(C, d)) * 3
    clients = []
    for k, n in enumerate(n_per):
        y = rng.integers(0, C, n)
        x = centres[y] + rng.normal(size=(n, d))
        clients.append(ClientData(torch.tensor(x, dtype=torch.float32), torch.tensor(y)))
    y_test = rng.integers(0, C, 200)
    x_test = torch.tensor(centres[y_test] + rng.normal(size=(200, d)), dtype=torch.float32)
    return clients, x_test, y_test


def test_local_train_is_deterministic_and_changes_the_head():
    cfg = TierAConfig(local_steps=5, feature_dim=12)
    g = TierAHead(12, C).state_dict()
    clients, _, _ = synthetic_clients()
    a = local_train(g, clients[0].x, clients[0].y, cfg, np.random.default_rng(1))
    b = local_train(g, clients[0].x, clients[0].y, cfg, np.random.default_rng(1))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a["classifier.weight"], g["classifier.weight"])


def test_run_federated_learns_and_supports_hooks():
    cfg = TierAConfig(rounds=8, local_steps=10, feature_dim=12, lr=1e-2)
    clients, xt, yt = synthetic_clients()
    res = run_federated(lambda: baselines.fedavg, clients, xt, yt, cfg, seed=0, rare_ids=[5, 6])
    assert len(res["round_infos"]) == 8
    assert res["final"]["balanced_accuracy"] > 0.6          # separable synthetic data

    calls = []

    def scaling_attack(*, round_num, client_id, global_state, data, honest_train):
        calls.append((round_num, client_id))
        s = honest_train()
        return {k: global_state[k] + 10 * (s[k] - global_state[k]) for k in s}

    res2 = run_federated(lambda: baselines.fedavg_clipped, clients, xt, yt, cfg, seed=0,
                         rare_ids=[5, 6], client_hooks={3: scaling_attack})
    assert calls == [(r, 3) for r in range(1, 9)]
    assert all(info["clip_scale"][3] < 1 for info in res2["round_infos"])

    # same seed, same factory -> identical run
    res3 = run_federated(lambda: baselines.fedavg, clients, xt, yt, cfg, seed=0, rare_ids=[5, 6])
    assert res3["final"] == res["final"]
