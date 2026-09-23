# Aggregation interface (v1)

The seam between the Pipeline & EARN track (`feature/pipeline`) and the Attacks & Eval track
(`feature/attacks`). Both sides build against this without syncing live. The source of truth
is the docstring of `src/federated/interface.py`; this page is the readable copy. Changing
the contract means bumping `INTERFACE_VERSION` and telling the other track the same day.

## Signature

```python
new_global_state, round_info = aggregate(client_states, client_sizes, global_state,
                                         round_num, **kwargs)
```

| Argument | Type | Meaning |
|---|---|---|
| `client_states` | `list[dict[str, Tensor]]`, length K | each client's full `state_dict` **after** local training - weights, not updates. Index k = centre k, fixed across rounds. |
| `client_sizes` | `list[int]`, length K | local training images per client. Used for size-weighted parts only. |
| `global_state` | `dict[str, Tensor]` | the state every client started from this round. Update = client - global. |
| `round_num` | `int` | 1-based. |
| `**kwargs` | | rule options. Every aggregator must accept and ignore unknown kwargs. The loop always passes `n_classes` and `head_prefix="classifier"`. |

An aggregator is any callable with this signature. Stateless rules are plain functions
(`src/federated/baselines.py`). Stateful rules (EARN) are class instances with `__call__`;
the loop takes a zero-argument **factory** and builds a fresh instance per run, so trust
never leaks between runs or seeds:

```python
run_federated(lambda: baselines.camp_a, ...)     # stateless
run_federated(lambda: EARN(step=0.1), ...)       # stateful (Week 3)
```

## Model layout

- **Head** = every key starting with `classifier.` - true for Tier A (`TierAHead`, one
  `nn.Linear`) and Tier B (torchvision DenseNet-121).
- **Head row c** = `classifier.weight[c]` + `classifier.bias[c]`: everything that scores
  disease c.
- **Body** = all other floating-point keys (Tier A has none). Non-float entries (BatchNorm's
  `num_batches_tracked`) are copied from `global_state` unchanged.

## `round_info`

All arrays are numpy. K = clients, C = classes.

**Required, every rule:**

| Key | Type / shape | Meaning |
|---|---|---|
| `interface_version` | `int` | `1` |
| `rule` | `str` | `"fedavg"`, `"krum"`, `"camp_a"`, `"earn"`, ... |
| `round` | `int` | `round_num` |
| `selected` | bool `(K,)` | client influenced the new global state at all |
| `body_weights` | float `(K,)`, sums to 1 | effective weight on body parameters (client-level weight for rules without a body/head split) |
| `head_row_weights` | float `(K, C)`, columns sum to 1 | effective weight client k got on head row c |
| `clip_scale` | float `(K,)`, in (0, 1] | factor each update was scaled by before aggregation; 1 = not clipped |

**Optional - evidence-based rules (Camp A, EARN).** If present, exactly these shapes:

| Key | Type / shape | Meaning | Who |
|---|---|---|---|
| `evidence` | float `(K, C)` | e(k,c) = L2 norm of k's (clipped) update to head weight row c | Camp A, EARN |
| `evidence_share` | float `(K, C)` | e(k,c) / Σ_j e(j,c); all-zero column → zeros | Camp A, EARN |
| `holders` | bool `(K, C)` | e(k,c) > tau | EARN |
| `coverage` | int `(C,)` | n(c) = holders per class | EARN |
| `peer_confidence` | float `(C,)` | p(c) = clip((n(c) - 1) / 4, 0, 1) | EARN |
| `agreement` | float `(K, C)` | a(k,c); NaN where not computed | EARN |
| `trust` | float `(K, C)` in [0, 1] | T(k,c) **after** this round | EARN |
| `trust_prev` | float `(K, C)` | T(k,c) before this round | EARN |
| `history_hash` | `str` | hex digest committed to the ledger | EARN |

Check any rule with `validate_round_info(info, K, C)`. The Tier A loop calls it every round,
so a rule that breaks the contract fails immediately.

## For attacks: client hooks

`run_federated(..., client_hooks={k: hook})` replaces client k's local training. A hook
returns that client's `state_dict` for the round and receives `honest_train`, which runs the
normal local training from this round's global state:

```python
def a1_sudden_expert(*, round_num, client_id, global_state, data, honest_train):
    y = data.y.clone()
    y[y == 6] = 1                                   # vascular lesion -> nevus
    state = honest_train(data.x, y)                 # train on flipped labels
    # ... inflate rare-class head rows here ...
    return state

def a3_scaling(*, round_num, client_id, global_state, data, honest_train):
    s = honest_train()
    return {k: global_state[k] + 10 * (s[k] - global_state[k]) for k in s}
```

**G1's "attacker's captured weight on a rare row"** is
`info["head_row_weights"][attacker, rare_class]`, compared with the same entry under
`fedavg` (`client_sizes[attacker] / sum(client_sizes)`).

## Splits

`src/data/splits.py`: `make_split("s1" | "s2", train.centers, train.labels, rare_ids)`
returns the training-image centre assignment; pass it to `make_clients`. S2 uses a fixed
seed inside the module, so every run and every team member gets the identical split.
