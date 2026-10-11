# Tier B run - console output (Colab GPU, 11 October 2026)

`scripts/12_tier_b.py`, unchanged, run on a Colab GPU after the local CPU attempt was stopped
(~15 h projected on CPU against ~1.8 h here). Device `cuda`, 20 rounds x 50 local steps, seed 42,
three runs as fixed in D8. Per-round wall time 109-134 s.

**Provenance.** This file is the console output, pasted from the Colab session. The result file it
mentions has since been recovered and committed at `results/tier_b.json` (and
`docs/results/tier_b.json`), and **that file supersedes this one**: every number below was checked
against it and matches, but it carries per-round, per-class and confusion detail this log does not.
See the R11 addendum in DEVIATIONS.md, which reads the attack's effect on rare-class *recall* - the
part the headline F1 hides.

## Test metrics, balanced accuracy / rare macro-F1

| round | fedavg, no attack | fedavg, A2 sleeper | fedavg + clipping, A2 |
| --- | --- | --- | --- |
| 5 | 0.535 / 0.420 | 0.535 / 0.420 | 0.531 / 0.381 |
| 10 | 0.586 / 0.486 | 0.586 / 0.486 | 0.578 / 0.469 |
| 15 | 0.587 / 0.608 | 0.587 / 0.608 | 0.596 / 0.621 |
| 20 | **0.592 / 0.523** | **0.563 / 0.598** | **0.562 / 0.615** |

The attacker is honest through round 15, and the first two columns are identical to three decimals
through that round - the harness is deterministic, so everything that differs at round 20 is the
attack and nothing else.

## The pre-registered criterion (D8)

> Tier B confirms the Tier A sleeper finding if the A2 drop in FedAvg's final rare macro-F1 is
> >= 0.10 and clipping recovers at least half of that drop.

Measured: **sleeper drop -0.074** (the attacked run ended *higher* on rare macro-F1 than the honest
one), **clipping recovery 0.017**. The criterion is not met. See DEVIATIONS.md R11.
