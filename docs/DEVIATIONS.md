# Deviations and amendments log

Every change to the pre-registered plan (EARN Project Design v2, 16 Sep 2026) is recorded here
**before** the run it affects, with the reason, and is committed before any result exists.
A result produced under an amendment is always reported next to the pre-registered result,
never in place of it. Entries are append-only: a later entry may supersede an earlier one but
never edits it.

Format: ID, date, what changes, why, which gate / result it touches, status.

---

## D1 - G0a retry: which of the two listed options is the retry (23 Sep 2026)

**What.** The design doc (Section 6.6, G0a "If it fails") lists two retries - the bias-row
change instead of the weight row, *or* evidence averaged over 3 rounds - and allows one retry.
We fix the **bias-row change** (listed first) as *the* retry that decides the gate. The
3-round average of the weight-row evidence is computed in the same script and reported, but
it does not decide the gate.

**Why.** Running both and letting the better one decide would be two retries.

**Measured exactly like G0a** (scripts/06_gate_g0a.py): round 1 of the Tier A loop, seed 42,
TierAConfig defaults, updates clipped to the median norm, Spearman across the 6 centres
between evidence and true training counts, per class; gate statistic = mean over the 8 classes;
pass at >= 0.7. Only the evidence definition changes: e(k,c) = |change of classifier.bias[c]|.

**Status.** Pre-registered in `scripts/06b_gate_g0a_retry.py` before its first run.

---

## D2 - Amendment M1: sign-aware evidence (23 Sep 2026) - METHOD CHANGE

**What.** A third evidence definition, run in the same script and reported separately as an
*amendment*, not as the design doc's retry:

    e_M1(k,c) = max(0, change of classifier.bias[c] in client k's clipped update)

i.e. the **signed** bias change, with negative values (the model was pushed *away* from
class c) set to zero. Same round, seed, clipping, statistic and 0.7 threshold as G0a.

**Why.** G0a came out *inverted* (mean Spearman -0.21): for both rare classes the centre with
the highest evidence (centre 4) holds zero images of them. Any magnitude-only measure (the
weight row, the bias row, or a multi-round average of either) cannot tell "pushed up" from
"pushed down". A centre that never sees class c pushes that class's score *down* on every
single step, so under Adam - which normalises step size per parameter - its row moves the
most, consistently. That is the mechanism behind the inversion, and it predicts that both
listed retries fail too.

The label-inference work the design doc cites for this signal (Section 4.1: "known signal
from label-inference work"; Ramakrishna & Dan 2022; also iDLG and LLG) uses the **sign** of
the output-layer gradient: a class present in the batch makes its bias gradient negative, so
its bias rises. The magnitude-only definition in G0a was therefore a mis-transcription of the
cited signal rather than the signal itself. We still treat the correction as a method change,
because it was made after seeing G0a fail.

**Risk stated in advance.** The local loss is class-balanced (effective-number weights), which
deliberately removes most count information: a centre holding 4 images of a class is weighted
as if the class were balanced. So e_M1 may detect *presence* well and *rank counts* poorly,
and can still fail the Spearman-vs-counts gate. That outcome will be reported as is.

**Decision rule, fixed now.**
- If D1 (the retry) passes: G0a passes; M1 is reported only.
- If D1 fails and M1 passes: by the design doc's letter G0a has **failed** and the project
  goes to Fallback F1. We will report that verdict, and in addition carry EARN forward on M1
  as a clearly labelled amended track, so the team can choose the framing with both sets of
  numbers in hand. Every EARN/Camp A number produced on M1 is labelled "amended evidence (M1)".
- If both fail: G0a has failed; Fallback F1 only. No further evidence variants will be tried.

**Status.** Pre-registered in `scripts/06b_gate_g0a_retry.py` before its first run.

---

## R1 - Result: G0a retry and M1 (23 Sep 2026)

`scripts/06b_gate_g0a_retry.py` -> `results/gate_g0a_retry.json`. Mean Spearman, threshold 0.7:

| Evidence | Role | Statistic | Holder AUROC |
|---|---|---|---|
| weight row, round 1 (original G0a) | failed 20 Sep | -0.210 | - |
| bias row, round 1 | D1 - the retry | **-0.463** | 0.137 |
| weight row, rounds 1-3 averaged | reported only | -0.240 | 0.145 |
| signed bias, round 1 | D2 - amendment M1 | 0.110 | 0.738 |

**Verdict under the rule fixed in D2: G0a FAILED; M1 failed too -> Fallback F1 only.** No
further evidence variants will be tried. Both magnitude retries came out inverted, as the D2
mechanism predicted. M1 separates holders from non-holders better than chance (AUROC 0.74)
but does not rank counts, the risk stated in advance.

---

## D3 - G0b retry: unfreeze the last dense block (23 Sep 2026)

**What.** The design doc's G0b retry ("unfreeze the last dense block and re-extract features.
One retry"), made concrete before running:

- Backbone: ImageNet DenseNet-121. Everything up to and including `transition3` stays frozen;
  `denseblock4` + `norm5` are trained, with a fresh 1024->8 classifier.
- Trained **federated** (FedAvg through the same aggregation interface, BatchNorm running
  statistics averaged, `num_batches_tracked` copied), on S1 **training** images only, all 6
  centres: 20 rounds x 50 local steps, batch 32, Adam (block lr 1e-4, classifier lr 5e-4),
  class-balanced loss, seed 42. No augmentation (inputs are cached `transition3` outputs).
  The test split is never touched.
- The trained block (eval mode) then re-extracts the 1024-d features for train and test
  (`data/features/fed_isic2019_densenet121_ft4_*.npz`); the classifier used during
  fine-tuning is discarded.
- G0b is then re-measured exactly as before: Tier A FedAvg on S1, TierAConfig defaults (40
  rounds), seeds 42/43/44, final-round balanced accuracy, mean over seeds, pass at >= 0.45.

**Why federated.** Pooling all hospitals' images to fine-tune would contradict the setting.

**Status.** Pre-registered in `scripts/08_g0b_retry.py` before its first run.

## D4 - Longer training, "more rounds" (23 Sep 2026) - AMENDMENT (status report C3)

**What.** All Tier A experiments from here on use **100 rounds** instead of 40, on the D3
features, whatever the D3 outcome. G0b is additionally reported at 100 rounds on both feature
sets, labelled as the amendment; the 40-round D3 number is the one that decides G0b.

**Why.** The 20 Sep baseline was still climbing at round 40 (0.397 at round 30, 0.415 at 40).
A comparison against an unconverged baseline mostly measures convergence speed. This was
decided after seeing that result, so it is an amendment, not part of the pre-registered
fallback.

**Status.** Fixed before any 100-round run exists.

---

## R2 - Result: G0b retry (23 Sep 2026)

`scripts/08_g0b_retry.py` -> `results/gate_g0b_retry.json`. Federated fine-tuning of
denseblock4 + norm5 took 12.8 min on Apple MPS.

| Features | Rounds | Role | Balanced acc (mean ± sd, 3 seeds) | Rare macro-F1 |
|---|---|---|---|---|
| frozen | 40 | G0b, failed 20 Sep | 0.415 ± 0.005 | 0.332 |
| **ft4** | **40** | **D3 - the retry** | **0.535 ± 0.002** | - |
| frozen | 100 | D4 amendment | 0.450 ± 0.007 | 0.348 |
| ft4 | 100 | D4 amendment (used from here on) | 0.536 ± 0.003 | 0.592 |

**Verdict: G0b PASSES on its pre-registered retry.** Disclosed limitations: the block was
fine-tuned once (seed 42), without attackers, on the S1 assignment of the training images; S2
reuses the same features (same images, different centre assignment). Attacks act on the
classifier head only.

---

## D5 - Scope after G0a: Fallback F1, plus EARN as a labelled exploratory analysis (23 Sep 2026)

**Decision (Arya, 23 Sep).** The paper's main result follows the design doc: **Fallback F1**,
an empirical study of how rare-class-aware aggregation behaves on a real hospital split,
including the never-cut items (the G1 attack experiment on S1 and S2, honest negative
results). In addition, EARN is built in full (aggregator, ablations, ledger, contract, gas and
latency) and run on an **oracle evidence signal** (D6). Every EARN number is labelled
"exploratory - oracle evidence" and none is presented as validating EARN, because the signal
EARN needs does not exist on our data (R1). Tier B and BOBA are cut (design doc cut lines 1
and 2); the real chain is kept (cut line 3 not needed).

## D6 - Pre-registration of the F1 study, G1, and the exploratory EARN runs (23 Sep 2026)

Fixed before any of these runs exists. Code: `src/attacks/attacks.py`,
`src/federated/earn.py`, `src/experiments/grid.py`, `scripts/09_run_grid.py`,
`scripts/10_analyse.py`.

**Common.** Features `ft4` (D3), Tier A, 100 rounds (D4), all other TierAConfig defaults;
seeds 42, 43, 44; splits S1 (natural) and S2 (`src/data/splits.py`, fixed seed); rare classes
5 (Dermatofibroma) and 6 (Vascular lesion); final-round global head on the pooled test split.

**Attacks** (constants in `src/attacks/attacks.py`). Target class: Melanocytic nevus (1).
- A1 sudden expert - attacker **centre 4** (no rare images on S1 or S2), every round: trains
  on rare->nevus flipped labels, then rewrites both rare head rows to norm 3x its largest
  honest row norm along minus its unit mean feature; rare biases move by -3x its largest
  honest bias change.
- A2 sleeper - attacker **centre 1** (a real holder of both rare classes on S1), honest for
  rounds 1-15, A1 behaviour from round 16.
- A3 scaling - attacker **centre 4**, flipped-label update multiplied by 10, every round.
- none.

**F1 methods** (main study): `fedavg`, `fedavg_clipped`, `krum` (f=1), `multi_krum` (f=1),
`trimmed_mean` (trim 1), `coordinate_wise_median`, `camp_a` (the design doc's Camp A:
evidence = head weight-row norm), and **`camp_a_reported`** - an addition: Camp A whose
evidence is each hospital's *reported* class counts (the cwFedAvg / FedSat / MLFCIL style,
design doc Section 3 row 1). An attacker that claims to be a holder (A1 always, A2 from
round 16) reports, for each rare class, the largest honest count of that class on that split;
otherwise every hospital reports its true training counts.

**Measured per run.** Balanced accuracy, macro-F1, F1 of each rare class, rare macro-F1;
attack success = share of rare-class test images predicted as nevus; the attacker's weight on
each rare head row averaged over the rounds its attack is active; centre 2's weight on each
rare row averaged over all rounds; the attacker's FedAvg weight (its training-size share).

**G1 (design doc 6.6, operationalised).** For split s and rare class c, under `camp_a`:
ratio = attacker's mean rare-row-c weight under A1 / its FedAvg weight;
dF1 = F1_c(no attack) - F1_c(A1); dBA = balanced acc(no attack) - balanced acc(A1); all means
over the 3 seeds. **G1 passes on s if some rare c has ratio >= 2 and dF1 >= 0.15, with
dBA < 0.03.** The same numbers for `camp_a_reported` are reported, not gating. The S1/S2
framing then follows the design doc's Section 6.2 table.

**Exploratory EARN (oracle evidence; not a gate).** Evidence = the reported counts used by
`camp_a_reported` (so an attacker can spoof it exactly as there); holder = at least 20 images
(tau = 19.5); EARNConfig defaults (step 0.1, halving, agreement 0.5, cap 0.5, divisor 4,
lam 1). Variants:
- `earn` (full), ablations `earn_no_blend`, `earn_no_ramp`, `earn_no_ledger` (the attacker's
  history is editable);
- sensitivity: `earn_step0.05`, `earn_step0.2`, `earn_tau10` (>= 10 images), `earn_tau40`
  (>= 40 images);
- `earn_real_signal`: EARN on the design doc's own evidence (weight-row norm, holder = above
  the class's mean evidence that round) - shows what EARN does with the signal that failed G0a.

All with the same attacks, splits and seeds. **G2-oracle** (reported with the design doc's G2
wording, explicitly not the gate, which is not evaluable after G0a failed): under A1 and A2,
`earn` rare macro-F1 within 0.05 of its no-attack value; with no attack, within 0.02 of
`camp_a_reported` and above `fedavg`; centre 2's trust on both rare classes >= 0.8 at round 15
with no attack. Evaluated per split.

**Output format (status report C7)** - `docs/results_format.md`.
