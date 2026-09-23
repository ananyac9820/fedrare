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
