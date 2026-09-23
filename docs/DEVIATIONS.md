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
