# Does rare-class F1 conceal attack damage?

Pre-registered as **D13** (11 October 2026) and run against result files that already existed - no
new training. Raw numbers: `f1_concealment.json`. Script: `scripts/22_f1_concealment.py`.

## The short answer

**The hypothesis is not supported as a general claim.** Across 612 comparable cases from the 408
pre-registered runs, rare-class F1 moved opposite to the damage metric in **14 cases - 3.8%** of
the 372 where either metric moved at all. D13 fixed in advance how a rate under 10% would be
described, so, in those words: **this can happen under specific conditions, and it is not a general
claim about the field.**

Two things sharpen that, one of which was not anticipated.

**It never happened under any averaging rule.** Zero cases in 320, across FedAvg, clipped FedAvg,
trimmed mean, coordinate-wise median, Camp A, Camp A (reported counts), and every EARN variant.

**All 14 came from the two selection rules.** Krum and Multi-Krum disagree in 14 of 52 cases -
**26.9%, 95% interval 17-40%**.

| rule family | cases with clear movement | disagreements | rate |
| --- | --- | --- | --- |
| selection (Krum, Multi-Krum) | 52 | 14 | **26.9%** [17-40%] |
| averaging (all others) | 320 | **0** | **0.0%** [0-1%] |

That split is mechanically plausible: Krum adopts a single client's update as the global model, so
the model jumps discretely from one client's parameters to another's. A discrete jump moves
precision and recall independently, and F1, being their harmonic mean, can rise while more of the
class is pushed into the attack's target. Averaging rules move the model continuously and never
produced the effect here.

It is worth noting what else that implies: the three rules the grid uses to *resist* attacks are
also the ones where the standard success metric is least reliable at telling you whether the attack
worked.

## The evidence, and how far it stretches

**Scope was bounded before the tests were designed, by what the files hold.** Only
`docs/results/tier_b.json` stores confusion matrices, so true recall and precision exist for
sixteen cases and nowhere else. The grid stores per-class F1 and `to_target_c` - the share of true
class-c images predicted as the target class - but not recall or precision, and they cannot be
recovered without re-running.

| test | data | comparable cases | result |
| --- | --- | --- | --- |
| 1. F1 against `to_target` | grid, 306 attacked runs paired with the honest run of the same split/method/seed, 2 rare classes | 612 (372 moved) | 3.8% disagreement |
| 2. F1 against true recall | Tier B, 2 attacked runs x 4 rounds x 2 rare classes | 16 (7 moved) | 4 of 7 |
| 3. is it explained by honest precision? | Tier B only | 16 | **not supported** |

**Test 2 is too small to carry anything.** Four of seven looks dramatic and means very little at
that count; it is reported because D13 said it would be, not because seven cases establish a rate.

**Test 3 contradicts the proposed explanation.** The hypothesis was that disagreement concentrates
where the honest baseline over-predicts the class, i.e. where its precision is low. In the only
data that can test it, mean honest precision is **0.407 where F1 and recall disagree** and **0.336
where they agree** - the opposite direction, on four cases against three. D13 declared this
descriptive before the run precisely because sixteen cells cannot support a correlation, and no
correlation is computed. The honest summary is that the mechanism proposed in the R11 addendum is
**not supported by the available evidence**, and the available evidence is too thin to rule it out
either.

**The converse is nearly as common as the effect.** F1 fell while the damage metric improved in 14
cases, exactly matching the 14 disagreements. If F1 systematically under-reported damage, the two
directions should not be balanced. Within the selection rules the asymmetry is real but modest -
14 disagreements against 7 converse - and within averaging rules it is 0 against 7.

**`to_target` is an imperfect stand-in for recall**, which D13 recorded in advance as a caveat.
Where both exist (Tier B), they point the same way in **7 of 10** cases. Test 1's conclusions are
therefore about `to_target`, and are worded that way throughout.

## Worked examples

The largest disagreement in the grid, and a typical one:

| | split | attack | method | class | F1 | `to_target` |
| --- | --- | --- | --- | --- | --- | --- |
| largest | S2 | A2 | Multi-Krum | Vascular lesion | 0.000 -> 0.143 | 0.500 -> 0.558 |
| typical | S1 | A2 | Krum | Dermatofibroma | 0.444 -> 0.472 | 0.200 -> 0.255 |

The median disagreement is a **+0.021** rise in F1 alongside a **+0.055** rise in the share of the
class pushed to nevus. For scale, the 344 agreement cases have a median F1 **fall** of 0.111 - so
where F1 and the damage metric agree, F1 moves about five times as far as it does when they
disagree. The concealment, where it happens, is small in magnitude.

The case that motivated the study - Tier B, Dermatofibroma, round 20, F1 0.390 -> 0.518 while
recall fell 0.709 -> 0.527 - remains a real and well-evidenced instance. It is a worked example of
how the effect *can* arise, and this study finds it is not representative.

## What we would claim, and what we would not

Defensible from this analysis:

- In this setting, rare-class F1 and `to_target` disagree in about 4% of attacked comparisons,
  and never once under an averaging aggregation rule.
- Where disagreement occurs it is confined to selection-based robust rules (Krum, Multi-Krum), at
  roughly a quarter of their cases, with a wide interval on 52 cases.
- A single documented case exists where F1 concealed a genuine loss of rare-class recall.

Not defensible from this analysis:

- That rare-class F1 systematically under-reports attack damage in long-tail federated learning.
  It did not do so in 320 averaging-rule cases.
- That over-prediction by the honest baseline explains disagreement. The only data able to test
  that points the other way, on seven cases.
- Any claim about the wider literature. One dataset, one task, one attack family, two rare classes.

**Nothing here re-scores a pre-registered gate**, as D13 required. The gates were decided on the
metrics fixed at the time, and they stand.

## Figures

- `figures/f1_concealment_scatter.png` - change in F1 against change in `to_target` for all 612
  grid cases, with the disagreement quadrant shaded.
- `figures/f1_concealment_precision.png` - the sixteen Tier B cases, honest precision against the
  change in F1, which is the evidence for Test 3 and shows how little of it there is.
