# Does the F1-concealment split replicate? No - it reverses

Pre-registered as **D14** before the battery ran. Raw numbers: `concealment_replication.json`,
`dermacon_concealment.json`. Scripts: `scripts/23_dermacon_concealment.py`,
`scripts/24_compare_concealment.py`.

## The answer

| dataset / damage metric | selection rules (Krum, Multi-Krum) | averaging rules |
| --- | --- | --- |
| **Fed-ISIC2019** vs `to_target` | **14 of 52 = 26.9%** [16.8-40.3%] | **0 of 320 = 0.0%** [0.0-1.2%] |
| **DermaCon-IN** vs `to_target` | 1 of 23 = 4.3% [0.8-21.0%] | **9 of 54 = 16.7%** [9.0-28.7%] |
| **DermaCon-IN** vs exact recall | 0 of 19 = 0.0% [0.0-16.8%] | **15 of 62 = 24.2%** [15.2-36.2%] |

The pre-registered verdict rule fired on the third outcome it listed: **refuted**. Averaging rules
disagree substantially on DermaCon-IN, so the claim that "averaging never does this" does not hold
outside Fed-ISIC2019. The pattern does not merely fail to replicate - it **reverses**: the family
with the near-zero rate on one dataset has the higher rate on the other, on both damage metrics.

**The proposed mechanism does not survive this.** D13 explained the split by selection rules
adopting one client's update outright, so the model jumps discretely while averaging moves
continuously. On DermaCon-IN the continuous rules are the ones producing disagreement, and the
discrete ones produce none. Whatever drove the Fed-ISIC2019 pattern, it was not that.

## What the Fed-ISIC2019 result now means

It stands as measured and is **dataset-specific**. The honest statement is: *on Fed-ISIC2019, F1 and
`to_target` disagreed in 26.9% of selection-rule cases and never in 320 averaging-rule cases; on
DermaCon-IN the same battery gives the opposite arrangement.* Any sentence of the form "selection
rules are where F1 misleads" is not supportable from these two datasets together, and the paper and
site should not carry one.

## Where DermaCon-IN's disagreements are

Fifteen cases, measured against exact recall:

| by method | | by attack | | by class | |
| --- | --- | --- | --- | --- | --- |
| camp_a_reported | 7 | A3 | 6 | Neoplasms and tumors | 8 |
| earn | 3 | A1 | 5 | Keratinisation Disorders | 7 |
| trimmed_mean | 2 | A2 | 4 | | |
| coordinate_wise_median | 2 | | | | |
| fedavg | 1 | | | | |

Ten of the fifteen come from the two evidence-weighted rules (Camp A and EARN). That is an
observation from this one battery, not a claim - it was not predicted, and no test here was
designed to support it.

The magnitudes are small, as they were on Fed-ISIC2019: the median concealed case is a **+0.028**
rise in F1 against **8.3 points of recall lost**, while the 64 agreement cases move F1 by a median
of **-0.086**. Concealment, where it happens, hides a modest number and the metric still moves
much further when the two agree.

## The over-prediction explanation, tested properly - and it fails

D13 could only test this on sixteen cells. The DermaCon battery stores confusion matrices for all
96 runs, so precision is exact for **81 cases with clear movement**. This test was **not**
pre-registered in D14 and is therefore post-hoc and exploratory - but it is the first time the
hypothesis has had a usable sample.

- mean honest-baseline precision: **0.120** where F1 and recall disagree (n=15), **0.126** where
  they do not (n=66);
- point-biserial correlation between honest precision and disagreement: **-0.038** (n=81);
- the honest model over-predicts the rare class in **72 of 81** cases, and only **15 of those 72**
  show disagreement.

So over-prediction is nearly universal in this setting and almost never produces the effect. The
mechanism proposed in the R11 addendum - F1 rising because the attack suppresses an over-predicting
baseline - explains the single Tier B cell it was drawn from, and does not generalise.

## Honest assessment

What survives: F1 and recall-style damage metrics **can** move in opposite directions under attack,
in both datasets, at single-digit-to-quarter rates depending on where you look. If you care whether
an attack worked, read recall or the share pushed to the target class, not F1 alone.

What does not survive: that the effect belongs to selection rules (reversed here), that averaging
rules are safe from it (16.7-24.2% here), and that baseline over-prediction explains it (no
relationship at n=81).

Differences between the two batteries, recorded rather than controlled away: 40 rounds here against
100 there; 7 classes against 8; constructed age-band centres against six real hospitals; 8 methods
against 17; and different attacker centres, chosen to preserve each attack's role. Any of these
could matter, and this study cannot say which does.

## Figure

`figures/concealment_replication.png` - the three panels above, side by side, with 95% intervals
and case counts on each bar.
