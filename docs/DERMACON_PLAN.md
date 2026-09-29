# DermaCon-IN: what we will do with it, written before we run anything

Dated 2026-09-29. Everything in this file was fixed before any model was trained on this
dataset, in the same way as the G-gates in `PROJECT_PLAN.md`. Results go in
`docs/results/`; anything we change afterwards goes in `DEVIATIONS.md` with the reason.

## Why this dataset

We were asked to include an Indian dataset. DermaCon-IN is the only public one that can
carry this project: 5,450 clinical photographs from 3,002 patients, collected across three
tertiary-care hospitals and affiliated clinics in North Karnataka, annotated by
board-certified dermatologists.

> Madarkar et al., *DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of
> Indian Skin Disorders for Clinical AI Research*, NeurIPS 2025 Datasets & Benchmarks.
> doi:10.7910/DVN/W7OUZM, CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution).

The two alternatives were rejected on size and scope: SkinDis-India covers only psoriasis
and vitiligo, and SkinDisNet is 1,710 images over 6 classes, collected in Bangladesh.

What it adds that Fed-ISIC2019 cannot: Fitzpatrick 3-5 covers 97% of its images (FST 4
alone is 48%), against an ISIC 2019 population that is overwhelmingly FST 1-2. It is also
clinical photography from smartphones rather than dermoscopy, so it tests the pipeline on
the kind of image a clinic actually produces.

## What it cannot support, and what we will not claim

**The release has no hospital or site column.** Three hospitals collected it; the metadata
(`Image_name, Subject_ID, Quality, Sex, Age, Gradability, Fitzpatrick, Monk_skin_tone,
Body_part, Descriptors, Main_class, Sub_class, Disease_label, Confidence`) does not record
which one took a given photograph, so the real split is not recoverable.

Therefore: every centre split on DermaCon-IN is **constructed by us**. In the paper, on the
site and in talks it is called a constructed split, never "hospitals". Fed-ISIC2019 remains
the only dataset in this project that carries a real-hospital claim, and the 1.87x finding
stays attached to it alone. The two datasets are reported side by side and never pooled.

## Fixed choices

**Labels.** `Main_class`, dropping `No Definite Diagnosis` (37 images) because it is not a
disease. That leaves **7 classes**, against Fed-ISIC2019's 8.

**Rare classes.** The project's existing rule, unchanged: a class is rare when its share is
below the head class's share divided by 25. Head is Infectious Disorders at 40.86%, so the
cutoff is 1.63%. That selects **Neoplasms and tumors (52 images, 0.95%)** and
**Keratinisation Disorders (72 images, 1.32%)**. Measured, not chosen.

**Train/test.** The split shipped with the dataset: 4,399 train / 1,051 test, stratified by
sub-class and cut by patient, so no patient appears on both sides. We verify that before
using it rather than trusting it.

**Splits into centres.** Six each, to match Fed-ISIC2019's six:

| id | rule | why |
| --- | --- | --- |
| D0 | patient id hashed into 6 groups | near-IID control: largest rare-share / size-share ratio 1.24x. EARN should do nothing here, and a method that "helps" is misfiring |
| D1 | six age bands (0-10, 10-20, 20-40, 40-60, 60-80, 80-100) | naturally uneven: the 60-80 band is 8% of the images but holds 1.95x its share of rare cases. An older cohort carrying more neoplasms is clinically coherent, not an artefact we imposed |

Both ratios were measured from the metadata before any training - `scripts/15_explore_dermacon.py`.

**Model and training.** Unchanged from Tier A: frozen ImageNet DenseNet-121, 1,024-d
features extracted once, a single linear head trained federated with a class-balanced local
loss. Baseline runs use 40 rounds, seed 42; the grid uses 100 rounds, seeds 42/43/44.

## Gates, with their thresholds set now

**DG0 - is the baseline good enough to study?**
FedAvg on D1, 40 rounds, seed 42: **balanced accuracy >= 0.40** over the 7 classes.
One retry allowed, and it is specified now: 100 rounds, same seed.
If DG0 does not clear, we do not report EARN results on this dataset. It is then used for
external validation only - running the Fed-ISIC2019 model on Indian skin and reporting what
happens - and we say plainly that the baseline did not reach the bar.

**DG2 - does EARN help here?**
On D1 without attack, EARN's rare-class macro-F1 must be **no more than 0.05 below** FedAvg
(not worse), and on the specialist band its weight share must exceed its size share.
On D0 (near-IID) EARN must stay within 0.05 of FedAvg either way - no benefit expected and
none should appear.

There is deliberately no DG1 here. "The problem is real" is a property of the data we have
already measured (1.95x), not something a training run decides.

## What gets reported

Indian results appear as their own section, next to the Fed-ISIC2019 results, with the
constructed-split caveat in the same paragraph as the numbers - not in a footnote. If EARN
does better on one dataset and not the other, both are shown.
