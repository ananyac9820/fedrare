# Who Gets a Say on Rare Diseases? An Attack Study of Rare-Class-Aware Federated Aggregation on a Real Hospital Split

*Draft, 23 September 2026 - for team review. All numbers are from `docs/results/`; every
experimental choice was fixed before its run (`docs/DEVIATIONS.md`).*

## Abstract

Federated learning lets hospitals train a shared diagnostic model without moving patient images,
but standard federated averaging (FedAvg) weights each hospital by its data size, not by what it
knows. On the six real hospitals of Fed-ISIC2019 we measure that the hospital holding the most
rare-disease data relative to its size receives 1.87x less influence than its share of those
images. Two families of fixes exist: *rare-class-aware* weighting, which gives more say to hospitals
that appear to hold a disease, and *Byzantine-robust* aggregation, which distrusts outliers. We
pre-registered a study of both under three attacks - a hospital faking rare-disease expertise, a
sleeper that turns after 15 honest rounds, and update scaling - on the natural split and on a
constructed split in which a single specialist holds almost every rare case. In 408 runs we find
that (i) reading a hospital's disease holdings from its model update, the signal several
contribution-estimation methods rely on, is inverted on real data (Spearman -0.21): hospitals that
never see a disease change its classifier row the most; (ii) weighting by that signal gives the
specialist *less* say than FedAvg; (iii) weighting by reported class counts is the most accurate
method without attack but lets a lying hospital capture up to 7.4x its fair weight when coverage is
low; (iv) Byzantine-robust filters can exclude a lone specialist entirely. We also built EARN, a
coverage-aware, ledger-anchored trust mechanism we had proposed; because its evidence signal failed
its pre-registered check, we report it only on an oracle signal, where it too fails, and we explain
why. A smart contract enforcing a maximum trust step costs about 184k gas and 1.1 ms per round on a
local chain.

## 1 Introduction

A rare disease is, by definition, held by few hospitals. In Fed-ISIC2019 [19] the two rarest
classes - dermatofibroma (1.03% of images) and vascular lesion (1.09%) - are held (>= 20 training
images) by four and three of six centres. FedAvg weights centre 2, which holds 27.0% of the
rare-disease training images, at 14.5%: a 1.87x gap between knowledge and influence.

Existing work pulls in two directions. *Camp A* - per-class or contribution-aware weighting
(MLFCIL, cwFedAvg, FedSat, CELM, CARE-FL [1,2,16-18]) - gives more influence to whoever appears to
be an expert on a class. *Camp B* - Byzantine-robust aggregation (Krum, trimmed mean, median, BOBA
[6]) - reduces the influence of whoever looks unusual. Camp A is exposed to a hospital that merely
*looks* like an expert; Camp B is exposed to the fact that a genuine expert on a rare disease looks
unusual by nature. The quantity both skip is **coverage**: how many hospitals are able to check a
claim about a disease. It is a property of the disease, and it is lowest exactly where data is
scarcest.

Our original plan was to build EARN, which blends peer checking and self-history checking in
proportion to coverage and anchors the history on a blockchain. We fixed pass/fail gates before
running anything. The first gate - whether a hospital's disease holdings can be read from its model
update - failed, and so did its one allowed retry. Following our own protocol, this paper is the
fallback: an empirical attack study of rare-class-aware aggregation on a real hospital split. We
report EARN's implementation and an oracle-signal analysis separately and label it exploratory.

**Contributions.** (1) A measurement, on real hospital data, that update-norm "evidence" of class
holdings is inverted under standard local training, with the mechanism. (2) A pre-registered
comparison of eight aggregation rules under three attacks on a natural and a specialist split,
reporting rare-class F1, attack success, and the specialist's share of influence. (3) A negative,
mechanistic result for coverage-aware trust (EARN) even with an oracle signal. (4) Gas and latency
for an on-chain maximum-trust-step rule on 2,400 real rounds.

## 2 Related work

*Per-class weighting* (MLFCIL, cwFedAvg, FedSat, C3E) assumes honest clients. *Contribution
estimation without validation data* (CELM, logit-maximisation probes) freezes weights after
warm-up. *Maverick-aware valuation* (FedEMD, FedMS) uses Shapley values without an attack model.
*Label-distribution inference* from output-layer updates (Ramakrishna & Dan; HiCS-FL) is the signal
our evidence measure was modelled on. *Reputation* (FLARE, TAIM) keeps one score per client;
BRFLATA uses slow-increase/fast-decrease trust at client level; attestedFL compares a client with
its own past but needs server-side validation data. *Attacks on contribution scores* (ACE, USENIX
Security 2024) inflate client-level reward scores. *Blockchain FL* (BFLC, BFEL) uses committees or
logs. We found no method that measures per-disease coverage and adapts verification to it; we do not
claim this is proof that none exists (final literature re-check pending).

## 3 Setting

**Data.** Fed-ISIC2019 via the Hugging Face mirror of FLamby: 23,247 dermoscopy images, 6 real
centres, 8 classes, 18,597 train / 4,650 test. Rare classes are selected by an imbalance-ratio
rule (share under 1/25 of the largest class, cutoff 1.95%) chosen after seeing the counts; we
disclose this. Centre 4 holds no image of either rare class.

**Splits.** *S1*: the natural split (rare coverage 4 and 3). *S2*: constructed - 90% of every
other centre's rare-class training images moved to centre 2 (fixed seed), so coverage is 1 for both
rare classes; it models a specialist clinic holding nearly all cases of an ultra-rare condition.

**Model (Tier A).** DenseNet-121 (ImageNet) with the last dense block and final norm fine-tuned
once by FedAvg on S1 training images (20 rounds; the pre-registered G0b retry), then frozen; every
experiment trains the 1024->8 classifier head federated: 6 clients, 100 rounds, 50 local Adam steps
(batch 32, lr 5e-4), class-balanced loss. Head row *c* (weights + bias) scores disease *c*.
Evaluation: final global head on the pooled test set; 3 seeds (42-44); mean ± s.d.

## 4 Methods compared

FedAvg; FedAvg with median-norm clipping; Krum and Multi-Krum (f = 1); coordinate-wise trimmed
mean (1 per side); coordinate-wise median; **Camp A (update evidence)** - each head row weighted by
size share plus evidence share, evidence e(k,c) = L2 norm of client k's clipped update to row c,
capped at 50% per client; **Camp A (reported counts)** - the same with evidence = the class counts a
client reports (cwFedAvg/FedSat style).

**EARN** (exploratory). Per round: clip to the median norm; evidence; holders(c) = {e > tau};
coverage n(c); peer confidence p(c) = clip((n(c) - 1)/4, 0, 1); for each holder, agreement
a = p · cos(row, median of other holders' rows) + (1 - p) · cos(row, own running-mean history);
trust T += 0.1 if a >= 0.5 else T ×= 0.5; head row c weighted by size share + T · evidence share,
capped at 50%. Each round's trust table, coverage and history hash are committed to a hash chain
(and, separately, to the `EarnLedger` contract), which rejects any trust rise above 0.1.

## 5 Threat model and attacks

One attacking hospital; target class nevus. **A1 sudden expert** (centre 4, no rare images): trains
with rare labels flipped, then rewrites both rare rows to 3x its largest honest row norm along minus
its mean feature (lowering those classes' scores), and - where evidence is reported - claims the
largest honest count. **A2 sleeper** (centre 1, a real holder on S1): honest for rounds 1-15, A1
afterwards. **A3 scaling** (centre 4): flipped-label update ×10.

## 6 Protocol

Gates and every experimental constant were committed before the corresponding run
(`docs/DEVIATIONS.md` D1-D6); each gate allowed one retry. Amendments made after seeing a result
(training 100 rounds instead of 40; a sign-aware evidence variant) are labelled and reported beside
the pre-registered result, never instead of it.

## 7 Results

### 7.1 The evidence signal is inverted (G0a)

With round-1 updates, the per-class Spearman correlation between evidence and true training counts
averages **-0.210** (needed >= 0.7). For both rare classes the highest evidence belongs to centre 4,
which holds none. The pre-registered retry (bias-row magnitude) scores **-0.463**; averaging the
weight-row signal over three rounds, -0.240. *Mechanism:* a centre that never sees class c pushes
c's score down on every step; Adam normalises step size per parameter, so a consistently-signed
gradient moves the row the most. A sign-aware amendment (positive bias change only) separates
holders from non-holders above chance (AUROC 0.74) but still cannot rank counts (0.110): the
class-balanced loss removes count information by design.

### 7.2 Baseline quality (G0b)

Frozen-feature FedAvg reached 0.415 balanced accuracy (needed 0.45). The pre-registered retry -
fine-tuning the last dense block by FedAvg and re-extracting - reaches **0.535 ± 0.002**; at 100
rounds, 0.536 ± 0.003 with rare macro-F1 0.592 ± 0.005.

### 7.3 Is the attack real? (G1)

Under Camp A (update evidence) and A1, the attacker captures 4.96x / 5.53x (S1) and 4.81x / 4.65x
(S2) its FedAvg weight on the dermatofibroma / vascular-lesion rows. On S1 the rare F1 drops are
0.086 / 0.017 - three or four honest holders pull the rows back. On S2 they are 0.207 / 0.237, but
balanced accuracy falls 0.051, above the 0.03 bar for a *quiet* attack. **G1 fails on both
splits**; by the pre-registered framing the study is reported as Fallback F1.

### 7.4 The attack study (Table 1, Figures 1-3)

**Table 1.** Rare macro-F1, mean over 3 seeds. On S1 every standard deviation is <= 0.03; on S2
they reach 0.19 (Multi-Krum under A2). Full values: `docs/results/analysis.md`.

| Method | S1 none | S1 A1 | S1 A2 | S1 A3 | S2 none | S2 A1 | S2 A2 | S2 A3 |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 0.592 | 0.553 | 0.286 | 0.569 | 0.490 | 0.454 | 0.012 | 0.408 |
| FedAvg + clipping | 0.592 | 0.574 | 0.538 | 0.592 | 0.479 | 0.436 | 0.035 | 0.479 |
| Krum | 0.541 | 0.545 | 0.546 | 0.545 | 0.242 | 0.198 | 0.084 | 0.198 |
| Multi-Krum | 0.547 | 0.566 | 0.574 | 0.566 | 0.000 | 0.535 | 0.397 | 0.535 |
| Trimmed mean | 0.560 | 0.555 | 0.564 | 0.554 | 0.338 | 0.323 | 0.012 | 0.425 |
| Median | 0.560 | 0.560 | 0.554 | 0.551 | 0.326 | 0.366 | 0.044 | 0.442 |
| Camp A (update evidence) | 0.558 | 0.507 | 0.491 | 0.558 | 0.527 | 0.305 | 0.000 | 0.527 |
| Camp A (reported counts) | **0.605** | 0.534 | 0.442 | 0.605 | **0.560** | 0.327 | 0.000 | 0.560 |


**Finding 1 - evidence weighting backfires.** Without attack, Camp A (update evidence) gives
centre 2 12.6% / 13.2% of the rare rows on S1, *below* FedAvg's 14.5%, and has lower rare F1 (0.558
vs 0.592). The inverted signal of 7.1 carries through to aggregation.

**Finding 2 - honest counts help, claimed counts are exploitable.** Reported-count weighting is the
best method without attack (0.605 S1, 0.560 S2) and moves centre 2 towards its knowledge share
(50% of the rare rows on S2, the cap). Under A1 the attacker claims the top count and captures
5.40x / 4.79x (S1) and 7.32x / 7.36x (S2) its fair weight; on S2 rare F1 falls to 0.327.

**Finding 3 - the sleeper is the most damaging attack.** A2 cuts FedAvg's rare F1 on S1 from 0.592
to 0.286 (balanced accuracy 0.536 -> 0.448); median-norm clipping recovers most of it (0.538).
On S2, where the sleeper's pre-attack record is as a non-holder, every method except Multi-Krum
(0.397 ± 0.167) falls close to zero.

**Finding 4 - robust filters can erase a lone specialist.** On S1, Krum keeps centre 2's update in
60% of rounds. On S2, without any attack, Multi-Krum gives centre 2 0% of the rare rows and rare F1
is 0.000; Krum 0.242 with centre 2 excluded every round. The "odd one out" is the expert.
Paradoxically, Multi-Krum's rare F1 *rises* under A1 on S2 (0.535): the attacker's update is
more outlying than the specialist's, so it takes the one excluded slot and centre 2's weight goes
from 0% to 20%.

**Scaling (A3).** Clipping neutralises A3 exactly - every clipped method is identical under A3 and
without attack - because the attacker holds no rare images, so its label flip is a no-op and the
x10 is undone. Unclipped FedAvg loses 0.047 balanced accuracy on S1 (0.536 -> 0.489).

### 7.5 EARN on an oracle signal (exploratory)

Replacing the failed signal with true (but spoofable) class counts, EARN matches reported-count
Camp A without attack (0.596 vs 0.605 on S1; 0.555 vs 0.560 on S2) and beats FedAvg, but does not
meet the G2 conditions: on S1 it loses 0.055 rare F1 under A2 and centre 2's trust at round 15 is
0.04 / 0.19; on S2 it loses 0.17 under A1 and all of it under A2. The two mechanisms (Figure 3):

- *S1, coverage 3-4:* honest holders' rare-row updates agree weakly with each other (centre 2's
  mean agreement 0.14 / 0.12 against the 0.5 bar), so the peer check rejects honest hospitals and
  EARN stays close to clipped FedAvg. The A2 sleeper, which earned trust 1.0 on vascular lesion in
  rounds 1-15, is caught immediately when it turns (trust halves to ~0 within a few rounds).
- *S2, coverage 1:* only the history check runs, and it verifies *consistency*, not correctness.
  The A1 attacker is consistent from round 1 and reaches full trust by round 10 (7.4x weight).
  Peer-only checking (no blend) holds the attacker to 1.05x but also zeroes the honest specialist.

Ablations: removing the slow ramp raises the attacker's weight (S1 A1 4.0x vs 2.2x); making the
history editable changes nothing measurable, because neither attacker needed to rewrite its past.
Using EARN on the real (failed) signal is worse than FedAvg (S1 0.563; S2 0.442).

### 7.6 Cost of the ledger

2,400 real EARN rounds (24 runs) were committed to `EarnLedger` on a local Hardhat chain: **183,781
gas per round** on average (median 183,478, max 282,365; deployment 667,469) and **1.13 ms** send-
to-receipt (local automine; a public chain adds its block time). On-chain trust equalled the
Python ledger's in every run, and a forged +0.3 boost was rejected in every run. EARN's aggregation
takes 3.8 ms per round against 0.3 ms for FedAvg.

## 8 Discussion

The common thread is **coverage**. When three or four hospitals hold a disease, honest holders
dilute an attacker (S1), and the attacks that hurt are those that come from a trusted insider (A2).
When one hospital holds it, every mechanism we tested must either trust a claim nobody can check
(Camp A, EARN's history branch) or distrust the only expert (Krum, Multi-Krum, EARN's peer-only
ablation). Self-history is not a substitute for peers: a patient or consistent attacker passes it.
A signal of class holdings that is both *faithful* and *hard to fake* is the missing ingredient;
update norms are neither under standard local training.

On the blockchain question, the contract does the one thing that is cheap to verify - bounding how
fast the coordinator may raise trust - and we measured it. The history-locking role did not matter
against the attacks we tested; a stronger adaptive attacker that rewrites history is future work.
A signed log held by a trusted third party would achieve much of this; a federation of competing
hospitals is the setting in which no such party exists.

## 9 Limitations

Tier A only (a head on features; the last block fine-tuned once, seed 42, without attackers); Tier B
and BOBA were cut. One attacker at a time; no collusion. S2 is constructed. The rare-class rule
was chosen after seeing counts. The 100-round schedule and the sign-aware evidence variant are
amendments made after seeing earlier results (logged beforehand). Local chain only. The design
doc's statement that common diseases have 5-6 holders is wrong for three of them (BCC and AK 3,
SCC 2), which strengthens rather than weakens the coverage argument.

## 10 Conclusion

On a real hospital split, the obvious ways to give rare-disease experts more say either read the
wrong signal, trust unverifiable claims, or remove the expert. Coverage - how many hospitals can
vouch for a disease - predicts which of these failures appears. We release the pre-registration,
all 408 runs, the contract, and a site that shows every result including the failures.

## References

See EARN Project Design v2, References [1]-[19]; FLamby (Ogier du Terrail et al., NeurIPS 2022
Datasets & Benchmarks); Blanchard et al., Krum (NeurIPS 2017); Yin et al., trimmed mean / median
(ICML 2018); Cui et al., class-balanced loss (CVPR 2019).

## Figures

1. `docs/results/figures/rare_f1_by_method_attack.png` - rare macro-F1 by method and attack, S1 and S2.
2. `docs/results/figures/attacker_weight_ratio_A1.png` - attacker's rare-row weight / FedAvg weight under A1.
3. `docs/results/figures/earn_trust_curves.png` - EARN (oracle) trust of centre 2 and of the attackers.
4. `docs/results/figures/specialist_weight.png` - centre 2's rare-row weight vs its share of the images.
5. `docs/results/figures/learning_curves.png` - rare macro-F1 per round, no attack.
