# Design note: why rare-class claims cannot be verified, and what to do instead

For discussion before any further compute. 14 October 2026. Nothing here has been run; the
pre-registered test at the end is a proposal, not a result.

---

## 1. The finding: four ways to check a claim, all closed

EARN, and every rare-class-aware method in the literature, needs to answer one question: **does
this hospital really hold cases of this disease, or is it saying so to gain influence?** There are
four ways a federated server could answer it. We have now tested all four, and each is closed.

**Channel 1 - infer holdings from the update itself.** This was the design doc's method and gate
G0a. It does not work: the signal is *inverted* (Spearman -0.21 against true counts; the one
allowed retry -0.46). We then searched for a replacement across two pre-registered passes and ten
definitions - magnitude, signed magnitude, per-client normalised, direction against peers with and
without self-inclusion, weighted and unweighted, single-round and averaged, rank-combined, and a
probe-based variant - over three seeds. The best faithful candidate reaches **0.391 +- 0.034**
against a required 0.7. A candidate that appeared to reach 0.448 turned out to be measuring hospital
*size*: comparing each hospital against the other five, instead of against a group containing
itself, took it from +0.448 to **-0.020**. (R9, R14.)

**Channel 2 - ask the other hospitals.** This works when several hold the disease and fails exactly
where it matters. Measured holders per class in Fed-ISIC2019: **6, 6, 3, 3, 5, 4, 3, 2**. For the
rarest class only two hospitals can vouch at all, and in the specialist scenario the project is
built around, the number is one. There is nobody to ask.

**Channel 3 - look for collateral damage.** The attacks relabel rare-disease images as a common
class that everyone holds, so in principle the lie should leave a mark on classes everyone can
check. It does not. Across 306 attacked runs paired with their honest counterparts:

| attack | change in rare-class F1 | change in common-class F1 |
| --- | --- | --- |
| A1 sudden expert | **-0.065** | +0.012 |
| A2 sleeper | **-0.247** | +0.010 |
| A3 scaling | +0.018 | +0.006 |

The rare classes are destroyed and the common classes are untouched - slightly *better*, if
anything. Relabelling forty rare images is invisible against 2,242 test images of the common class.
Only 18-43% of runs show any common-class drop at all. There is no fingerprint to audit.

**Channel 4 - check the hospital against its own past.** EARN does this, with the history locked on
a blockchain so it cannot be rewritten. It detects a liar and does not stop one. In the sleeper
test, the attacker's trust in the rare class falls from 1.00 to 0.06 within four rounds of it
turning - and rare-class F1 still collapses from 0.589 to 0.018. Detection is not protection. Worse,
a hospital that lies *consistently* passes a consistency check by definition. (R4, R8, D7.)

**Taken together this is a boundary result:** at coverage 1, a rare-class claim cannot be verified
through any channel available to a federated server. It is not that our method failed to verify it;
it is that the information is not there to be had.

---

## 2. Why this matters beyond our project

Rare-class-aware weighting is an active area, and the published methods we reviewed in full give
more influence to whoever *appears* to hold a rare disease, without a mechanism for checking the
appearance:

- **Camp A / reported counts** - the family our baseline stands in for. In our attack study a
  hospital holding **no** cases of either rare disease captured up to **7.4x** the weight its data
  size would earn it, simply by claiming to be a specialist.
- **CELM** (2026) uses per-class probes and uses "class coverage" as a normalising term, and states
  explicitly that it does not address Byzantine clients.
- **CARE-FL / DFed-MCDA** (Neurocomputing 2026) weights by a per-class sample term and an
  inverse-frequency rarity factor; the abstract describes no attack model.

None of them is wrong about the problem - rare classes genuinely are under-served by size-weighted
averaging, which is our own 1.87x finding. The gap is that the fix they propose is only safe where
a claim can be checked, and nobody has measured where that is. We have.

---

## 3. The proposal: stop inferring, start attesting

If evidence cannot be inferred and cannot be verified, it has to be **declared** - and the design
question becomes how to make a declaration accountable rather than free.

**The mechanism.** Each hospital declares its per-class case counts once, at enrolment. The
declaration is signed and committed to the ledger EARN already has. It cannot be revised later.
Trust then audits behaviour *against that fixed declaration* over time, and weight follows trust as
it does now.

**What this changes relative to Camp A**, which is the same family:

| | Camp A today | attested declaration |
| --- | --- | --- |
| who states the evidence | the hospital, every round | the hospital, once |
| can it be revised | yes, silently | no - the commitment is immutable |
| is it attributable | no | yes, signed and timestamped |
| cost of a false claim | none | a permanent, auditable record |

This also settles a question we had about the blockchain component, which has looked like an
expensive accessory. Under this design it is the only thing that makes a declaration safe to use:
without an immutable, attributable record, a declaration is just Camp A, which we have already shown
is captured 7.4x over.

---

## 4. What attestation does not solve - stated plainly

**A hospital that declares falsely at enrolment and then behaves consistently with its own lie will
pass.** Trust audits consistency, not truth, and a consistent liar is consistent. Attestation
raises the cost of lying and removes the ability to revise the lie; it does not detect it.

Three further limits worth being honest about now:

- it requires something outside the training loop - a registry figure, an ethics submission, an
  institutional signature - so it is a governance mechanism, not a purely technical one;
- it cannot be validated against a real adversary in this project, only against the simulated
  attacker we have;
- if a hospital's case mix genuinely changes, an immutable declaration becomes stale, and we have no
  mechanism for revision that an attacker could not also use.

The honest claim is therefore narrow: attestation converts an unverifiable inference problem into an
accountable declaration problem. It does not make rare-class weighting safe. It makes the unsafe
part visible, attributable, and bounded.

---

## 5. What changes in the project

**The title does not change.** "Who Gets a Say on Rare Diseases? An Attack Study of Rare-Class-Aware
Federated Aggregation on a Real Hospital Split" already describes what was done.

**The objectives do not change.** Build the mechanism, test it against attacks, measure it on a real
hospital split - all done, all reported.

**One word shifts in the aim.** From *"EARN makes rare-class aggregation safe"* to *"coverage
determines whether rare-class aggregation can be made safe - we establish where the boundary is, and
what must be added to cross it."* Every negative result then becomes evidence for the thesis instead
of a shortfall against it, because that is what they actually are.

Nothing is retracted. The gates stand as recorded, including the failures.

---

## 6. The test we would run, with its bar set in advance

**Setting.** Split S2, where one hospital holds almost every rare case and coverage is 1 - the
hardest case and the one the boundary result is about. Attacks A1, A2, A3 and the specialist-sleeper
A2s; seeds 42, 43, 44; 100 rounds. Two arms: **Camp A** (declarations free and re-stated each round,
as published) against **attested EARN** (declaration committed once at round 1, immutable, trust
auditing behaviour against it).

**Pass/fail, fixed now.**

1. The attested arm reduces the attacker's captured rare-row weight by at least **0.10** relative to
   Camp A, averaged over seeds, on at least three of the four attacks.
2. With no attack, the honest specialist keeps at least its size share of the rare rows under the
   attested arm, and rare-class macro-F1 is within **0.05** of Camp A's.
3. Rare-class **recall** is reported alongside F1 throughout, because this project has already shown
   F1 can move opposite to real attack damage.

**A prediction recorded in advance, which we expect to fail.** Against a *consistent* liar - one
that declares false counts at enrolment and then behaves exactly as that declaration implies -
attestation should give **no** improvement over Camp A. We expect condition 1 to fail for that
attacker specifically. Writing this down now means the result cannot later be presented as a
surprise, and it is the honest boundary of what the mechanism can do.

**Cost.** About 24 runs, two to three hours on a laptop CPU. No new data.

---

*Prepared by the pipeline track. Supporting evidence: `docs/DEVIATIONS.md` entries R4, R8, R9, R14;
`docs/results/evidence_signal_v2.json`; `docs/results/runs.csv` (408 pre-registered runs). A separate
test of a coverage-blend change (D16) is running and will be reported independently of this note.*
