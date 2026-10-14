# How other work solves the verification problem

A targeted review, 14 October 2026, prompted by the boundary result in
`DESIGN_NOTE_attestation.md`: at coverage 1, a rare-class claim cannot be verified through any
channel available to a federated server. The question here is how the literature handles the same
problem, and whether anyone has already solved it.

**The short answer: every method that successfully verifies a client's claim uses information from
outside the training loop.** Four families do it four ways. None of them addresses the case where
the outside information is itself the scarce thing - which is exactly the rare-class case.

---

## 1. Inferring label distributions from gradients - and why ours failed

This is the family our gate G0a belongs to, and the literature's successes come with a condition we
do not meet.

- **LLG, "User-Level Label Leakage from Gradients" (PETS 2022)** recovers labels with **over 98%
  accuracy** - but *"with an auxiliary dataset similar to the training data"*, at an early stage of
  training, under FedSGD. Gradient compression of 80% or more defeats it.
- **"Whispers of Data" (2025)**, which targets label *proportions* specifically, states that earlier
  approaches *depend heavily on auxiliary data*, and that differential-privacy noise further degrades
  them.
- **Label inference in vertical FL (USENIX Security 2022)** reaches accuracy 1.00 - but by reading
  label signs from server-side gradients in an architecture that is not split learning. Its passive
  and active attacks, which are the realistic ones, reach about 0.75.
- **A 2026 re-evaluation** argues these accuracies are generally overstated in realistic settings.

**What this means for us.** Our G0a failure is not a failure of implementation. The published
successes all have a server-side auxiliary dataset; EARN was designed deliberately not to need one,
because for rare diseases the server having examples is precisely what cannot be assumed. Our result
- ten definitions, ceiling 0.391 against a 0.7 bar, with no auxiliary data - is *consistent* with
this literature rather than contradicted by it. If asked "but label inference attacks work", the
answer is: they work with auxiliary data, and the auxiliary data is the thing we do not have.

## 2. Give the server a small clean dataset - FLTrust

**FLTrust (NDSS 2021)** is the canonical answer. The server holds a small **root dataset** - *fewer
than 100 examples* - trains its own model on it, and scores each client update by how far its
direction departs from the server's. With 40-60% malicious clients it matches attack-free FedAvg.

This is the clearest statement that **trust must be bootstrapped from outside**. The paper says so
directly: without it, from the service provider's perspective every client could be malicious.

**Why it does not solve our case.** A root dataset that can validate a *rare*-class claim must
contain rare-class examples. For a disease held by one hospital out of six, the server having its
own examples is the assumption the whole problem denies. FLTrust's root set would validate the
common classes and be silent on exactly the classes under attack. Later work also criticises
reliance on a single root set, noting its quality determines the result and proposing iterative
expansion of a trusted-client set instead.

**This is testable with what we have**, and is the one experiment this review suggests: implement
the root-dataset bootstrap and sweep how many rare-class examples the server holds, from zero
upward. The prediction is that it works for common classes at any root size and fails for rare
classes until the server holds a meaningful number of rare examples - which would make
"external information is required, and for rare classes it is unavailable" a measured claim rather
than an argued one.

## 3. Make lying unprofitable instead of detectable - incentive mechanisms

A separate tradition does not try to detect a false claim at all. It designs the payment so that
honest reporting is each client's best strategy.

- Auction and contract theory are the standard tools; a mechanism is **incentive compatible** when
  truthful reporting is a dominant strategy. Vickrey-style payments make a client's reward depend
  only on *others'* reports, so misreporting its own gains nothing.
- **QuoTa** (multi-dimensional reverse auction) proves that over-reporting data volume gains nothing
  and under-reporting loses.
- **Deposit-and-forfeit** schemes (2025) require a stake that is lost if claims prove false.
- **ICLR 2024 / heterogeneous FL work** achieves only *approximate* incentive compatibility once
  update manipulation is possible, which is an honest limit worth noting.

**What this means for us.** This is the tradition our attestation proposal belongs to, and it should
be positioned as such rather than as a new idea. A signed, immutable, one-shot declaration is a
commitment device; adding a penalty for a declaration contradicted later is the deposit mechanism.
The contribution is not the mechanism type but its application: nobody has applied it to the case
where the claim is *about which diseases a hospital sees* and cannot be checked by any other means.

## 4. Prove it cryptographically - ZKPs and trusted hardware

- A **2025 SoK on verifiable FL** frames training as integrity claims by clients and aggregator,
  including *data provenance*, with ZKPs or **TEE attestation quotes** as the artifacts.
- **ZK-FL** has clients prove correct computation with zk-SNARKs; the field is described as
  early-stage, with moderate-to-high overhead.
- **TEEs** (e.g. Olive, VLDB 2023) are cheap at runtime but require trusting the hardware vendor,
  and attestation proves *which binary ran*, not that a dataset has a claimed property.

**The relevant gap, stated by the survey literature itself:** there is no mature standard for
proving properties of a client's dataset - such as its class distribution - in zero knowledge.
Existing designs fall back on provenance claims, reference datasets, or TEE-backed execution.

**What this means for us.** The cryptographic route is the principled long-term answer and is out of
scope for this project, but it is worth one paragraph in the paper: our attestation proposal is a
governance-level stand-in for a cryptographic guarantee that does not yet exist, and we should say
so rather than imply it is equivalent.

---

## What this changes

**Nothing is retracted, and one defence is strengthened.** The G0a result now has literature behind
it: inference works with auxiliary data and we have none by design.

**The boundary result is positioned, not weakened.** Every family above imports outside information
- auxiliary data, a root dataset, an economic stake, or a hardware root of trust. Our contribution
is to show *where* that import becomes unavoidable, and that for rare classes the usual imports are
the scarce resource itself.

**Attestation is a known mechanism type, applied to a new case.** It should be cited into the
incentive-mechanism literature rather than presented as novel machinery. That is a smaller novelty
claim and a far more defensible one.

**One experiment is now worth more than the attestation test.** Implementing FLTrust's root-dataset
bootstrap and sweeping its rare-class content would turn the central claim from an argument into a
measurement. It uses machinery we already have, and it tests the field's standard solution against
the project's own hard case.

---

### Sources

- LLG, User-Level Label Leakage from Gradients, PETS 2022 - https://petsymposium.org/popets/2022/popets-2022-0043.pdf
- Whispers of Data: Unveiling Label Distributions in Federated Learning, 2025 - https://arxiv.org/pdf/2504.21436
- EC-LDA: Label Distribution Inference against Federated Graph Learning, 2025 - https://arxiv.org/pdf/2505.15140
- Label Inference Attacks Against Vertical Federated Learning, USENIX Security 2022 - https://www.usenix.org/system/files/sec22summer_fu.pdf
- Revisiting Label Inference Attacks in Vertical FL, 2026 - https://arxiv.org/html/2603.18680v1
- FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping, NDSS 2021 - https://arxiv.org/pdf/2012.13995
- Incentives in Federated Learning (book chapter) - https://web.mit.edu/jaillet/www/general/federated-learning-ch16.pdf
- Incentivizing Truthful Collaboration in Heterogeneous FL - https://arxiv.org/html/2412.00980
- QuoTa: Quality-Aware Incentive Mechanism - https://www.mdpi.com/2076-3417/14/2/833
- Long-Term Client Selection: A Truthful Auction Approach, 2025 - https://arxiv.org/pdf/2508.09181
- SoK: Verifiable Integrity Claims for Privacy-Preserving FL, 2025 - https://eprint.iacr.org/2025/2296.pdf
- Zero-Knowledge Federated Learning, 2025 - https://arxiv.org/pdf/2503.15550
- Olive: Oblivious Federated Learning on TEE, VLDB 2023 - https://www.vldb.org/pvldb/vol16/p2404-kato.pdf
