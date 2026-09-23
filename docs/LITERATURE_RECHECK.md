# Final literature re-check (23 Sep 2026)

The design doc requires, before submission, (a) reading CELM, CARE-FL, BOBA and ACE in full and
noting any overlap with our claims, and (b) one more focused search ("coverage-aware verification",
"class-level contribution attack", "low-coverage classes federated"). This is that check. Where a
full text could not be accessed it says so; nothing below is inferred from a title alone.

## Summary

No paper found measures per-class coverage **and changes how a client's claim is verified**
because of it, and none evaluates class-level attacks on contribution-aware aggregation on a real
hospital split. Two papers come close to parts of our framing and must be cited with care:
**CELM** (uses the word "class coverage") and **CALM** (per-class peer-agreement gating). Neither
has an attack model, trust accumulation over rounds, or a ledger. **Wording change required in the
paper (done):** we must not say "no prior work considers class coverage" — CELM does, as a
normalisation term; our claim is narrower (coverage-dependent *verification* and its failure modes
under attack).

## Paper by paper

| Paper | What it does (from the full text unless noted) | Overlap with our claims | Consequence |
|---|---|---|---|
| **CELM** (arXiv 2605.18892, 2026) | Server probes each client model with class-wise **logit maximisation** (Adam, 200 steps per class-client probe) to get per-class evidence; normalises it across clients ("per-class competence and class coverage"); averages to one score per client; **computes weights only during a warm-up (5% of rounds) and then freezes them**; explicitly **does not address Byzantine clients**; no trust or blockchain. Evaluates on FedISIC: balanced accuracy 70.18% vs FedAvg 61.25% (full models). | Uses "class coverage" as a normalising denominator. Its evidence comes from probes, **not** from update norms, so our G0a inversion finding does **not** transfer to CELM and must not be claimed against it. | Cite as the closest Camp A method; keep our design doc's critique (frozen weights, no attack model) - confirmed. Narrow our novelty wording on coverage (done in the paper draft). Note their FedAvg figure is for full-model training, not comparable to our Tier A 0.536. |
| **CARE-FL / DFed-MCDA** (Neurocomputing 2026) | **Abstract only - full text paywalled (HTTP 403).** Quality-Samples-Rarity aggregation: a *validation-derived* quality score, a *per-class sample* term and an inverse-frequency rarity factor; decentralised FL; gains in worst-class F1 on medical imaging. | Its per-class sample term is the same family as our "Camp A (reported counts)" baseline; it needs validation data, which EARN was designed to avoid. The abstract mentions no attack model. | Cite as Camp A; state that our reported-count baseline stands in for this family and that a client misreporting counts is exactly our A1 finding (7.4x weight on S2). Re-check the full text through the library before submission. |
| **BOBA** (AISTATS 2024, arXiv 2208.12932) | Two stages: (1) robustly fit a **(c-1)-dimensional affine subspace** of honest gradients by trimmed truncated SVD over the n-f nearest gradients; (2) use a **small clean labelled dataset on the server** (one virtual client per class) to estimate each client's label distribution and drop clients outside the simplex. | Handles label skew generally, not low coverage. **Not applicable as specified to our setting:** with n = 6 clients, f = 1 and c = 8 classes it needs a 7-dimensional subspace from 5 gradients (rank <= 4 after centring); and it needs server-side clean samples of every class, i.e. the rare-disease data the whole problem is about. | Our cut of BOBA (design doc cut line 2) is justified on substance, not only time. Stated in the paper's limitations and related work. |
| **ACE** (USENIX Security 2024, arXiv 2405.20975) | A malicious client manipulates its update so that **client-level** contribution evaluation (five methods, cosine-based analysis) scores it highly; six countermeasures found inadequate. | Confirms the design doc: ACE is client-level reward manipulation; ours is class-level influence capture where coverage is low. | Cite as the closest attack; claim stands. |
| **CALM** (arXiv 2609.05884, Sep 2026) - *found by the focused search* | Decentralised federated distillation; per-class teacher gate a = exp(-deviation from per-class peer consensus / tau). **No trust accumulation over rounds, no adaptation to how many peers hold a class, no Byzantine clients, no ledger.** | Per-class peer agreement is related to EARN's peer term. | Cite in related work; it strengthens rather than weakens our coverage point (peer consensus is exactly what fails at coverage 1). |

## Focused search

Queries: "coverage-aware verification", "class-level contribution attack", "low-coverage classes
federated", "per-class trust blockchain rare class peer verification history". Other hits were
checked by abstract and are client-level trust or poisoning defences (e.g. FoggyTrust 2606.27622,
FIDELIS 2508.10042, Honest-Score client selection 2311.05826, verifiable robust FL 2609.15521):
none is per-class and coverage-dependent. The search is not exhaustive; "we did not find it" is
still not proof that it does not exist.

## Sources

- CELM: https://arxiv.org/abs/2605.18892 (full text: https://arxiv.org/html/2605.18892)
- CARE-FL: https://www.sciencedirect.com/science/article/abs/pii/S0925231226019375 (abstract via search; full text not accessed)
- BOBA: https://arxiv.org/abs/2208.12932 (full PDF read)
- ACE: https://arxiv.org/abs/2405.20975 (abstract)
- CALM: https://pith.science/paper/2609.05884
