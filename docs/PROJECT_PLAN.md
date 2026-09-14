# Project Plan — 17 August to 30 September 2026

Six and a half weeks. The plan is ordered so that the **make-or-break question is answered
in Week 4**, not in the final week. If the contribution does not hold up, you find out with
three weeks left to adapt rather than three days.

---

## Shape of the schedule

Weeks 1 and 2 are deliberately light: they are dominated by access requests that depend on
external approval, so the work is to start those requests early and wait. The heavy
implementation lands in Weeks 3–4, and Week 4 carries the decisive experiment.

---

## Week 1 — 17–23 Aug: Access and environment

Two of these depend on external approval, so start them before anything else.

- [ ] **Day 1:** Start PhysioNet CITI training for eICU — select *"Massachusetts Institute of
      Technology Affiliates"* as your affiliation, otherwise CITI charges a fee. 1–2 week wait.
- [ ] **Day 1:** Request CARE-FL full text through institutional library access
- [ ] Create the GitHub repo, push this scaffold
- [ ] `pip install -r requirements.txt`
- [ ] `python scripts/01_verify_setup.py` — downloads the dataset (144 MB) and checks the
      centre counts against the published figures
- [ ] Confirm you have a working GPU (Colab / Kaggle is fine if no local GPU)

No ISIC registration is needed: the Hugging Face mirror `flwrlabs/fed-isic2019` carries the
same preprocessed data at 144 MB instead of the ~9 GB raw download plus a local resize step.

**Exit criterion:** `01_verify_setup.py` passes every check.

---

## Week 2 — 24–30 Aug: Data grounding

- [ ] Run `python scripts/02_explore_data.py`
- [ ] Record the **measured** rare class IDs in `configs/default.yaml` — do not hardcode
      from memory; verify the class ordering matches the loaded labels
- [ ] Identify which centre holds the largest share of each rare class. That centre is your
      "specialist hospital" and the one to watch in every later experiment.
- [ ] Read CARE-FL if access has come through, and adjust the novelty claim if needed
- [ ] Commit the distribution results to the repo

**Exit criterion:** you can state, with real numbers, which classes are rare and which
centre holds them.

---

## Week 3 — 31 Aug – 6 Sep: FedAvg baseline

- [ ] Implement `src/federated/fedavg.py` — local training loop, weight averaging
- [ ] Train for the full round count on all 6 centres
- [ ] Log per-round rare-class macro-F1, not just accuracy
- [ ] Save the baseline results — every later number is compared against this

**Expected finding:** accuracy looks respectable while rare-class F1 stays poor. That gap
*is* the motivation for the project, so capture it clearly.

**Exit criterion:** a reproducible FedAvg number for rare-class macro-F1.

---

## Week 4 — 7–13 Sep: The decisive week

Both reputation methods, back to back, so they can be compared immediately.

- [ ] Implement `src/federated/scalar_reputation.py` — one score per client per round
- [ ] Implement `src/federated/class_reputation.py` — a score *vector* per client;
      per-class weights applied to the classifier head rows, feature extractor by plain FedAvg
- [ ] Run all three methods with 3 seeds each (rare-class variance is high — a single seed
      will mislead you)
- [ ] Compare rare-class macro-F1 across the three

> **Go / no-go checkpoint.** If class-conditional does not beat scalar on the rare classes,
> stop and diagnose before building anything further. Possible causes: warmup too short,
> reputation momentum too high, or the validation split holding too few rare samples to
> score against. Fixing this in Week 4 is recoverable; discovering it in Week 7 is not.

**Exit criterion:** a three-way comparison table with error bars.

---

## Week 5 — 14–20 Sep: Robustness

- [ ] Implement `src/attacks/poisoning.py` — label flipping, gaussian noise, sign flipping
- [ ] Run with 20% malicious clients across all three methods
- [ ] Verify the key claim: **malicious clients are down-weighted while the rare-class
      specialist is not.** This distinction is the core of the robustness argument.
- [ ] Produce the reputation-evolution plot (per-class scores over rounds, per hospital) —
      this single figure communicates the contribution better than any table

**Exit criterion:** evidence the mechanism separates malicious behaviour from legitimate
rare-class specialisation.

---

## Week 6 — 21–27 Sep: Blockchain layer

- [ ] Write the smart contract that stores per-round reputation vectors
- [ ] Deploy locally on Ganache or Hardhat (a local chain is entirely legitimate — say so
      plainly in the write-up rather than implying a public deployment)
- [ ] Integrate: after each aggregation round, commit the reputation vector on-chain
- [ ] Record gas cost and latency overhead — reviewers ask for this
- [ ] If time runs short, simulate the ledger in software and state that honestly. A working
      result with a simulated ledger beats a broken result with a real one.

**Exit criterion:** reputation vectors verifiably retrievable from the chain.

---

## Week 7 — 28–30 Sep: Consolidation

- [ ] Regenerate every figure from final results
- [ ] Write the results and discussion sections
- [ ] Finalise related work — cite CARE-FL properly and state the distinction
- [ ] Clean the repo, complete the README, tag a release

---

## Cut lines, decided in advance

If time runs short, drop in this order. Deciding now prevents panic decisions in Week 6.

1. **eICU / tabular track** — already a stretch goal, drop first
2. **Real blockchain deployment** — simulate instead, disclose it
3. **Differential privacy layer** — mention as future work
4. **Dashboard** — a static matplotlib figure conveys the same finding

**Never cut:** the three-way comparison (FedAvg vs scalar vs class-conditional) on
rare-class macro-F1. That comparison *is* the paper. Everything else is supporting material.

---

## What "done" means

- Three methods compared on rare-class macro-F1, 3 seeds, with error bars
- Robustness result under 20% malicious clients
- Reputation vectors recorded and retrievable (real chain or disclosed simulation)
- Reproducible repo: clone, install, run, obtain the same numbers
- Related work honestly positioned against CARE-FL
