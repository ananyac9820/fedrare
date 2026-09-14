# Class-Conditional Blockchain-Coordinated Federated Learning for Rare Disease Diagnosis

Federated learning across multiple hospitals for rare skin-disease diagnosis, where each
hospital receives a **separate trust score per disease class** rather than one overall score,
with those scores recorded on-chain for auditability.

**Dataset:** Fed-ISIC2019 (23,247 dermoscopy images, 6 real data centres, 8 classes)
**Rare classes:** Dermatofibroma and Vascular Lesion (each under 1% of samples)

---

## The problem

Existing federated learning systems assign each hospital **one** trust or contribution score
per training round. A hospital holding most of the rare-disease cases, but performing only
averagely on common conditions, receives a low overall score and gets less influence —
precisely when its contribution matters most.

## The approach

| Stage | What happens |
|---|---|
| 1. Local training | Each hospital trains DenseNet-121 on its own images, sends only model updates |
| 2. Per-class evaluation | Coordinator scores each update separately for all 8 classes → a reputation *vector*, not a scalar |
| 3. Weighted aggregation | Per-class scores weight each row of the classifier head; feature extractor uses standard FedAvg |
| 4. On-chain record | Reputation vector committed via smart contract each round |

## Experiments

Three configurations compared on **rare-class macro-F1** (not overall accuracy, which is
dominated by common classes):

1. `fedavg` — plain FedAvg, no reputation (baseline)
2. `scalar` — one reputation score per hospital (current standard practice)
3. `classcond` — per-class reputation (**this project's contribution**)

Each is additionally evaluated under injected malicious client updates.

---

## Setup

```bash
git clone <this-repo-url>
cd fedrare

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Getting the dataset

The dataset is **not** in this repo. Download it from the Hugging Face mirror:

```python
from datasets import load_dataset
ds = load_dataset("flwrlabs/fed-isic2019")
```

144 MB, no registration. This is the same FLamby-derived data - identical 23,247 images,
already resized to 224px and colour-constancy corrected - published as parquet.

Licence: CC BY-NC 4.0 (non-commercial; academic research qualifies). Cite the FLamby paper
and the three source datasets (HAM10000, BCN20000, MSK) in any publication.

| Centre | Train | Test | Total |
|---|---|---|---|
| 0 | 9,930 | 2,483 | 12,413 |
| 1 | 3,163 | 791 | 3,954 |
| 2 | 2,691 | 672 | 3,363 |
| 3 | 1,807 | 452 | 2,259 |
| 4 | 655 | 164 | 819 |
| 5 | 351 | 88 | 439 |

Note the 28x spread between the largest and smallest centre. Any aggregation weighted by
sample count alone would nearly silence centre 5 - which is precisely the failure mode this
project addresses, before rare classes even enter the picture.

<details>
<summary>Alternative: full FLamby install</summary>

Only needed to regenerate preprocessing from source. Requires ISIC 2019 and HAM10000 licence
acceptance, a ~9 GB download, and a local resize step.

```bash
pip install git+https://github.com/owkin/FLamby.git
```
Then follow https://owkin.github.io/FLamby/fed_isic.html and set `dataset.source: flamby`
plus `dataset.root` in `configs/default.yaml`.
</details>

### Verify the setup

```bash
python scripts/01_verify_setup.py     # checks imports, GPU, dataset path
python scripts/02_explore_data.py     # class distribution per centre → results/
```

`02_explore_data.py` is the important one. It produces the per-centre, per-class counts that
the entire project depends on — you need to see the real imbalance before building anything.

---

## Repository layout

```
configs/          YAML configuration
src/data/         Dataset loading, transforms, per-centre splits
src/models/       DenseNet-121 definition
src/federated/    FedAvg, scalar reputation, class-conditional reputation
src/attacks/      Poisoning attack simulation
src/utils/        Metrics (rare-class macro-F1), seeding
scripts/          Numbered entry points, run in order
docs/             Project plan
results/          Generated outputs (gitignored except .gitkeep)
```

## Status

- [x] Repository scaffold
- [ ] Dataset downloaded and verified
- [ ] Per-centre class distribution confirmed
- [ ] FedAvg baseline
- [ ] Scalar reputation baseline
- [ ] Class-conditional reputation
- [ ] Poisoning robustness experiments
- [ ] Blockchain logging layer

See `docs/PROJECT_PLAN.md` for the full week-by-week schedule.
