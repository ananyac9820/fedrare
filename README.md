# EARN - Earned trust for rare diseases

Federated learning across the six real hospitals of **Fed-ISIC2019** (23,247 dermoscopy images,
8 skin-disease classes), asking one question: *who should get a say on the rarest diseases, and
what happens when a hospital lies about them?*

Live site: **https://fedrare.vercel.app** · Design: EARN Project Design v2 (16 Sep 2026) ·
Every change to that plan: [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)

## Where the project stands (23 Sep 2026)

Every gate was fixed before its result existed; each allowed one retry.

| Gate | Pass condition | Result | Verdict |
|---|---|---|---|
| **G0a** evidence signal | evidence ranks hospitals' true class counts, Spearman >= 0.7 | -0.210; the one retry (bias row) -0.463 | **Failed** |
| **G0b** baseline quality | Tier A FedAvg balanced accuracy >= 0.45 | 0.415; retry (last dense block fine-tuned) **0.535** | **Passed on retry** |
| **G1** problem is real | under Camp A, attack A1: >= 2x weight AND rare F1 drop >= 0.15 AND balanced-acc drop < 0.03 | S1: 5x weight but F1 drop 0.09 · S2: F1 drop 0.24 but balanced acc fell 0.051 | **Failed** |
| **G2** EARN works | see design doc 6.6 | not evaluable (G0a failed); exploratory oracle version not met | - |

So, as the design doc prescribes, the paper is **Fallback F1**: an empirical study of how
rare-class-aware aggregation behaves on a real hospital split, with attacks. EARN is fully
built and was run on an **oracle** evidence signal, clearly labelled exploratory.

## Main findings (408 runs, 3 seeds, all pre-registered)

1. **The weight gap is real.** Centre 2 holds 27.0% of the rare-disease training images but gets
   14.5% of FedAvg's weight (1.87x).
2. **Reading "evidence" from model updates backfires.** A hospital that never sees a disease moves
   that disease's row the *most* (G0a: -0.21). Weighting by that signal (Camp A) gives the
   specialist *less* say than FedAvg (12.6% vs 14.5%) and lowers rare F1 (0.558 vs 0.592).
3. **Honest counts help; claimed counts are exploitable.** Weighting by reported class counts is the
   best method without attack (rare F1 0.605 on S1), but an attacker claiming to be a big holder
   captures ~7.4x its fair weight on the specialist split S2 and rare F1 falls to 0.33.
4. **The sleeper is the worst attack.** A real holder that turns after 15 honest rounds cuts FedAvg's
   rare F1 from 0.592 to 0.286 on S1; median-norm clipping recovers most of it (0.538).
5. **Robust filters can erase a lone specialist.** On S2, Multi-Krum gives centre 2 0% of the rare
   rows and rare F1 is 0.000.
6. **EARN on an oracle signal (exploratory)** does not meet the G2 conditions: on S1 honest
   hospitals' rare-row updates agree too weakly for the peer check, so nobody earns trust; on S2 the
   history check rewards *consistency*, so the A1 attacker earns full trust.
7. **The ledger works as a constraint.** 2,400 real EARN rounds replayed on the `EarnLedger` contract
   (local Hardhat): ~184k gas and ~1.1 ms per round; a forged trust boost is rejected in every run.
   The "no ledger" ablation changed nothing measurable - the tested attackers never needed to
   rewrite history.

Full tables: [`docs/results/analysis.md`](docs/results/analysis.md) · figures:
[`docs/results/figures/`](docs/results/figures) · paper draft: [`docs/paper/draft.md`](docs/paper/draft.md) ·
status report: [`docs/STATUS_2026-09-23.md`](docs/STATUS_2026-09-23.md).

## Reproduce

```bash
git clone https://github.com/ananyac9820/fedrare && cd fedrare
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/02_explore_data.py            # per-hospital distribution, rare classes
python scripts/05_extract_features.py        # frozen DenseNet-121 features (~5 min Apple GPU, ~35 min CPU)
python scripts/06_gate_g0a.py                # G0a
python scripts/06b_gate_g0a_retry.py         # G0a retry (D1) + amendment M1 (D2)
python scripts/07_tier_a_s1_baselines.py --no-camp-a   # G0b, first attempt
python scripts/08_g0b_retry.py               # G0b retry: fine-tune last dense block (~13 min Apple GPU)
python scripts/09_run_grid.py --grid all     # the 408-run study (~10 min, 7 CPU workers)
(cd ledger && npm install && npx hardhat test && npx hardhat run scripts/measure.js)
python scripts/10_analyse.py                 # G1, framing, tables, figures, overhead
python -m pytest                             # 44 tests

cd web && npm install && npm run sync-data && npm run dev   # the site, http://localhost:3000
```

On macOS with the python.org build, set `SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")`
before the dataset download. Everything runs on a laptop: Tier A trains only the classifier head
on saved features (design doc Section 7).

## Repository layout

```
configs/default.yaml     dataset and training configuration
src/data/                loading, saved features (frozen / ft4), splits S1 and S2
src/federated/           aggregation interface, FedAvg, robust rules, Camp A, EARN, Tier A loop
src/attacks/attacks.py   A1 sudden expert, A2 sleeper, A3 scaling (client hooks)
src/ledger/chain.py      in-memory hash-chain ledger with the trust-step rule
src/experiments/grid.py  the pre-registered experiment grid
scripts/                 numbered entry points, run in order
ledger/                  EarnLedger.sol, Hardhat tests, on-chain replay of real EARN rounds
tests/                   44 Python unit tests (interface, rules, splits, EARN per step, ledger, attacks)
web/                     Next.js site; reads only web/data/*.json (npm run sync-data)
docs/                    DEVIATIONS.md, results_format.md, aggregation_interface.md, results/, paper/
```

`results/` and `data/` are gitignored; the key outputs are copied into `docs/results/` and the
site's `web/data/`.

## Honest limitations

- **EARN is not validated.** Its evidence signal failed G0a; its results use an oracle signal.
- **All results are Tier A** (a classifier head on DenseNet-121 features, last block fine-tuned
  once by FedAvg). Tier B full fine-tuning and the BOBA baseline were cut (design doc cut lines 1-2).
- **S2 is constructed** (90% of other hospitals' rare images moved to centre 2) and disclosed.
- **The rare-class rule was chosen after seeing the counts** (under 1/25 of the largest class).
- **Design doc correction:** it says common diseases have 5-6 holders; on the real counts basal
  cell carcinoma and actinic keratosis have 3, squamous cell carcinoma 2 (status report C6).

## Dataset

Hugging Face mirror `flwrlabs/fed-isic2019` (FLamby-derived, 144 MB, CC BY-NC 4.0). Cite FLamby
and the source datasets (HAM10000, BCN20000, MSK) in any publication.
