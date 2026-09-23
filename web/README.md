# EARN - web prototype

A multi-page visual demo of the EARN project: federated learning across the six real
Fed-ISIC2019 hospitals, the rare-disease weighting gap, and the proposed coverage-aware,
ledger-anchored trust mechanism.

It is deliberately honest about project status. Every figure carries a label:

| Label | Meaning |
|---|---|
| **VERIFIED** (green) | Real and measured, computed from the research repo's `results/` files |
| **BASELINE** (blue) | A real working method we compare against (FedAvg) - not our novelty |
| **IN PROGRESS** (amber) | Being built or validated - EARN, the G0b retry |
| **FAILED** (red) | Measured, and did not meet its pre-set bar (gates G0a, G0b) |
| **NOT RUN YET** (grey, dashed) | No result exists - no number is shown anywhere |
| **SAMPLE DATA** (striped) | Placeholder values for layout only - never results |

## Run it

```bash
cd web
npm install
npm run dev          # http://localhost:3000
```

`npm run build && npm start` for a production build. No network is needed to build: fonts are
self-hosted from `@fontsource-variable/*`.

## Pages

| Route | What it shows |
|---|---|
| `/` | Hero with the 3D hospital network, the 1.87x headline, page cards, gate summary |
| `/problem` | Weight-vs-knowledge gap, per-hospital breakdown, rare classes, holder counts |
| `/method` | EARN's six-step round and the coverage blend (in progress) |
| `/results` | FedAvg baseline with seed spread and learning curve, G0a failure, what hasn't run |
| `/ledger` | Tamper-proof ledger concept with the interactive 3D chain (sample contents) |
| `/status` | Gates, six-week plan, run vs not run |

## Where the numbers come from

The UI reads only `web/data/*.json`, via `src/lib/data.ts`. Every file has `_meta.status`
(`verified` or `sample`) and `_meta.sources`. Derived figures - the 1.87x gap, per-centre
shares, holder counts, seed spreads - are computed in `src/lib/data.ts` from those files,
never hard-coded in components.

```bash
npm run sync-data    # regenerate web/data/ from ../results/
```

`scripts/sync-results.mjs` upgrades a file from sample to verified only when its source
appears in `results/`; anything missing is left as it is. `results/` is gitignored in the
research repo, so the generated JSON is committed here and the site works from a fresh clone.

| Data file | Source in `results/` | Status now |
|---|---|---|
| `dataset.json` | `class_distribution.csv`, `rare_classes.yaml`, `gate_g0a.json` | verified |
| `g0a.json` | `gate_g0a.json` | verified |
| `g0b.json` | `gate_g0b.json` | verified |
| `baseline_results.json` | `tier_a_s1_<rule>.csv` from `scripts/07_tier_a_s1_baselines.py` - per-seed finals and the per-round curve | verified (FedAvg only) |
| `roadmap.json` | gate states from `gate_g0*.json`; plan, notes and progress hand-maintained | verified |
| `ledger.json` | `ledger_rounds.json` | **sample** |
| `missing.json` | hand-maintained list of everything not run yet, with owner and branch | - |

To plug in real results: produce the file in `results/`, run `npm run sync-data`, commit the
changed JSON. When Camp A runs, `tier_a_s1_camp_a.csv` is picked up automatically and appears
next to FedAvg. Update `missing.json` and the progress notes in `roadmap.json` by hand.

## Layout

```
data/                  JSON the UI reads (generated + hand-maintained)
scripts/               sync-results.mjs
src/app/               layout (fonts, header, footer) and one folder per page
src/lib/data.ts        typed data access + derived numbers
src/components/
  Chrome.tsx           shared header (desktop links, mobile menu) and footer
  charts/              Recharts charts, light theme
  home/                hero (3D + parallax) and ticker
  ledger/              interactive ledger explorer
  three/               HospitalNetwork, LedgerChain - react-three-fiber scenes
  ui/                  status tags, pills, cards, motion, horizontal card rows
```

## Design choices

- **Palette:** warm neutrals - cream `#faf7f0`, paper `#fffdf9`, sand `#f2ebdd`, ink
  `#1d2929` - with one restrained accent, deep teal `#2c6a64`. Each status uses a deep text
  colour on a soft tint so it stays readable on the light background.
- **Type:** Fraunces (display serif) for headings, Inter for body, JetBrains Mono for labels.
- **Layout:** large section gaps (`py-24 md:py-36`), cards with 32-40px padding, pill labels
  beside headings, horizontal snap-scrolling card rows with previous/next buttons, a faint
  graph-paper texture in page headers.
- **Motion:** quiet fade-and-rise reveals and slow two-layer parallax in page headers and the
  home hero. Under `prefers-reduced-motion`: no transforms, no parallax, no count-ups, no
  ticker; both 3D scenes render a single static frame and the ledger shows a pre-built chain.
  3D render loops pause off screen, and each canvas has a no-WebGL text fallback.
