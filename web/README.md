# EARN - web prototype

A one-page visual demo of the EARN project: federated learning across the six real
Fed-ISIC2019 hospitals, the rare-disease weighting gap, and the proposed coverage-aware,
ledger-anchored trust mechanism.

It is deliberately honest about project status. Every number carries one of these labels:

| Label | Meaning |
|---|---|
| **VERIFIED** (emerald) | Real and measured, computed from the research repo's `results/` files |
| **BASELINE** (blue) | A real working method we compare against (FedAvg, Camp A) - not our novelty |
| **IN PROGRESS** (amber) | EARN, our proposed method: specified, not yet validated |
| **SAMPLE DATA** (striped, dashed) | Placeholder values for layout only - never results |

EARN is never shown as working. Its first internal check (gate G0a) failed, and the page
says so with the real measurement.

## Run it

```bash
cd web
npm install
npm run dev          # http://localhost:3000
```

`npm run build && npm start` for a production build. No network is needed to build: fonts are
self-hosted from `@fontsource-variable/*`.

## Where the numbers come from

The UI reads only `web/data/*.json`, via `src/lib/data.ts`. Every file has `_meta.status`
(`verified` or `sample`) and `_meta.sources`. Derived figures - the 1.87x gap, per-centre
shares, coverage - are computed in `src/lib/data.ts` from those files, never hard-coded.

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
| `roadmap.json` | gate states from `gate_g0a.json`, `gate_g0b.json`; plan hand-maintained | verified |
| `baseline_results.json` | `tier_a_s1_metrics.csv` (`scripts/07_tier_a_s1_baselines.py`) | **sample** |
| `ledger.json` | `ledger_rounds.json` | **sample** |
| `missing.json` | hand-maintained list of what is still sample and who owns it | - |

To plug in real results: produce the file in `results/`, run `npm run sync-data`, commit
the changed JSON. No component changes needed.

## Layout

```
data/                  JSON the UI reads (generated + hand-maintained)
scripts/               sync-results.mjs
src/app/               layout (fonts, metadata), page (composes sections)
src/lib/data.ts        typed data access + derived numbers
src/components/
  sections/            Hero, Problem, HowItWorks, Results, LedgerSection, Roadmap
  three/               HospitalNetwork (hero), LedgerChain - react-three-fiber scenes
  ui/                  Status tags, motion primitives, chart tooltip
```

## Design choices

- **Palette:** near-black ink (`#04060b`) with a teal accent (`#2dd4bf`) for data flow. Status
  colours are reserved for status tags only, so a colour always means the same thing.
- **Type:** Space Grotesk (display), Inter (body), JetBrains Mono (labels, numbers).
- **3D:** hero = six hospital nodes sized by real image counts around a wireframe global model,
  with particles for model updates in and out (images never travel). Ledger = rounded blocks
  linked by hash; new blocks commit every ~2.6 s; a "tamper" button shows every later link
  breaking.
- **Motion:** framer-motion scroll reveals (fade, slide, 3D tilt-in) and a two-layer parallax
  background per section; the hero's 3D layer scrolls slower than its text.
- **Reduced motion:** `MotionConfig reducedMotion="user"` drops transform animations; parallax,
  count-ups, the step highlight and chart animations are disabled; both 3D scenes render a
  single static frame (`frameloop="demand"`) and the ledger shows a pre-built chain. 3D render
  loops also pause when scrolled off screen, and each canvas has a no-WebGL text fallback.
