# The live system

An input/output console on top of the research pipeline: hospitals submit labelled cases and
train together, clinics send an image and get a diagnosis, an administrator sees all of it.

```bash
python server/app.py            # the API, on http://127.0.0.1:8100
npm --prefix web run dev -- --port 3007
```

Then open **http://localhost:3007/console**.

## Accounts

Seeded on first run, all with the password `fedrare`:

| username | role | can |
| --- | --- | --- |
| `hospital1` … `hospital6` | hospital | submit labelled cases, run a round, see their own trust and influence |
| `clinic` | clinic | diagnose an image on either model, see their own history |
| `admin` | admin | everything, plus the trust matrix, round history and a sandbox reset |

These are demonstration credentials with known passwords, and the API has no TLS or rate
limiting. It is built to run on one machine, not to be exposed.

## What is kept apart from the research

The published numbers are read-only here. The console reads
`models/tier_a_s1_fedavg_head.pt` once to seed its own model and never writes it; everything
the console produces - its model, its rounds, its trust, the uploads - lives in
`server/sandbox/`, which is gitignored. Nothing in `results/` or `web/data/` is touched, so a
sandbox round cannot move a number that appears in the paper.

## How a round works

1. A hospital uploads labelled images. They are turned into 1,024-d features by the same
   frozen DenseNet-121 the study used, and only the features take part in training.
2. The class counts in that upload are the hospital's **declaration**.
3. On a round, every hospital with data trains the shared head locally, and the updates are
   merged by EARN (or FedAvg, for comparison).
4. EARN scores each hospital's agreement with its peers and with its own locked record, and
   trust rises slowly, falls fast, and buys influence on the diseases it holds.
5. The result is scored on the held-out test split and written to the console database.

A hospital declaring its counts and being audited against them over rounds is the attested
evidence design in `docs/DESIGN_NOTE_attestation.md`, running for real.

## Things worth knowing before a demonstration

- **Trust starts at zero and needs 20 cases of a class.** That is the holder threshold the
  study pre-registered, so a hospital that has uploaded five cases will show zero trust. This
  is the mechanism working, not a fault, and the console says so on screen.
- **Trust rises about 0.1 a round**, so a visible trust story needs several rounds.
- **The model learns what you feed it.** Uploading images whose labels do not match their
  content will degrade the sandbox model, exactly as it should.
- **Resetting the sandbox** (admin → Overview → Danger zone) clears the model and the round
  history and leaves the uploads and the published study alone.
