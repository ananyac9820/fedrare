# Live demo

Runs the model this project actually trained, on your own machine: a frozen ImageNet
DenseNet-121 as the feature extractor, and the Tier A classifier head trained by FedAvg across
the six real Fed-ISIC2019 centres. Pick one of eight held-out test images (or upload one) and
the page shows the predicted class and the probability of all eight.

## Run it

Two terminals, both from the repository root.

```bash
python demo/server.py
```

```bash
npm --prefix web run dev
```

Then open <http://localhost:3000/demo> (the port the dev server prints).

The backend listens on `http://127.0.0.1:8000`. To use another port, start it with
`python demo/server.py --port 8123` and run the site with
`NEXT_PUBLIC_DEMO_API=http://127.0.0.1:8123 npm --prefix web run dev`.

If the backend is not running, the page says so and tells you the command - it does not crash
or go blank.

## Files

| file | what it does |
| --- | --- |
| `server.py` | Flask API: `GET /health`, `GET /samples`, `GET /samples/<file>`, `POST /predict` |
| `prepare_demo.py` | trains the head, saves `models/tier_a_s1_fedavg_head.pt`, cuts the eight sample images |
| `check_demo.py` | verifies preprocessing against training, then prints every sample's prediction |
| `samples/` | the eight images and their true labels |

Both the checkpoint and the sample images are committed, so a fresh clone can run the demo
without the 93 MB feature files. To rebuild them you need
`data/features/fed_isic2019_densenet121_{train,test}.npz` (from `scripts/05_extract_features.py`),
then:

```bash
python demo/prepare_demo.py
```

## Checking it before you present

```bash
python demo/check_demo.py --server http://127.0.0.1:8000
```

This does three things: it re-derives each sample's 1024-d feature from the image file and
compares it against the feature saved during training (they agree to ~1e-6, which is what
proves the demo preprocesses images exactly as training did); it prints each sample's true and
predicted label; and it checks the running server returns the same numbers as a local run.

## Honest framing

The model's balanced accuracy on the held-out test set is 0.417 and its rare-class macro-F1 is
0.332 - better than chance across eight classes, nowhere near clinical use. It gets 6 of the 8
sample images right. The page always shows the true label beside the prediction, including when
they disagree; `check_demo.py` prints the same.

An uploaded photo gets the Shades-of-Gray colour-constancy correction FLamby applies to raw ISIC
images; the dataset's own images already carry it, so they are not corrected twice. An ordinary
phone photo is still far from a dermoscopy image, and the model will answer confidently anyway.
This is a student research prototype, not a medical device.

## Data

The sample images come from the Fed-ISIC2019 test split via the Hugging Face mirror
`flwrlabs/fed-isic2019`, derived from FLamby's Fed-ISIC2019 (ISIC 2019 / HAM10000 / BCN20000),
licensed CC BY-NC 4.0. Research use only.
