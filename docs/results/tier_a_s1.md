# Tier A baselines on S1 (natural split)

**G0b: FAIL** - FedAvg balanced accuracy 0.415 ± 0.005 (seeds 42, 43, 44), threshold 0.45.

Tier A: frozen DenseNet-121 features, linear head, 40 rounds, 50 local Adam steps per centre per round (batch 32, lr 0.0005), class-balanced loss: True. Final-round global head on the pooled test split; mean ± std over seeds.

| rule | balanced acc | rare macro-F1 | macro-F1 | accuracy | F1 Dermatofibroma | F1 Vascular lesion |
|---|---|---|---|---|---|---|
| fedavg | 0.415 ± 0.005 | 0.332 ± 0.021 | 0.385 ± 0.004 | 0.591 ± 0.006 | 0.192 | 0.471 |

Per-round data: `tier_a_s1_fedavg.csv`, `tier_a_s1_fedavg_weights.csv`.
