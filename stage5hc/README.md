# FedPCL — Personalized Federated Contrastive Learning for Recommendation

**B.Tech Project Implementation**  
Based on: *Wang et al., "Personalized Federated Contrastive Learning for Recommendation," IEEE TCSS Vol.12 No.5, Oct 2025*

---

## Overview

This repository implements **FedPCL**, a federated recommendation system that addresses two core challenges in federated learning for recommendation:

1. **Data sparsity** — each client holds only one user's interactions, making local graphs thin
2. **Data heterogeneity (non-IID)** — users have fundamentally different preferences, so a single global model performs poorly

FedPCL solves these by combining three components: structural contrastive learning on the client side, personalized cluster-level models on the server side, and local differential privacy for gradient protection.

---

## Paper Summary

### Problem
In federated recommendation, user data never leaves the device. Each client trains locally on its own user–item interaction graph and sends only encrypted gradients to the server. This creates two problems:
- Local graphs are tiny (one user, ~20–100 items), making GNN learning unreliable
- Different users have different tastes (non-IID), so FedAvg's single global model underserves everyone

### FedPCL's Solution

**On the client side — Structural Contrastive Learning (Stage 4):**  
LightGCN naturally aggregates homogeneous neighbours at even-numbered layers. In a bipartite user–item graph, a user's even-layer embedding captures information from structurally similar users (via shared items). FedPCL uses this as a self-supervised signal: the even-layer embedding of a node should be close to its layer-0 embedding (positive pair) and far from other users' layer-0 embeddings (negatives). This is applied to both user embeddings (Eq. 5) and item embeddings (Eq. 6), reducing the effect of sparse local data.

**On the server side — Personalised Clustering (Stage 3):**  
Instead of one global model, the server maintains K cluster models (K=5) plus one global model. Users are grouped by K-means on their uploaded embeddings. Each user receives a personalised model: `h_personal = μ₁ * h_cluster + μ₂ * h_global` (μ₁=μ₂=0.5). This handles non-IID data by giving each user a model tuned to their cluster's shared preferences.

**Privacy protection — Local Differential Privacy (Stage 5):**  
Before uploading gradients, each client applies: `g̃ = clip(g, σ) + Laplace(0, λ)`. Clipping bounds the sensitivity; Laplacian noise obscures the individual gradient. The privacy budget is ε = σ/λ (default σ=0.1, λ=0.001, ε=100).

### Key Results (Paper Table I)

| Dataset | HR@10 | NDCG@10 | vs FedAvg baseline |
|---|---|---|---|
| Steam | 80.36% | 65.55% | +9.15% |
| ML-100K | 63.81% | 45.03% | +21.11% |
| ML-1M | 62.86% | 44.12% | +18.16% |
| Amazon-Electronics | 34.04% | 22.93% | +7.51% |

---

## Implementation Stages

The implementation is built progressively across 5 stages, each adding one component:

| Stage | Description | Key file |
|---|---|---|
| 1 | Centralized LightGCN (baseline) | `train_central.py` |
| 2 | Basic FedAvg with local LightGCN | `train_fedavg.py` |
| 3 | FedAvg + K-means clustering + personalised embeddings | `train_stage3.py` |
| 4 | Stage 3 + structural contrastive learning | `train_stage4.py` |
| 5 | Stage 4 + Local Differential Privacy | `train_stage5.py` |

---

## File Structure

```
fedpcl/
│
│── Core training pipeline
│   ├── train_stage4.py              Stage 4 entry point (no LDP)
│   ├── train_stage5.py              Stage 5 entry point (with LDP)
│   ├── federated_core_stage4.py     Stage 4 training loop
│   ├── federated_core_stage5.py     Stage 5 training loop
│   ├── client_stage3.py             Stage 3 client (clustering-aware)
│   ├── client_stage4.py             Stage 4 client (+ contrastive loss)
│   ├── client_stage5.py             Stage 5 client (+ LDP on uploads)
│   ├── server_stage3.py             Server with K-means clustering
│   ├── server_stage4.py             Server with 2-hop neighbourhood
│   ├── server_stage5.py             Stage 5 server (= Stage 4)
│   ├── contrastive.py               Structural contrastive loss functions
│   └── data_loader.py               Dataset loading for all 4 datasets
│
│── Utilities
│   ├── show_results2.py             Read and display training logs
│   ├── visualize_kde.py             KDE embedding visualisation (paper Fig.6)
│   └── run_multiseed.py             Multi-seed reproducibility runner
│
│── Datasets
│   ├── u.data                       ML-100K ratings
│   ├── u.item                       ML-100K item names
│   ├── ratings.dat                  ML-1M ratings
│   ├── movies.dat                   ML-1M movie names
│   ├── steam_processed.json         Steam-200K (preprocessed)
│   └── amazon_processed.json        Amazon Electronics (preprocessed)
│
└── Results (generated during training)
    ├── stage5_log_{dataset}.json    Training log per dataset
    ├── emb_{dataset}_round0001.npy  Item embeddings at round 1 (for KDE)
    └── emb_{dataset}_round0400.npy  Item embeddings at round 400 (for KDE)
```

---

## Quick Start

### Stage 5 (Full FedPCL + LDP)

```bash
# ML-100K
python3 train_stage5.py \
  --dataset ml100k --data_path u.data \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.3 --max_neigh 20

# ML-1M  (needs 800 rounds to converge — large user base)
python3 train_stage5.py \
  --dataset ml1m --data_path ratings.dat \
  --n_rounds 800 --clients_per_round 200 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.3 --max_neigh 20

# Steam
python3 train_stage5.py \
  --dataset steam --data_path steam_processed.json \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.1 --max_neigh 10

# Amazon
python3 train_stage5.py \
  --dataset amazon --data_path amazon_processed.json \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.1 --max_neigh 10
```

### Stage 4 (No LDP — for ablation)

```bash
python3 train_stage4.py \
  --dataset ml100k --data_path u.data \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.3 --max_neigh 20
```

### Disable LDP (compare Stage 4 vs Stage 5)

```bash
python3 train_stage5.py --dataset steam --data_path steam_processed.json --no_ldp
```

### View results

```bash
python3 show_results2.py --file stage5_log_ml100k.json
python3 show_results2.py --file stage5_log_ml100k.json --plot   # saves training_curves.png
```

### Multi-seed reproducibility (5 runs)

```bash
python3 run_multiseed.py \
  --stage 5 --dataset ml100k --data_path u.data \
  --seeds 42 123 456 789 1234 \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.3 --max_neigh 20
```

### KDE embedding visualisation (paper Fig. 6)

```bash
# Embeddings at round 1 and 400 are saved automatically every run.
python3 visualize_kde.py \
  --files emb_ml100k_round0001.npy emb_ml100k_round0400.npy \
  --labels "Round 1 (random init)" "Round 400 (trained)" \
  --save kde_ml100k.png
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| d | 64 | Embedding dimension |
| K_gnn | 2 | GNN layers |
| T | 400 | Communication rounds |
| N | 128 | Clients per round |
| E | 10 | Local epochs |
| K | 5 | Number of clusters |
| μ₁, μ₂ | 0.5 | Cluster/global model weights |
| τ | 0.2 | Contrastive temperature |
| β₁ | 0.1 | Contrastive loss weight |
| λ | 1.0 | Item CL weight |
| σ | 0.1 | LDP clipping bound |
| λ_lap | 0.001 | LDP Laplacian noise scale |
| ε | 100 | Privacy budget (σ/λ_lap) |

---

## Evaluation Protocol

Leave-one-out split: for each user the most recent interaction is held as the test item. Evaluation ranks the test item against 100 randomly sampled negative items not in the user's history. Metrics: HR@10 (hit rate) and NDCG@10 (normalised discounted cumulative gain).

---

## Dependencies

```
python >= 3.8
torch
numpy
scikit-learn
scipy
matplotlib
```
