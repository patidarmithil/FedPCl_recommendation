# FedPCL: Personalized Federated Contrastive Learning for Recommendation

> Official project implementation and experiments for **FedPCL**  
> Paper: *Personalized Federated Contrastive Learning for Recommendation*  
> IEEE link: [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10839577](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10839577)

---

## 📌 Overview

Modern recommendation systems face two major issues in federated settings:

1. **Data Sparsity**  
   Each client has very limited local interactions, so local graph learning is weak.

2. **Data Heterogeneity (Non-IID)**  
   User preferences differ significantly across clients, making a single global model suboptimal.

**FedPCL** addresses both with a 3-part design:

- **Structural Contrastive Learning** (client-side) to improve representation quality in sparse local graphs.
- **Multi-center Personalized Aggregation** (server-side clustering) to better handle non-IID users.
- **Local Differential Privacy (LDP)** on uploaded updates to protect user information.

---

## ✨ Core Contributions Implemented

- ✅ **LightGCN-based federated recommendation**
- ✅ **Stage-wise pipeline** from baseline FedAvg to full FedPCL
- ✅ **2-hop neighborhood expansion** for richer local subgraphs
- ✅ **User + item structural contrastive objectives**
- ✅ **K-means based personalized cluster aggregation**
- ✅ **Client-side LDP** with clipping + Laplace noise
- ✅ **Automatic logging, metrics tracking, and embedding snapshot saving**

---

## 🧠 Method Summary

### 1) Structural Contrastive Learning

For each client, embeddings are trained with BPR plus contrastive losses:

- User-side contrastive objective (anchor vs structural neighbors)
- Item-side contrastive objective (two stochastic views)

This helps compensate for sparse local supervision.

### 2) Personalized Federated Aggregation

The server maintains:

- One **global** item embedding model
- **K cluster-specific** models

Each client receives personalized item embeddings as a weighted combination of cluster + global representations.

### 3) Local Differential Privacy (Stage 5)

Before upload, each client applies:

\[
\tilde{g} = \text{clip}(g,\sigma) + \text{Laplace}(0,\lambda)
\]

with privacy budget:

\[
\varepsilon = \sigma / \lambda
\]

---

## 🧱 Repository Structure

```text
.
├── FedPCL_stage5h/           # Most complete staged implementation (recommended)
│   ├── train_stage3.py
│   ├── train_stage4.py
│   ├── train_stage5.py
│   ├── federated_core_stage3.py
│   ├── federated_core_stage4.py
│   ├── federated_core_stage5.py
│   ├── client_stage3.py
│   ├── client_stage4.py
│   ├── client_stage5.py
│   ├── server_stage3.py
│   ├── server_stage4.py
│   ├── server_stage5.py
│   ├── contrastive.py
│   ├── ldp.py
│   ├── data_loader.py
│   ├── show_results.py
│   ├── run_multiseed.py
│   └── README.md
│
├── FedPCL/                   # Earlier/parallel implementation
├── FedPCL_stage5/            # Alternate stage implementation
├── Fedavg/                   # FedAvg baseline implementation
├── PerFedRec/                # Additional baseline/experiment
├── Datasets/                 # Raw datasets (ml-100k, ml-1m, steam, amazon, etc.)
└── Presentation(Latex-Code)  # Report/presentation assets
```

## 🚀 Quick Start (Recommended: FedPCL_stage5h)

### 1) Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy scipy scikit-learn matplotlib
```

*If you use GPU, install the correct PyTorch build for your CUDA version.*

### 2) Move into implementation folder
```bash
cd FedPCL_stage5h
```

### 3) Run Stage 5 (Full FedPCL + LDP)

**ML-100K**
```bash
python3 train_stage5.py \
  --dataset ml100k --data_path u.data \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.3 --max_neigh 20
```

**ML-1M**
```bash
python3 train_stage5.py \
  --dataset ml1m --data_path ratings.dat \
  --n_rounds 800 --clients_per_round 200 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.3 --max_neigh 20
```

**Steam**
```bash
python3 train_stage5.py \
  --dataset steam --data_path steam_processed.json \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.1 --max_neigh 10
```

**Amazon**
```bash
python3 train_stage5.py \
  --dataset amazon --data_path amazon_processed.json \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.1 --max_neigh 10
```

### 🧪 Ablations / Comparisons

**Stage 4 (No LDP)**
```bash
python3 train_stage4.py \
  --dataset ml100k --data_path u.data \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.3 --max_neigh 20
```

**Disable LDP in Stage 5**
```bash
python3 train_stage5.py \
  --dataset steam --data_path steam_processed.json \
  --no_ldp
```

**Multi-seed Reproducibility**
```bash
python3 run_multiseed.py \
  --stage 5 --dataset ml100k --data_path u.data \
  --seeds 42 123 456 789 1234 \
  --n_rounds 400 --local_epochs 10 \
  --tau 0.2 --beta1 0.1 --drop_rate 0.3 --max_neigh 20
```

## 📊 Evaluation Protocol

- Leave-one-out evaluation per user
- 100 sampled negatives per test user
- **Ranking Metrics**: HR@10, NDCG@10

## ⚙️ Key Hyperparameters (Typical)

| Parameter                | Value          |
|--------------------------|----------------|
| Embedding dim (d)        | 64             |
| GNN layers (K_gnn)       | 2              |
| Rounds (T)               | 400 (or 800 for ML-1M) |
| Clients / round          | 128            |
| Local epochs (E)         | 10             |
| Clusters (K)             | 5              |
| mu1, mu2                 | 0.5, 0.5       |
| Contrastive temperature (tau) | 0.2–0.3 |
| Contrastive weight (beta1) | 0.1          |
| Item CL weight (lam)     | 1.0            |
| LDP clip (sigma)         | 0.1            |
| Laplace scale (lambda_laplace) | 0.001    |

## 🗂️ Logs and Outputs

Typical generated artifacts:

- `stage5_log_<dataset>.json` — round-wise metrics and summary
- `emb_<dataset>_round0001.npy` — early embeddings
- `emb_<dataset>_round0400.npy` — final embeddings
- `*_meta.json` — metadata for embedding snapshots

**View Results**
```bash
python3 show_results.py --file stage5_log_ml100k.json
```

## 🛡️ Privacy Note

LDP is applied client-side before upload to server:

- Uploaded item deltas are privatized
- Uploaded user embeddings (for clustering) are privatized
- Raw user interaction data never leaves clients

## 📚 Citation

If you use this project, please cite the FedPCL paper:

```bibtex
@article{wang2025personalized,
  title   = {Personalized Federated Contrastive Learning for Recommendation},
  author  = {Wang, Shanfeng and Zhou, Yuxi and Fan, Xiaolong and Li, Jianzhao and Lei, Zexuan and Gong, Maoguo},
  journal = {IEEE Transactions on Computational Social Systems},
  year    = {2025}
}
```
