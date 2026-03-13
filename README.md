# FedPCL: Personalized Federated Contrastive Learning for Recommendation


[![Paper](https://img.shields.io/badge/Paper-IEEE_TCSS-blue)](https://ieeexplore.ieee.org/document/10834524)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[cite_start]Official implementation of **FedPCL**, a novel federated recommendation framework based on Graph Neural Networks (GNNs) that simultaneously addresses **data sparsity** and **personalization**[cite: 10, 67].



## 📌 Overview

[cite_start]Traditional centralized recommendation systems pose significant privacy concerns and face strict regulations like GDPR[cite: 6, 35, 36]. While Federated Learning (FL) offers a privacy-preserving alternative, it faces two major challenges in recommendation scenarios:
1.  [cite_start]**Data Sparsity:** Individual user data is often insufficient for models to learn reliable representations[cite: 8, 58, 60].
2.  [cite_start]**Data Heterogeneity (Non-IID):** Varying user preferences (non-IID data) make a single global aggregated model suboptimal[cite: 9, 52, 54].

[cite_start]**FedPCL** solves these by introducing a personalized federated GNN framework that extracts potential information by mining relationships between nodes[cite: 10, 66, 67, 193].

## 🚀 Key Features

* [cite_start]**Structural Neighbor Contrastive Learning:** Maximizes similarity between nodes and their high-order structural neighbors (k-hop neighbors) to enhance embedding quality and mitigate data sparsity[cite: 11, 13, 69, 76, 226].
* [cite_start]**Multicenter Personalized Aggregation:** Groups similar users into clusters using K-means on the server side[cite: 14, 70, 71, 298].
* **Personalized Dual-Model:** Each client combines a cluster-level model with a global model to obtain a personalized local model: 
    [cite_start]$$h_{u}^{t} = \mu_{1}h_{C(u)}^{t} + \mu_{2}h_{global}^{t}$$[cite: 15, 71, 280, 282].
* [cite_start]**Local Differential Privacy (LDP):** Encrypts gradients with Laplacian noise before uploading to the server to prevent privacy risks like rating inference[cite: 206, 285, 286, 289, 291].



## 📊 Experimental Results

[cite_start]FedPCL was evaluated on five real-world datasets and consistently outperformed competitive baselines like FedAvg, FedMF, and FedGNN[cite: 16, 78, 389, 441].

| Dataset | HR@10 (FedPCL) | NDCG@10 (FedPCL) | Improv. (HR) |
| :--- | :--- | :--- | :--- |
| **ML-1M** | 62.86% | 44.12% | 1.65% |
| **ML-100k** | 63.81% | 45.03% | 1.95% |
| **Steam** | 80.36% | 65.55% | 2.46% |
| **FilmTrust** | 16.81% | 8.61% | 7.34% |
| **Amazon-Elec** | 34.04% | 22.93% | 3.91% |

[cite_start]*Note: Improvements are relative to the second-best federated results[cite: 413, 415, 441, 442].*

## 🛠️ Implementation Details

* [cite_start]**Backbone Model:** LightGCN[cite: 420, 421].
* [cite_start]**Optimization:** Adam optimizer with BPR (Bayesian Personalized Ranking) loss[cite: 187, 422, 429].
* [cite_start]**Default Hyperparameters:** * Learning rate $\eta$: 0.1[cite: 430].
    * [cite_start]Temperature $\tau$: 0.3[cite: 430].
    * [cite_start]Weight $\mu_{1}, \mu_{2}$: 0.5[cite: 431].
    * [cite_start]Number of clusters $K$: 5[cite: 431].
    * [cite_start]Embedding dimensions: 64[cite: 431].

## ⚙️ Project Structure

```text
├── data/               # Preprocessing scripts for ML-1M, Steam, etc.
├── models/
│   ├── gnn.py          # GNN architecture (LightGCN)
│   ├── fed_pcl.py      # Core FedPCL logic (Contrastive & Personalized modules)
├── utils/
│   ├── ldp.py          # Local Differential Privacy implementation
│   ├── clustering.py   # Server-side K-means clustering
├── main.py             # Training entry point
└── requirements.txt
```

## 🚧 Current Progress

We are actively working on the implementation of this research paper. Our current focus includes:

* [cite_start]**Framework Setup:** Initializing the federated environment consisting of a trusted central server and $N$ distributed clients[cite: 194]. [cite_start]The server coordinates training and aggregates gradients, while each client maintains a private local subnetwork $\mathcal{G}_{i}$[cite: 195, 196].
* [cite_start]**GNN Integration:** Implementing **LightGCN** as the backbone model[cite: 91, 421]. [cite_start]This involves coding the linear propagation stage to aggregate $k$-hop neighbor information and the update function to form final node representations[cite: 92, 178, 179].

* **Module Development:**
    * [cite_start]**Structural Contrastive Learning:** Coding the contrastive objective (InfoNCE) to minimize the distance between a node's representation and its structural (homogeneous) neighbors found in even-numbered GNN layers[cite: 227, 228, 247].
    * [cite_start]**Multicenter Aggregation:** Developing the server-side logic to group similar users into $K$ clusters using K-means based on their embeddings to mitigate the impact of non-IID data[cite: 274, 298].

    * [cite_start]**Local Differential Privacy (LDP):** Implementing the privacy module that clips gradients and adds Laplacian noise to prevent the inference of user rating information[cite: 283, 285, 289, 291].
* [cite_start]**Dataset Preprocessing:** Converting interaction records from the **ML-1M, ML-100k, Steam, FilmTrust,** and **Amazon-Electronic** datasets into implicit feedback formats ($r_{uv}=1$ for interactions, $r_{uv}=0$ otherwise)[cite: 390, 399]. [cite_start]We are adopting the leave-one-out method for evaluation[cite: 400].

---

## 📝 Citation

If you use this code or method in your research, please cite:

```bibtex
@article{wang2025personalized,
  title={Personalized Federated Contrastive Learning for Recommendation},
  author={Wang, Shanfeng and Zhou, Yuxi and Fan, Xiaolong and Li, Jianzhao and Lei, Zexuan and Gong, Maoguo},
  journal={IEEE Transactions on Computational Social Systems},
  volume={12},
  number={5},
  pages={2986-2998},
  year={2025},
  publisher={IEEE}
}
