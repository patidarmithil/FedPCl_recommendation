## 🔍 Research & Reference Repositories

To implement **FedPCL**, I am researching various GitHub repositories to integrate their functional logic, particularly regarding GNN backbones, personalized aggregation, and federated recommendation baselines.

### 1. Model Backbone (LightGCN)
* **[LightGCN (Official Implementation)](https://github.com/kuandeng/LightGCN/tree/master)**
    * [cite_start]**Description:** This is the core backbone for our system[cite: 420, 421]. [cite_start]It provides the simplified Graph Convolution Network (GCN) logic that performs linear propagation on user-item interaction graphs to learn embeddings without non-linear activations[cite: 91, 92].

### 2. Federated Frameworks & Client-Server Logic
* **[FedPDA (Official)](https://github.com/tian0920/FedPDA/tree/main)** ⭐ **(Important Reference)**
    * **Description:** This is a crucial resource for our framework setup. [cite_start]It provides a comprehensive implementation of general federated client-server communication, which is vital for building our server-side coordination and client-side local training loops[cite: 194, 195].
* **[PFGA (Personalized FedGraph Augmentation)](https://github.com/longtao-09/PFGA/tree/main)**
    * **Description:** Focuses on personalized federated learning with graph data. It provides insights into how local graphs can be augmented or processed to handle data heterogeneity.
* **[CreamFL](https://github.com/FLAIR-THU/CreamFL/tree/main)**
    * **Description:** A framework for heterogeneous Federated Learning. [cite_start]It offers strategies for managing diverse client models, which is helpful for our **Multicenter Personalized Aggregation** strategy[cite: 270, 271].

### 3. Personalized & Contrastive Recommendation Baselines
* **[FedDCA](https://github.com/ZifanJun/FedDCA)**
    * **Description:** Implements federated dual contrastive learning for recommendation. [cite_start]This is highly relevant for our implementation of structural contrastive learning to pull similar nodes closer in the embedding space[cite: 66, 217].
* **[PerFedRec Official](https://github.com/sichunluo/PerFedRec/tree/main)** & **[Study Repo](https://github.com/KENTAROSZK/PerFedRec-_study/tree/main)**
    * **Description:** These implement **PerFedRec**, a key baseline that uses GNNs for personalized recommendations. [cite_start]It serves as a benchmark for comparing our multi-center clustering approach[cite: 417, 443].
* **[PFedRec (IJCAI '23)](https://github.com/Zhangcx19/IJCAI-23-PFedRec/tree/main)**
    * [cite_start]**Description:** Implements a personalized federated framework with dual personalization for users and items[cite: 55, 725].
* **[Federated Recommendation](https://github.com/GuanyunFeng/federated_recommendation/tree/master)**
    * [cite_start]**Description:** A general repository for implementing federated baselines like **FedAvg** and **FedMF**[cite: 403, 405].

---

## 🛠️ Implementation Mapping

| FedPCL Component | Reference Repository | Logic to Extract |
| :--- | :--- | :--- |
| **GNN Backbone** | `kuandeng/LightGCN` | [cite_start]Linear propagation and layer aggregation[cite: 179]. |
| **Client-Server Setup** | `tian0920/FedPDA` | General federated training loops and communication logic. |
| **Structural Contrastive** | `ZifanJun/FedDCA` | [cite_start]Contrastive loss and negative sampling strategies[cite: 247]. |
| **Multicenter Aggregation** | `FLAIR-THU/CreamFL` | [cite_start]Server-side user grouping and cluster-level model updates[cite: 274]. |
| **Privacy (LDP)** | `GuanyunFeng/federated_rec` | [cite_start]Gradient clipping and Laplacian noise injection[cite: 289]. |
