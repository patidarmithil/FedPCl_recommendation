## 🔍 Research & Reference Repositories

To implement **FedPCL**, I am researching various GitHub repositories to integrate their functional logic, particularly regarding GNN backbones, personalized aggregation, and federated recommendation baselines.

### 1. Model Backbone (LightGCN)
* **[LightGCN (Official Implementation)](https://github.com/kuandeng/LightGCN/tree/master)**
    * [cite_start]**Description:** This is the core backbone for our system[cite: 421]. [cite_start]It provides the simplified Graph Convolution Network (GCN) logic that performs linear propagation on user-item interaction graphs to learn embeddings without non-linear activations[cite: 91, 92].

### 2. Personalized Federated Recommendation Baselines
* **[PerFedRec Study](https://github.com/KENTAROSZK/PerFedRec-_study/tree/main)** & **[PerFedRec Official](https://github.com/sichunluo/PerFedRec/tree/main)**
    * [cite_start]**Description:** These repositories implement **PerFedRec**, a key baseline that uses GNNs for personalized recommendations in a federated setting[cite: 417]. [cite_start]It focuses on joint representation learning and model adaptation[cite: 713].
* **[PFedRec (IJCAI '23)](https://github.com/Zhangcx19/IJCAI-23-PFedRec/tree/main)**
    * [cite_start]**Description:** Implements a personalized federated framework with a dual personalization mechanism[cite: 55]. [cite_start]It is used to compare fine-grained personalization for both users and items[cite: 725].

### 3. Federated Learning & Frameworks
* **[Federated Recommendation](https://github.com/GuanyunFeng/federated_recommendation/tree/master)**
    * **Description:** A general repository for federated recommendation algorithms. [cite_start]It serves as a reference for implementing the **FedAvg** and **FedMF** baselines mentioned in the research[cite: 403, 405].
* **[CreamFL](https://github.com/FLAIR-THU/CreamFL/tree/main)**
    * **Description:** A framework for heterogeneous Federated Learning. [cite_start]It provides insights into cross-client model training and aggregation strategies that are useful for developing our **Multicenter Personalized Aggregation** module[cite: 270, 271].

---

## 🛠️ Implementation Mapping

| FedPCL Component | Reference Repository | Logic to Extract |
| :--- | :--- | :--- |
| **GNN Backbone** | `kuandeng/LightGCN` | [cite_start]Linear propagation and layer aggregation[cite: 179]. |
| **Personalization** | `sichunluo/PerFedRec` | [cite_start]Client-side local model adaptation[cite: 123]. |
| **Clustering** | `FLAIR-THU/CreamFL` | [cite_start]Server-side user grouping and aggregation[cite: 274, 298]. |
| **Base FL Logic** | `GuanyunFeng/federated_rec` | [cite_start]Gradient communication and FedAvg implementation[cite: 118]. |
