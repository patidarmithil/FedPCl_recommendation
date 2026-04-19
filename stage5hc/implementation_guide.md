Step-by-Step Implementation Guide for FedPCL
This guide will help you implement the FedPCL paper from scratch, reproducing its experimental results. We'll break down the process into logical stages, from setting up the baseline to integrating all federated components.

Prerequisites
Python 3.8+ with PyTorch (≥1.9)

Libraries: numpy, scikit-learn (for clustering), pandas, tqdm, etc.

Datasets: MovieLens-1M, MovieLens-100k, Steam, FilmTrust, Amazon-Electronics (download instructions in paper)

Optional: A federated learning framework like Flower can simplify communication, but we'll outline a manual implementation for clarity.

Stage 1: Implement Centralized LightGCN (Baseline)
Before adding federated complexity, get the core GNN recommender working in a centralized manner. This ensures your model architecture and loss functions are correct.

1.1 Data Preparation
Load each dataset as a user–item interaction matrix (implicit feedback: 1 if interacted, 0 otherwise).

Build a bipartite graph: adjacency list for users and items.

Split data: For each user, hold out some interactions for testing (e.g., leave-one-out or 80/20). The paper uses negative sampling evaluation.

Files:

data_loader.py: functions to load dataset, build graph, generate train/test splits.

1.2 LightGCN Model
Implement LightGCN following He et al. 2020. Key components:

Embedding layer: user_emb and item_emb of dimension d (64).

Propagation: For each layer l (0..K-1):

New user emb = aggregate of neighbor item embs from previous layer (normalized by degree).

New item emb = aggregate of neighbor user embs.

Final embedding: sum or mean of all layer embeddings.

Prediction: dot product of final user and item embeddings.

File: model.py – class LightGCN.

1.3 Training Loop (Centralized)
Loss: BPR (Bayesian Personalized Ranking) loss.

Optimizer: Adam with learning rate 0.001 (or as in paper: 0.1? Note: paper says learning rate 0.1 for federated, but for centralized LightGCN typical LR is lower. You may need to tune.)

Negative sampling: For each positive pair (u,i), sample a random negative item j.

Evaluation: HR@10, NDCG@10 using 100 random negative samples per test user.

File: train_central.py.

✅ Goal: Achieve numbers close to the "Center" column in Table I.

Stage 2: Basic Federated Learning with FedAvg
Now extend to federated setting. Start with simple FedAvg (no personalization, no contrastive) to verify communication and aggregation.

2.1 Client Simulation
Each client holds one user’s data (subgraph). The client's local graph includes the user node and all items they interacted with. (Optionally, you can expand the graph by including 2-hop neighbors as in FedGNN, but for simplicity start with direct interactions.)

Clients have:

Their own user embedding (trainable locally)

Global item embeddings (shared and aggregated)

A copy of the GNN model (parameters).

2.2 Server-Client Protocol
Define a communication round:

Server selects a fraction of clients (e.g., 128 per round).

Server sends current global item embeddings and model parameters to selected clients.

Each client:

Receives global parameters.

Performs several local epochs of training on its local graph using BPR loss.

Computes gradients for item embeddings and model parameters.

(Optional: clip and noise for privacy later.)

Sends gradients back to server.

Server aggregates gradients (FedAvg) and updates global item embeddings and model.

2.3 Implementation
Create classes: Client, Server.

Server maintains: item_emb, model_params.

Client maintains: user_id, local_graph, user_emb (local), copy of model.

Aggregation: weighted average by dataset size.

Files: federated_core.py, client.py, server.py.

✅ Test: Compare with FedAvg baseline in Table I (should be around 44.7 HR@10 on ML-1M). This verifies your federated setup.

Stage 3: Add Personalization via Clustering
Implement the server-side clustering and personalized model combination.

3.1 User Clustering (Server)
After each round (or every few rounds), collect user embeddings from all clients (or only selected ones).

Apply K-means (from sklearn) to assign each user to a cluster. Number of clusters K=5 (as in paper).

Maintain cluster assignments and update when clustering is performed.

3.2 Cluster Models
Server maintains a separate model for each cluster (initialized same as global model).

After each round, for each cluster, aggregate gradients only from clients belonging to that cluster → update cluster model.

Global model remains aggregated from all clients.

3.3 Personalized Model at Client
When server sends models to a client, it sends:

Global model h_global

Cluster model for that user's cluster h_cluster

Client combines them:

python
personalized_model_params = mu1 * cluster_model_params + mu2 * global_model_params
with mu1 = mu2 = 0.5 (paper).

The client uses this combined model for local training and inference.

Implementation details:

Models are just parameter dictionaries; combination is element-wise weighted sum.

Ensure both models have identical architecture.

✅ Test: Should improve over FedAvg, approaching PerFedRec/FPFR numbers.

Stage 4: Add Structural Contrastive Learning (Client Side)
Now enhance local training with contrastive loss using even-layer outputs.

4.1 Modify LightGCN to Output Layer-wise Embeddings
During forward pass, store outputs of each layer for users and items.

For contrastive loss, we need the even layer outputs (e.g., layer 2 if using 3 layers total). For a user, even layer outputs aggregate homogeneous neighbors (users from 2-hop).

4.2 Contrastive Loss Computation
For each user u in the local batch:

Positive pair: (e_u^{(l)}, e_u^{(0)}) – the even-layer output and the initial embedding.

Negative samples: initial embeddings of other users in the same client (or a random subset). The denominator sums over all users in the client’s user set (or a sampled set). Since each client has only one user, you need to consider items as well? Wait: each client corresponds to one user, so the "other users" in the same client does not exist. But the paper defines contrastive loss on both user and item sides. For a user client, the user set is just that one user. How can we have negative samples? The paper's formulation assumes each client has multiple users? That seems contradictory.

Key insight: In a real federated scenario, each client holds only one user. But the paper's Figure 1 shows each client having its own local subgraph, which includes the user and items. For contrastive loss on the user side, you need multiple users to form negative samples. However, the paper states: "For user u, we consider the homogeneous nodes on even layers ... and the homogeneous neighboring nodes of other users to be negative samples." This implies that within a client, you might have multiple users if you expand the local graph to include similar users (like FedGNN's graph expansion). But the paper doesn't explicitly say they expand the graph. Alternatively, they might be performing contrastive learning across clients? No, loss is computed locally.

Actually, re-reading Section III-D-1: "For user u, we consider the homogeneous nodes on even layers of the local GNN as positive samples, and the homogeneous neighboring nodes of other users to be negative samples." This suggests that within a local subgraph, there are indeed multiple users. How? In a bipartite graph, if you include the user and all items they interacted with, the only users are the central user. However, if you perform graph expansion as in FedGNN (adding 2-hop neighbors), you can bring in other users who share items. The paper references FedGNN for graph expansion in Section III-C: "We extend the local subnetwork using a similar approach to federated graph neural network (FedGNN) [38], which includes the user-item interaction data stored locally and the neighbors of each user." So indeed, each client's local graph is expanded to include other users (and their items) that are structurally close (e.g., users who interacted with the same items). That gives multiple users per client, making contrastive learning possible.

Thus, before implementing contrastive loss, you must implement graph expansion as in FedGNN.

4.3 Graph Expansion (FedGNN-style)
For a given user u, find all items they interacted with.

For each such item, find all other users who interacted with that item (these are homogeneous neighbors at distance 2).

Include those users and their interacted items in the local subgraph (subject to privacy constraints? FedGNN does this in a privacy-preserving way using pseudo-item sampling; but for implementation, you can simulate by having server share anonymized neighbor info or by using public item-user mappings. The paper may assume server can share such information safely.)

Now the local graph contains multiple users and items, enabling contrastive loss.

4.4 Implement Contrastive Loss
During local training, after forward pass, extract:

user_emb_0, user_emb_l for all users in local graph.

item_emb_0, item_emb_l for all items.

Compute losses per Equation (5) and (6). Use temperature τ (0.3 initially). Sum losses and add to total loss with weight β1.

Note: For computational efficiency, you might sample negatives instead of using all nodes.

File: contrastive.py – function structural_contrastive_loss.

✅ Test: With contrastive loss, performance should improve over plain federated GNN.

Stage 5: Add Local Differential Privacy
Finally, incorporate privacy protection.

5.1 Gradient Clipping
Before sending gradients, clip each gradient tensor to have max norm σ (a hyperparameter). Paper uses clip(g, σ) – likely per-coordinate clipping to range [-σ, σ].

5.2 Add Laplacian Noise
Generate Laplacian noise with scale λ for each gradient coordinate.

Add noise to clipped gradients.

Implementation:

python
def apply_ldp(grad, clip_sigma, lambda_laplace):
    grad_clipped = torch.clamp(grad, -clip_sigma, clip_sigma)
    noise = torch.distributions.Laplace(0, lambda_laplace).sample(grad.shape)
    return grad_clipped + noise
Apply to both item gradients and model gradients.

✅ Test: Ensure performance drop is acceptable while privacy is added.

Stage 6: Putting It All Together
6.1 Hyperparameter Tuning
Use values from paper: learning rate 0.1, L2 reg 1e-6, τ 0.3, λ 1.0, μ1=μ2=0.5, K=5 clusters.

Experiment with β1 (contrastive weight) – paper doesn't give exact value; you may need to tune.

Number of local epochs: typically 1 or a few.

Communication rounds: 400.

6.2 Evaluation
After training, evaluate each user's personalized model on their test items.

Report average HR@10 and NDCG@10 across all users.

Compare with baselines: you may need to implement FedAvg, FedMF, FedGNN, PerFedRec, FPFR for fair comparison (or use existing implementations).

6.3 Code Organization
Suggested structure:

text
fedpcl/
├── data/
│   ├── dataset.py          # dataset loading, splitting, graph construction
│   └── expanded_graph.py   # graph expansion (FedGNN style)
├── models/
│   ├── lightgcn.py         # LightGCN model with layer outputs
│   └── contrastive.py      # contrastive loss functions
├── federated/
│   ├── client.py           # Client class
│   ├── server.py           # Server class (clustering, aggregation)
│   └── utils.py            # FedAvg, LDP, etc.
├── train_central.py        # Centralized training script
├── train_federated.py      # Main federated training loop
├── config.py               # Hyperparameters
└── eval.py                 # Evaluation metrics
Stage 7: Reproducing Results
Run each baseline with same settings.

Run FedPCL with best hyperparameters.

Compare with Table I.

Potential challenges:

Graph expansion might be complex; you can start without it and only use the single user per client, but then contrastive loss on user side would be impossible. You might need to adapt: use items as the only nodes for contrastive? But the paper uses both. Perhaps you can simulate multiple users per client by grouping users with similar items in a preprocessing step. Another approach: In each client, treat the user's own embeddings from even layers as positive, and use embeddings of items from even layers as negative for users? That deviates from paper.

Carefully read FedGNN paper for graph expansion details.

