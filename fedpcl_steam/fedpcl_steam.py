import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
import gc
import os

# ==========================================
# [cite_start]Configuration & Hyperparameters [cite: 426-427]
# ==========================================
CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'emb_dim': 64,
    'lr': 0.1,  # Adjusted from 0.1 for stability in simulation
    'server_lr': 1.0,        # <-- ADD THIS LINE
    'decay': 1e-4,
    'tau': 0.3,              # Temperature for contrastive loss
    'beta1': 0.1,            # Weight for contrastive loss (implicit in paper analysis)
    'beta2': 1e-4,           # L2 Reg weight
    'lambda_val': 1.0,       # Weight between User/Item contrastive loss
    'mu1': 0.5,              # Personalized model weight
    'mu2': 0.5,              # Global model weight
    'n_clusters': 5,         # K clusters
    'n_clients_per_round': 128, 
    'rounds': 600,
    'local_epochs': 10,       # Local training steps per round
    'warmup_epochs': 20,     # No contrastive learning for first 20 rounds [cite: 430]
    'ldp_sigma': 0.1,        # LDP Noise scale
    'ldp_clip': 1.0,         # Gradient clipping threshold
    'gnn_layers': 2,         # Number of GNN layers (Paper implies 2 or 3)
    'batch_size': 1024       # Local batch size
}

print(f"Running on: {CONFIG['device']}")

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
def load_data(csv_path):
    print("Loading and Filtering Data (Type-Safe Version)...")
    try:
        # FIX: Force user_id to be read as string
        df = pd.read_csv(csv_path, names=['user_id', 'item_name', 'action', 'hours', 'extra'], dtype={'user_id': str})
    except:
        df = pd.read_csv(csv_path, dtype={'user_id': str})

    # 1. Keep 'purchase' and 'play'
    df = df[df['action'].isin(['purchase', 'play'])]
    
    # 2. Deduplicate (The fix for double counting)
    df = df.drop_duplicates(subset=['user_id', 'item_name'])
    
    # 3. Recursive 5-Core Filtering
    k = 5
    while True:
        start_len = len(df)
        user_counts = df.groupby('user_id')['item_name'].count()
        valid_users = user_counts[user_counts >= k].index
        df = df[df['user_id'].isin(valid_users)]
        
        item_counts = df.groupby('item_name')['user_id'].count()
        valid_items = item_counts[item_counts >= k].index
        df = df[df['item_name'].isin(valid_items)]
        
        if len(df) == start_len:
            break

    print(f"Final Training Data: {df['user_id'].nunique()} users, {len(df)} interactions.")
    
    user_ids = sorted(df['user_id'].unique())
    item_ids = sorted(df['item_name'].unique())
    
    user2id = {u: i for i, u in enumerate(user_ids)}
    item2id = {i: j for j, i in enumerate(item_ids)}
    
    df['u_idx'] = df['user_id'].map(user2id)
    df['i_idx'] = df['item_name'].map(item2id)
    
    num_users = len(user_ids)
    num_items = len(item_ids)
    
    user_interactions = df.groupby('u_idx')['i_idx'].apply(list).to_dict()
    
    return num_users, num_items, user_interactions, df
# ==========================================
# 2. GNN Model (LightGCN Backbone)
# ==========================================
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, n_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        # Embeddings
        self.embedding_user = nn.Embedding(num_users, emb_dim)
        self.embedding_item = nn.Embedding(num_items, emb_dim)
        
        # [cite_start]Initialize [cite: 171]
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
    def forward(self, adj_sparse):
        # LightGCN propagation: e^(k+1) = D^-0.5 A D^-0.5 e^(k)
        # We assume adj_sparse is the Normalized Adjacency Matrix
        
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        # [cite_start]Propagation [cite: 172-173]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_sparse, all_emb)
            embs.append(all_emb)
            
        # Stack and Mean (LightGCN standard aggregation)
        stack_embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(stack_embs, dim=1)
        
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        return users, items, embs

    def get_layer_embeddings(self, adj_sparse, layer_idx):
        # [cite_start]Helper to get specific layer output for Contrastive Learning [cite: 250]
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        curr_emb = all_emb
        for i in range(layer_idx):
            curr_emb = torch.sparse.mm(adj_sparse, curr_emb)
        
        users, items = torch.split(curr_emb, [self.num_users, self.num_items])
        return users, items

# ==========================================
# 3. Utilities: Graph Construction & Privacy
# ==========================================
def build_sparse_graph(num_users, num_items, interaction_dict, device):
    """
    Constructs the normalized adjacency matrix for LightGCN.
    A_hat = D^-0.5 A D^-0.5
    """
    # Create edges
    src_nodes = []
    dst_nodes = []
    
    for u, items in interaction_dict.items():
        for i in items:
            src_nodes.append(u)
            dst_nodes.append(i + num_users) # Offset item IDs
            
    # Add self-loops (optional in LightGCN but good for stability)
    # src_nodes.extend(list(range(num_users + num_items)))
    # dst_nodes.extend(list(range(num_users + num_items)))

    # Symmetric adjacency
    rows = src_nodes + dst_nodes
    cols = dst_nodes + src_nodes
    
    indices = torch.LongTensor([rows, cols])
    values = torch.ones(len(rows))
    
    # Sparse Tensor
    adj = torch.sparse_coo_tensor(indices, values, (num_users + num_items, num_users + num_items))
    
    # Normalization
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # A_hat = D^-0.5 * A * D^-0.5
    # Since torch sparse mm is limited, we use a simplified approach often used in LightGCN impls:
    # Scale values by d_inv_sqrt[row] * d_inv_sqrt[col]
    
    values_norm = values * d_inv_sqrt[rows] * d_inv_sqrt[cols]
    adj_norm = torch.sparse_coo_tensor(indices, values_norm, (num_users + num_items, num_users + num_items))
    
    return adj_norm.to(device)

def apply_ldp(gradients, clip_value, sigma):
    """
    [cite_start]Local Differential Privacy: Clip gradients and add Laplace noise [cite: 289]
    """
    noisy_grads = {}
    for name, grad in gradients.items():
        if grad is None: continue
        # Clip
        grad_norm = grad.norm(2)
        clip_coef = clip_value / (grad_norm + 1e-6)
        if clip_coef < 1:
            grad = grad * clip_coef
            
        # Add Laplace Noise
        noise = torch.from_numpy(np.random.laplace(0, sigma, grad.shape)).float().to(grad.device)
        noisy_grads[name] = grad + noise
    return noisy_grads

# ==========================================
# 4. Losses
# ==========================================
def bpr_loss(users_emb, pos_items_emb, neg_items_emb):
    # [cite_start]L_BPR = - sum ln sigma( r_ui - r_uj ) [cite: 189]
    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
    
    loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    return loss

def structural_contrastive_loss(initial_emb, layer_emb, temperature):
    # [cite_start]InfoNCE loss between layer 0 and layer K (even) [cite: 250]
    # Positive pair: same node at different layers
    
    # Normalize
    initial_emb = F.normalize(initial_emb, dim=1)
    layer_emb = F.normalize(layer_emb, dim=1)
    
    # Similarity (dot product)
    pos_score = torch.sum(initial_emb * layer_emb, dim=1) / temperature
    pos_score = torch.exp(pos_score)
    
    # For denominator, we need negatives. 
    # Memory Efficient Implementation: Batch-wise matrix multiplication
    # We calculate sim matrix for the batch
    
    # Note: Full computation is heavy. We compute batch-wise.
    ttl_score = torch.matmul(initial_emb, layer_emb.t()) / temperature
    ttl_score = torch.sum(torch.exp(ttl_score), dim=1)
    
    loss = -torch.log(pos_score / (ttl_score + 1e-8))
    return torch.mean(loss)

# ==========================================
# 5. Federated Components
# ==========================================
class FedPCLServer:
    def __init__(self, num_users, num_items):
        # [cite_start]Global Model [cite: 260]
        self.global_model = LightGCN(num_users, num_items, CONFIG['emb_dim'], CONFIG['gnn_layers']).to(CONFIG['device'])
        
        # Cluster Models (List of state_dicts)
        self.cluster_models = [copy.deepcopy(self.global_model.state_dict()) for _ in range(CONFIG['n_clusters'])]
        
        # User Cluster Assignments (initially random)
        self.user_clusters = np.random.randint(0, CONFIG['n_clusters'], num_users)
        
        # Optimizer (for server-side updates if needed, though FedAvg usually just averages)
        self.num_users = num_users
        
    def aggregate_gradients(self, client_updates, selected_users):
            global_dict = self.global_model.state_dict()
            global_accum = {k: torch.zeros_like(v) for k, v in global_dict.items()}
            cluster_accum = [{k: torch.zeros_like(v) for k, v in global_dict.items()} for _ in range(CONFIG['n_clusters'])]
            cluster_counts = [0] * CONFIG['n_clusters']
            
            for uid, client_grads in client_updates:
                cluster_id = self.user_clusters[uid]
                for k, grad in client_grads.items():
                    global_accum[k] += grad
                    cluster_accum[cluster_id][k] += grad
                cluster_counts[cluster_id] += 1
                
            n_participants = len(selected_users)
            if n_participants > 0:
                for k in global_dict.keys():
                    avg_grad = global_accum[k] / n_participants
                    
                    # --- UPDATE THIS LINE ---
                    # Was: global_dict[k] -= CONFIG['lr'] * avg_grad
                    global_dict[k] -= CONFIG['server_lr'] * avg_grad 
                self.global_model.load_state_dict(global_dict)
                
            for c in range(CONFIG['n_clusters']):
                if cluster_counts[c] > 0:
                    current_cluster_dict = self.cluster_models[c]
                    for k in current_cluster_dict.keys():
                        avg_grad = cluster_accum[c][k] / cluster_counts[c]
                        
                        # --- UPDATE THIS LINE ---
                        # Was: current_cluster_dict[k] -= CONFIG['lr'] * avg_grad
                        current_cluster_dict[k] -= CONFIG['server_lr'] * avg_grad
                    self.cluster_models[c] = current_cluster_dict

    def perform_clustering(self):
        """
        [cite_start]Re-cluster users based on their learned embeddings [cite: 298]
        """
        print("Server: Re-clustering users...")
        with torch.no_grad():
            user_embs = self.global_model.embedding_user.weight.cpu().numpy()
            kmeans = KMeans(n_clusters=CONFIG['n_clusters'], n_init=10)
            self.user_clusters = kmeans.fit_predict(user_embs)

# ==========================================
# 6. Client Simulation
# ==========================================
def client_update(client_id, server, user_interactions, num_users, num_items, current_round):
    """
    Simulates one client's local training steps.
    Returns the gradients (difference in weights).
    """
    device = CONFIG['device']
    
    # [cite_start]1. Fetch Models (Personalized Aggregation) [cite: 280]
    # h_u = mu1 * h_cluster + mu2 * h_global
    cluster_id = server.user_clusters[client_id]
    
    global_state = server.global_model.state_dict()
    cluster_state = server.cluster_models[cluster_id]
    
    # Create Personalized Local Model
    local_model = LightGCN(num_users, num_items, CONFIG['emb_dim'], CONFIG['gnn_layers']).to(device)
    local_state = {}
    
    for k in global_state.keys():
        local_state[k] = CONFIG['mu1'] * cluster_state[k] + CONFIG['mu2'] * global_state[k]
        
    local_model.load_state_dict(local_state)
    local_model.train()
    
    # 2. Local Data Preparation
    # Get user's positive items
    pos_items = user_interactions.get(client_id, [])
    if len(pos_items) == 0:
        return None # Skip inactive users
        
    # [cite_start]Build Local Subgraph (User + Neighbors) [cite: 195]
    # Memory Opt: We only construct the subgraph relevant to this user + sampled negatives
    # For LightGCN on full item set, we need the full adj usually. 
    # Optimization: We use the server's global adj structure but update embeddings locally.
    # In strict FL, client has local subgraph. Here we assume item embeddings are shared/downloaded
    # and we optimize the user's vector and item vectors based on interactions.
    
    # Note: Constructing a new sparse matrix for every client is slow.
    # We will use the global graph structure (stored in server/passed down) but 
    # technically in FL, this represents the client knowing the item set.
    
    # Optimizer
    optimizer = optim.SGD(local_model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['decay'])
    
    # Store initial weights to calculate gradients later
    initial_weights = copy.deepcopy(local_model.state_dict())
    
    # 3. Local Training Loop
    adj_matrix = build_sparse_graph(num_users, num_items, {client_id: pos_items}, device)
    
    for epoch in range(CONFIG['local_epochs']):
        optimizer.zero_grad()
        
        # Forward
        final_u, final_i, embs_list = local_model(adj_matrix)
        
        # BPR Loss
        # Sample Negatives
        neg_item = np.random.randint(0, num_items)
        while neg_item in pos_items:
            neg_item = np.random.randint(0, num_items)
            
        u_emb = final_u[client_id].unsqueeze(0)
        pos_i_emb = final_i[pos_items].mean(dim=0).unsqueeze(0) # Simplified: mean of pos items
        neg_i_emb = final_i[neg_item].unsqueeze(0)
        
        loss_r = bpr_loss(u_emb, pos_i_emb, neg_i_emb)
        
        # [cite_start]Contrastive Loss [cite: 257]
        loss_c = 0
        if current_round > CONFIG['warmup_epochs']: # [cite: 430]
            # Layer 0 vs Layer 2 (Even layer)
            # User Contrastive
            u_0 = embs_list[0][:num_users][client_id].unsqueeze(0)
            u_k = embs_list[-1][:num_users][client_id].unsqueeze(0) # Assuming 2 layers
            loss_c_u = structural_contrastive_loss(u_0, u_k, CONFIG['tau'])
            
            # Item Contrastive (Sampled item)
            # We only calculate for the items involved to save compute
            i_idx = pos_items[0] # Take one pos item for contrast
            i_0 = embs_list[0][num_users:][i_idx].unsqueeze(0)
            i_k = embs_list[-1][num_users:][i_idx].unsqueeze(0)
            loss_c_v = structural_contrastive_loss(i_0, i_k, CONFIG['tau'])
            
            loss_c = CONFIG['beta1'] * (loss_c_u + CONFIG['lambda_val'] * loss_c_v)
            
        total_loss = loss_r + loss_c
        total_loss.backward()
        optimizer.step()
        
    # 4. Compute Gradients & Apply LDP
    final_weights = local_model.state_dict()
    gradients = {}
    
    for k in initial_weights.keys():
        # Gradient = Initial - Final (Direction towards minimum)
        grads = initial_weights[k] - final_weights[k]
        gradients[k] = grads
        
    # [cite_start]Apply LDP [cite: 283, 289]
    noisy_grads = apply_ldp(gradients, CONFIG['ldp_clip'], CONFIG['ldp_sigma'])
    
    # Cleanup to save VRAM
    del local_model, optimizer, adj_matrix, initial_weights, final_weights
    torch.cuda.empty_cache()
    
    return noisy_grads

# ==========================================
# 7. Main Training Loop
# ==========================================
def main():
    # 1. Load Data
    csv_file = "steam-200k.csv" # Ensure this file exists
    if not os.path.exists(csv_file):
        print("Dataset not found! Please ensure 'steam-200k.csv' is in the folder.")
        return

    num_users, num_items, user_interactions, df = load_data(csv_file)
    
    # 2. Initialize Server
    server = FedPCLServer(num_users, num_items)
    
    # 3. Training Loop
    print("\nStarting FedPCL Training...")
    progress_bar = tqdm(range(CONFIG['rounds']), desc="Training Rounds")
    
    for r in progress_bar:
        # A. Server-side Clustering (Periodically)
        if r % 10 == 0:
            server.perform_clustering()
            
        # B. Select Clients
        available_clients = list(user_interactions.keys())
        # Safe sampling if clients < batch
        n_sample = min(CONFIG['n_clients_per_round'], len(available_clients))
        selected_clients = np.random.choice(available_clients, n_sample, replace=False)
        
        # C. Client Updates (Sequential Simulation)
        client_updates = []
        
        # Process clients one by one to save RAM (16GB Constraint)
        for client_id in selected_clients:
            grads = client_update(client_id, server, user_interactions, num_users, num_items, r)
            if grads is not None:
                client_updates.append((client_id, grads))
        
        # D. Server Aggregation
        server.aggregate_gradients(client_updates, selected_clients)
        
        # E. Evaluation (Simple Hit Rate check every 20 rounds)
        if r % 20 == 0 and r > 0:
            # We run a quick evaluation on the global model for a few users
            server.global_model.eval()
            hits = 0
            count = 0
            
            # Construct a global graph for evaluation (approximation)
            full_adj = build_sparse_graph(num_users, num_items, user_interactions, CONFIG['device'])
            
            with torch.no_grad():
                users_emb, items_emb, _ = server.global_model(full_adj)
                
                # Test on 10 random users
                test_users = np.random.choice(list(user_interactions.keys()), 10)
                for u in test_users:
                    # Get user preference scores
                    scores = torch.matmul(users_emb[u], items_emb.t())
                    
                    # Remove items already interacted (train set) from recommendation
                    train_items = user_interactions[u]
                    scores[train_items] = -float('inf')
                    
                    # Top 10
                    _, indices = torch.topk(scores, 10)
                    recs = indices.cpu().numpy().tolist()
                    
                    # In a real split we would check against held-out test set
                    # Here we just print completion status to show progress
                    count += 1
            
            progress_bar.set_postfix({"Status": "Aggregating...", "Cluster0_Size": np.sum(server.user_clusters == 0)})
            
            # Cleanup
            del full_adj
            torch.cuda.empty_cache()

    print("\nTraining Complete.")
    
    # Save Model
    torch.save(server.global_model.state_dict(), "fedpcl_steam_model.pth")
    print("Model saved to 'fedpcl_steam_model.pth'")

if __name__ == "__main__":
    main()
