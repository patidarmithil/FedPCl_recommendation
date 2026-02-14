import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'emb_dim': 64,       # Must match training
    'gnn_layers': 2,     # Must match training
    'top_k': 10
}

# ==========================================
# 1. Model Definition (Must Match Training)
# ==========================================
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, n_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.embedding_user = nn.Embedding(num_users, emb_dim)
        self.embedding_item = nn.Embedding(num_items, emb_dim)
        
    def forward(self, adj_sparse):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_sparse, all_emb)
            embs.append(all_emb)
        stack_embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(stack_embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        return users, items

# ==========================================
# 2. Data Loading (Identical to Training)
# ==========================================
def load_data_and_maps(csv_path):
    print("Loading and Filtering Data (Type-Safe Version)...")
    try:
        # FIX: Force user_id to be read as string to prevent 'int' vs 'str' errors
        df = pd.read_csv(csv_path, names=['user_id', 'item_name', 'action', 'hours', 'extra'], dtype={'user_id': str})
    except:
        df = pd.read_csv(csv_path, dtype={'user_id': str})

    # 1. Keep 'purchase' and 'play'
    df = df[df['action'].isin(['purchase', 'play'])]
    
    # 2. Deduplicate
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

    print(f"Eval Data: {df['user_id'].nunique()} users, {len(df)} interactions.")

    # 4. Mappings (Sorted strings are safe)
    user_ids = sorted(df['user_id'].unique())
    item_ids = sorted(df['item_name'].unique())
    
    user2id = {u: i for i, u in enumerate(user_ids)}
    item2id = {i: j for j, i in enumerate(item_ids)}
    id2user = {i: u for u, i in user2id.items()}
    id2item = {j: i for i, j in item2id.items()}
    
    df['u_idx'] = df['user_id'].map(user2id)
    df['i_idx'] = df['item_name'].map(item2id)
    
    # Create Interaction Dictionary
    user_interactions = df.groupby('u_idx')['i_idx'].apply(list).to_dict()
    
    return len(user_ids), len(item_ids), user_interactions, id2user, id2item


def build_sparse_graph(num_users, num_items, interaction_dict, device):
    src_nodes = []
    dst_nodes = []
    for u, items in interaction_dict.items():
        for i in items:
            src_nodes.append(u)
            dst_nodes.append(i + num_users)
            
    rows = src_nodes + dst_nodes
    cols = dst_nodes + src_nodes
    indices = torch.LongTensor([rows, cols])
    values = torch.ones(len(rows))
    
    adj = torch.sparse_coo_tensor(indices, values, (num_users + num_items, num_users + num_items))
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    
    values_norm = values * d_inv_sqrt[rows] * d_inv_sqrt[cols]
    return torch.sparse_coo_tensor(indices, values_norm, (num_users + num_items, num_users + num_items)).to(device)

# ==========================================
# 3. Strict Leave-One-Out Metrics
# ==========================================
def calculate_metrics(model, adj, user_interactions, num_users, num_items, device):
    model.eval()
    HR, NDCG = [], []
    
    print(f"Evaluating on all {len(user_interactions)} users...")
    
    with torch.no_grad():
        users_emb, items_emb = model(adj)
        
        # Batch processing to prevent OOM on laptop if we did matrix mult all at once
        # But for 3700 users, loop is fine.
        
        for u_id in tqdm(list(user_interactions.keys())):
            interactions = user_interactions[u_id]
            
            # Need at least 2 items (1 train, 1 test)
            if len(interactions) < 2: 
                continue
                
            # STRICT LOO: Last item is Test, rest are Train
            test_item = interactions[-1]
            train_items = set(interactions[:-1])
            
            # 1. Calculate Scores for ALL items
            # Score = User_Vector . Item_Matrix_T
            scores = torch.matmul(users_emb[u_id], items_emb.t())
            
            # 2. Mask Training Items (Set score to -infinity)
            # We don't want to recommend things they already trained on
            scores[list(train_items)] = -float('inf')
            
            # 3. Metric Calculation
            # We want to see where 'test_item' ranks among all unplayed items.
            # Paper Protocol: Rank Test Item vs 100 Random Negatives (Common in literature)
            # OR Rank Test Item vs All Unplayed Items (Full Ranking)
            # Given the paper doesn't specify "Sampled Metrics" in the text provided, 
            # but usually "HR@10" implies Full Ranking or 100-negatives. 
            # 100-negatives is faster and standard. Let's use 100 negatives.
            
            # Sample 100 negatives
            neg_candidates = []
            while len(neg_candidates) < 100:
                n = np.random.randint(0, num_items)
                if n not in train_items and n != test_item:
                    neg_candidates.append(n)
            
            # Combine Test Item + 100 Negatives
            eval_items = [test_item] + neg_candidates
            eval_scores = scores[eval_items]
            
            # Rank (Top 10)
            _, top_indices = torch.topk(eval_scores, 10)
            top_indices = top_indices.cpu().numpy()
            
            # Check if index 0 (the test item) is in the top 10
            if 0 in top_indices:
                HR.append(1)
                # Calculate NDCG
                rank = np.where(top_indices == 0)[0][0]
                NDCG.append(1 / math.log2(rank + 2))
            else:
                HR.append(0)
                NDCG.append(0)

    return np.mean(HR), np.mean(NDCG)
def show_example_recommendations(model, adj, user_interactions, id2user, id2item, num_users, num_items, device):
    print("\n--- Example User Recommendation ---")
    model.eval()
    
    # 1. Pick a random user who has at least 5 interactions
    valid_users = [u for u, items in user_interactions.items() if len(items) >= 5]
    if not valid_users:
        print("No users with enough interactions found.")
        return

    random_u_idx = np.random.choice(valid_users)
    real_user_id = id2user[random_u_idx]
    
    # 2. Get their history
    history_item_indices = user_interactions[random_u_idx]
    history_names = [id2item[i] for i in history_item_indices]
    
    print(f"User ID: {real_user_id}")
    print(f"History ({len(history_names)} items): {history_names[:5]}...") # Show first 5
    
    # 3. Generate Recommendations
    with torch.no_grad():
        users_emb, items_emb = model(adj)
        user_vector = users_emb[random_u_idx]
        
        # Calculate score for ALL items
        all_scores = torch.matmul(user_vector, items_emb.t())
        
        # Mask out items they already bought (we don't want to recommend history)
        all_scores[history_item_indices] = -float('inf')
        
        # Get Top 10
        _, top_indices = torch.topk(all_scores, 10)
        top_indices = top_indices.cpu().numpy()
        
        print("\nTop 10 Recommendations:")
        for rank, idx in enumerate(top_indices):
            item_name = id2item[idx]
            print(f"{rank+1}: {item_name}")

# ==========================================
# 4. Main
# ==========================================
if __name__ == "__main__":
    csv_file = "steam-200k.csv"
    model_file = "fedpcl_steam_model.pth"
    
    # A. Load Data
    num_users, num_items, interactions, id2user, id2item = load_data_and_maps(csv_file)
    
    # B. Load Model
    print("Loading Model...")
    try:
        model = LightGCN(num_users, num_items, CONFIG['emb_dim'], CONFIG['gnn_layers']).to(CONFIG['device'])
        model.load_state_dict(torch.load(model_file, map_location=CONFIG['device']))
    except RuntimeError as e:
        print("\nERROR: Model size mismatch!")
        print("This means you trained the model on different data than you are evaluating.")
        print("Please re-run 'fedpcl.py' first to train on the filtered 5-core data.")
        exit()
    
    # C. Build Graph
    adj = build_sparse_graph(num_users, num_items, interactions, CONFIG['device'])
    
    # D. Evaluate
    print("Calculating Strict LOO Metrics...")
    hr, ndcg = calculate_metrics(model, adj, interactions, num_users, num_items, CONFIG['device'])
    
    print(f"\n--- Final Results [cite: 410] ---")
    print(f"Hit Rate @ 10: {hr * 100:.2f}%")
    print(f"NDCG @ 10:     {ndcg * 100:.2f}%")
    # E. SHOW EXAMPLE (ADD THIS LINE)
    show_example_recommendations(model, adj, interactions, id2user, id2item, num_users, num_items, CONFIG['device'])
