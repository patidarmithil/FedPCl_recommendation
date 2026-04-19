import pandas as pd
from collections import defaultdict, Counter
import random
import json

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv('steam-200k.csv',
                 header=0,
                 names=['user_id', 'item_name', 'action', 'hours', 'extra'])

# ── 2. STRATEGY: Purchase only + user_min_interactions >= 5 ──────────────────
# This is by far the closest match to paper's 3753/5134/114713
# Reasoning: play data is secondary signal, purchase = primary positive signal

purchase_df = df[df['action'] == 'purchase'][['user_id', 'item_name']].copy()
purchase_df = purchase_df.drop_duplicates()   # safety dedup (shouldn't matter)

# Count purchases per user
user_purchase_counts = purchase_df['user_id'].value_counts()

# Keep only users with >= 5 purchases (user-only filtering, items untouched)
valid_users = user_purchase_counts[user_purchase_counts >= 5].index
filtered_df = purchase_df[purchase_df['user_id'].isin(valid_users)]

interactions = list(zip(filtered_df['user_id'], filtered_df['item_name']))

print("── AFTER STRATEGY FILTER ────────────────────────────────────────")
print(f"Users:        {len(set(u for u,i in interactions)):>6}   (paper: 3753)")
print(f"Items:        {len(set(i for u,i in interactions)):>6}   (paper: 5134)")
print(f"Interactions: {len(interactions):>6}   (paper: 114713)")

# ── 3. RE-INDEX TO CONSECUTIVE INTEGERS ──────────────────────────────────────
unique_users = sorted(set(u for u, i in interactions))
unique_items = sorted(set(i for u, i in interactions))

user2id = {u: idx for idx, u in enumerate(unique_users)}
item2id = {i: idx for idx, i in enumerate(unique_items)}
id2user = {v: k for k, v in user2id.items()}
id2item = {v: k for k, v in item2id.items()}

interactions_indexed = [(user2id[u], item2id[i]) for u, i in interactions]

# ── 4. GROUP INTERACTIONS BY USER ────────────────────────────────────────────
user_items = defaultdict(list)
for u, i in interactions_indexed:
    user_items[u].append(i)

# Sanity check distribution
counts = [len(v) for v in user_items.values()]
print(f"\nInteractions per user:")
print(f"  min={min(counts)}, max={max(counts)}, "
      f"avg={sum(counts)/len(counts):.1f}, "
      f"median={sorted(counts)[len(counts)//2]}")

# ── 5. LEAVE-ONE-OUT SPLIT ────────────────────────────────────────────────────
# Steam has no timestamp column → use random shuffle with fixed seed
# Document this in your report: "Random leave-one-out with seed=42"
random.seed(42)

train_data = {}   # user_id (int) → list of item_ids for training
test_data  = {}   # user_id (int) → single item_id for testing

for u, items in user_items.items():
    items_copy = items.copy()
    random.shuffle(items_copy)
    test_data[u]  = items_copy[-1]     # 1 held-out item → test
    train_data[u] = items_copy[:-1]    # everything else → train

print(f"\nSplit:")
print(f"  Train interactions: {sum(len(v) for v in train_data.values())}")
print(f"  Test  interactions: {len(test_data)} (1 per user)")

# ── 6. VERIFY NO TRAIN/TEST OVERLAP ──────────────────────────────────────────
overlaps = sum(1 for u in test_data
               if test_data[u] in set(train_data[u]))
print(f"  Train/test overlaps: {overlaps}  (must be 0)")

# ── 7. NEGATIVE SAMPLING FOR EVALUATION ──────────────────────────────────────
# Standard protocol: 100 random unseen items per user
# Combined with test item → rank among 101 candidates
all_item_ids = set(range(len(item2id)))

test_negatives = {}
for u in user_items:
    seen_items    = set(user_items[u])
    candidate_neg = list(all_item_ids - seen_items)

    if len(candidate_neg) < 100:
        print(f"WARNING: user {u} has only {len(candidate_neg)} possible negatives")
        test_negatives[u] = candidate_neg
    else:
        test_negatives[u] = random.sample(candidate_neg, 100)

neg_counts = [len(v) for v in test_negatives.values()]
print(f"\nNegative samples: min={min(neg_counts)}, max={max(neg_counts)}")

# ── 8. BUILD LOCAL SUBGRAPH STRUCTURE ────────────────────────────────────────
# For federated setting: each client n has local bipartite graph G_n
# G_n = edges between user n and their training items only
# This is what gets passed to the GNN on each client

local_graphs = {}
for u in train_data:
    # Edge list: (user_node, item_node) pairs
    local_graphs[u] = [(u, i) for i in train_data[u]]

# Verify a sample
print(f"\nSample local graph for user 0:")
print(f"  Training items: {train_data[0]}")
print(f"  Test item:      {test_data[0]}")
print(f"  Negatives (first 5): {test_negatives[0][:5]}")
print(f"  Graph edges (first 3): {local_graphs[0][:3]}")

# ── 9. DATASET STATISTICS REPORT ─────────────────────────────────────────────
n_train = sum(len(v) for v in train_data.values())
n_test  = len(test_data)
density = len(interactions) / (len(user2id) * len(item2id)) * 100

print("\n── FINAL DATASET STATISTICS ─────────────────────────────────")
print(f"{'Metric':<30} {'Yours':>10} {'Paper':>10} {'Diff':>8}")
print("-"*60)
print(f"{'Users':<30} {len(user2id):>10} {3753:>10} "
      f"{len(user2id)-3753:>+8}")
print(f"{'Items':<30} {len(item2id):>10} {5134:>10} "
      f"{len(item2id)-5134:>+8}")
print(f"{'Total interactions':<30} {len(interactions):>10} {114713:>10} "
      f"{len(interactions)-114713:>+8}")
print(f"{'Train interactions':<30} {n_train:>10}")
print(f"{'Test interactions':<30} {n_test:>10}")
print(f"{'Density (%)':<30} {density:>10.4f}")
print(f"{'Negatives per user':<30} {'100':>10}")
print(f"{'Random seed':<30} {'42':>10}")

# ── 10. SAVE PROCESSED DATA ───────────────────────────────────────────────────
processed = {
    'dataset':        'steam-200k',
    'n_users':        len(user2id),
    'n_items':        len(item2id),
    'n_interactions': len(interactions),
    'train_data':     {str(k): v for k, v in train_data.items()},
    'test_data':      {str(k): int(v) for k, v in test_data.items()},
    'test_negatives': {str(k): v for k, v in test_negatives.items()},
    'user2id':        {str(k): v for k, v in user2id.items()},
    'item2id':        {str(k): v for k, v in item2id.items()},
}

with open('steam_processed.json', 'w') as f:
    json.dump(processed, f, indent=2)

print("\nSaved → steam_processed.json")
