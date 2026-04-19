import pandas as pd
from collections import defaultdict, Counter
import random

df = pd.read_csv('steam-200k.csv',
                 header=0,
                 names=['user_id', 'item_name', 'action', 'hours', 'extra'])

# ── DIAGNOSIS: TEST ALL STRATEGIES ───────────────────────────────────────────

def k_core_filter(interactions, k=5):
    while True:
        user_counts = Counter(u for u, i in interactions)
        item_counts = Counter(i for u, i in interactions)
        filtered = [(u,i) for u,i in interactions
                    if user_counts[u] >= k and item_counts[i] >= k]
        if len(filtered) == len(interactions):
            break
        interactions = filtered
    return filtered

def stats(interactions, label):
    u = len(set(u for u,i in interactions))
    it = len(set(i for u,i in interactions))
    n = len(interactions)
    print(f"{label:45s} → users:{u:5d}  items:{it:5d}  inters:{n:7d}")

print("="*80)
print(f"{'Strategy':<45}   {'users':>5}  {'items':>5}  {'inters':>7}")
print(f"{'Paper target':<45}   {'3753':>5}  {'5134':>5}  {'114713':>7}")
print("="*80)

# RAW COUNTS
purchase_pairs = list(zip(
    df[df['action']=='purchase']['user_id'],
    df[df['action']=='purchase']['item_name']))

play_pairs = list(zip(
    df[(df['action']=='play') & (df['hours']>0)]['user_id'],
    df[(df['action']=='play') & (df['hours']>0)]['item_name']))

play_pairs_h1 = list(zip(
    df[(df['action']=='play') & (df['hours']>1)]['user_id'],
    df[(df['action']=='play') & (df['hours']>1)]['item_name']))

# Deduplicated union
union_dedup = list(set(purchase_pairs) | set(play_pairs))

stats(purchase_pairs,        "Purchase only (raw)")
stats(play_pairs,            "Play only >0h (raw)")
stats(play_pairs_h1,         "Play only >1h (raw)")
stats(union_dedup,           "Union deduplicated (raw)")
print("-"*80)

# WITH K-CORE FILTERING
for k in [5, 10, 15, 20, 30]:
    stats(k_core_filter(purchase_pairs, k),   f"Purchase only + {k}-core")
for k in [5, 10, 15, 20]:
    stats(k_core_filter(play_pairs, k),        f"Play >0h only + {k}-core")
for k in [5, 10, 15]:
    stats(k_core_filter(union_dedup, k),       f"Union dedup + {k}-core")
print("-"*80)

# USER-ONLY FILTERING (no item filtering) — key insight from diagnosis
print("\n── USER-ONLY MIN INTERACTION FILTER (items untouched) ──")
for min_u in [5, 10, 15, 20, 25, 30]:
    uc = Counter(u for u,i in purchase_pairs)
    filtered = [(u,i) for u,i in purchase_pairs if uc[u] >= min_u]
    stats(filtered, f"Purchase + user_min_interactions>={min_u}")

print("\n── PLAY ONLY, USER-ONLY MIN FILTER ──")
for min_u in [5, 10, 15, 20]:
    uc = Counter(u for u,i in play_pairs)
    filtered = [(u,i) for u,i in play_pairs if uc[u] >= min_u]
    stats(filtered, f"Play >0h + user_min_interactions>={min_u}")
