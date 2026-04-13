"""
data_loader.py  v2
══════════════════
Handles all 5 datasets for the FedPCL paper centralized baseline.

KEY FIXES vs v1:
  FIX 1 — k-core only for sparse datasets (NOT ML-100K / ML-1M)
           ML-100K and ML-1M are already curated benchmarks with dense
           interactions. Applying 5-core removes 40% of items and makes
           ranking artificially easy (fewer items = higher HR@10).
           Paper reports 943/1682 for ML-100K → no k-core applied.

  FIX 2 — Timestamp-based leave-one-out split for ML-100K and ML-1M
           Both datasets have a timestamp column. The correct split is:
           sort each user's interactions by time → last item = test.
           Random split inflates scores by accidentally putting easy items
           in the test set.

  FIX 3 — Item name loading
           load_item_names(dataset, path) returns {item_id_int: name_str}
           for use in recommendation display.

Dataset k-core policy:
  ml100k   → NO k-core  (943 users, 1682 items as in paper)
  ml1m     → NO k-core  (6040 users, 3706 items as in paper)
  steam    → 5-core on users only (paper: 3757 users, 5113 items)
  filmtrust→ 5-core (sparse dataset, paper: 1110 users, 1775 items)
  amazon   → 5-core (very sparse, paper: 1435 users, 1522 items)

Usage:
    from data_loader import load_dataset, load_item_names, build_edge_index
    bundle   = load_dataset('ml100k', 'u.data')
    id2name  = load_item_names('ml100k', 'u.item')
"""

import os, json, random, math
from collections import defaultdict
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Implicit feedback thresholds ──────────────────────────────────────────────
THRESHOLDS = {
    'ml100k':    0,    # any rating = implicit positive (gets all 1682 items)
    'ml1m':      0,    # any rating = implicit positive (gets all 3706 items)
    'filmtrust': 0,    # any rating
    'amazon':    4,
    'steam':     None, # already preprocessed
}

# ── K-core policy per dataset ─────────────────────────────────────────────────
# FIX 1: ML-100K and ML-1M are standard benchmarks — no k-core needed
KCORE = {
    'ml100k':    0,    # 0 = disabled
    'ml1m':      0,
    'steam':     0,    # user-only filtering already done in preprocessing
    'filmtrust': 5,
    'amazon':    5,
}

# ── Timestamp availability ────────────────────────────────────────────────────
# FIX 2: these datasets have timestamps → use temporal split
HAS_TIMESTAMP = {'ml100k', 'ml1m'}


# ══════════════════════════════════════════════════════════════════════════════
#  DataBundle
# ══════════════════════════════════════════════════════════════════════════════
class DataBundle:
    """
    Unified data container for one dataset.

    Attributes:
        n_users, n_items   int
        train_dict  {uid: [iid, ...]}    training items per user
        test_dict   {uid: iid}           single held-out test item per user
        neg_dict    {uid: [iid, ...]}    100 negatives per user (evaluation)
        adj_user    {uid: [iid, ...]}    same as train_dict (GNN adjacency)
        adj_item    {iid: [uid, ...]}    item→users adjacency
        deg_user    np.array [n_users]   user interaction degree
        deg_item    np.array [n_items]   item interaction degree
        all_items   set                  complete item ID set {0..n_items-1}
        name        str                  dataset name
    """
    def __init__(self):
        self.n_users = self.n_items = 0
        self.train_dict = self.test_dict = self.neg_dict = {}
        self.adj_user = self.adj_item = {}
        self.deg_user = self.deg_item = None
        self.all_items = set()
        self.name = ''

    def __repr__(self):
        n_train = sum(len(v) for v in self.train_dict.values())
        density = n_train / max(self.n_users * self.n_items, 1) * 100
        return (f"DataBundle({self.name})  users={self.n_users}  "
                f"items={self.n_items}  train={n_train}  density={density:.3f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset(dataset_name: str, data_path: str,
                 n_negatives: int = 100) -> DataBundle:
    """
    Load and preprocess a dataset into a DataBundle.

    Args:
        dataset_name:  'steam' | 'ml100k' | 'ml1m' | 'filmtrust' | 'amazon'
        data_path:     path to raw data file
        n_negatives:   number of eval negatives per user (paper: 100)
    """
    name = dataset_name.lower().replace('-','').replace('_','')

    if name == 'steam':
        interactions = _load_steam(data_path)
        use_timestamp = False
    elif name == 'ml100k':
        interactions = _load_ml100k(data_path)
        use_timestamp = True
    elif name == 'ml1m':
        interactions = _load_ml1m(data_path)
        use_timestamp = True
    elif name == 'filmtrust':
        interactions = _load_filmtrust(data_path)
        use_timestamp = False
    elif name in ('amazon','amazonelectronic','amazonelectronics'):
        interactions = _load_amazon(data_path)
        use_timestamp = False
        name = 'amazon'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    k = KCORE.get(name, 5)
    return _build_bundle(interactions, name, n_negatives,
                         use_timestamp=use_timestamp, kcore=k)


# ══════════════════════════════════════════════════════════════════════════════
#  ITEM NAME LOADER  (FIX 3)
# ══════════════════════════════════════════════════════════════════════════════
def load_item_names(dataset_name: str, names_path: str,
                    item2id: dict = None) -> dict:
    """
    Load human-readable item names for display in recommendations.

    Args:
        dataset_name:  dataset name
        names_path:    path to item name file
        item2id:       {raw_item_id: int_id} mapping from DataBundle
                       (needed to convert raw IDs to model's integer IDs)

    Returns:
        {int_item_id: name_string}

    File formats:
        ml100k  → u.item:  movie_id|movie_title|year|...  (pipe-separated)
        ml1m    → movies.dat:  MovieID::Title::Genres
        steam   → embedded in steam_processed.json as item2id
        others  → no standard name file, returns empty dict
    """
    name = dataset_name.lower().replace('-','').replace('_','')
    id2name_raw = {}

    if not os.path.exists(names_path):
        print(f"  [Names] File not found: {names_path}  — run with u.item in same folder")
        return {}

    try:
        if name == 'ml100k':
            # u.item: movie_id|movie_title|release_date|...  (pipe-separated, latin-1)
            with open(names_path, encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        try:
                            raw_id = int(parts[0])
                            title  = parts[1].strip()
                            id2name_raw[raw_id] = title
                        except: continue

        elif name == 'ml1m':
            # movies.dat: MovieID::Title::Genres
            with open(names_path, encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 2:
                        try:
                            raw_id = int(parts[0])
                            title  = parts[1].strip()
                            id2name_raw[raw_id] = title
                        except: continue

        elif name == 'steam':
            with open(names_path) as f: d = json.load(f)
            return {int(v): k for k,v in d.get('item2id',{}).items()}

        print(f"  [Names] Loaded {len(id2name_raw)} item names from {names_path}")

    except Exception as e:
        print(f"  [Names] Error loading: {e}")
        return {}

    # Convert raw IDs → model integer IDs
    if item2id is None:
        return {k: v for k, v in id2name_raw.items()}

    int2name = {}
    for raw_id, name_str in id2name_raw.items():
        if raw_id in item2id:
            int2name[item2id[raw_id]] = name_str
    print(f"  [Names] Mapped {len(int2name)} names to model item IDs")
    return int2name


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET-SPECIFIC LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_steam(path: str):
    """steam_processed.json — already preprocessed."""
    with open(path) as f: d = json.load(f)
    train = {int(k): v for k,v in d['train_data'].items()}
    test  = {int(k): int(v) for k,v in d['test_data'].items()}
    rows  = []
    for uid, items in train.items():
        for iid in items: rows.append((uid, iid, 0))
    for uid, iid in test.items():
        rows.append((uid, iid, 0))
    return rows   # (user, item, timestamp=0)


def _load_ml100k(path: str):
    """
    u.data: user_id TAB item_id TAB rating TAB timestamp
    FIX: preserve timestamp for temporal split.
    Returns (user_id, item_id, timestamp) for rating >= 4.
    """
    rows = []
    with open(path) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 4: continue
            uid, iid, rating, ts = int(p[0]), int(p[1]), float(p[2]), int(p[3])
            if rating >= THRESHOLDS['ml100k']:
                rows.append((uid, iid, ts))
    return rows


def _load_ml1m(path: str):
    """
    ratings.dat: UserID::MovieID::Rating::Timestamp
    FIX: preserve timestamp for temporal split.
    Returns (user_id, item_id, timestamp) for rating >= 4.
    """
    rows = []
    with open(path, encoding='latin-1') as f:
        for line in f:
            p = line.strip().split('::')
            if len(p) < 4: continue
            uid, iid, rating, ts = int(p[0]), int(p[1]), float(p[2]), int(p[3])
            if rating >= THRESHOLDS['ml1m']:
                rows.append((uid, iid, ts))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET-SPECIFIC LOADERS
# ══════════════════════════════════════════════════════════════════════════════

# UPDATED: FilmTrust parser now supports comma, tab, or space separated files
# Reason: many FilmTrust distributions use "user,item,rating" instead of
# the original "user item rating" format. Without this fix the loader
# reads zero interactions and the dataset becomes empty.

# UPDATED: generate increasing timestamp to allow chronological split
def _load_filmtrust(path: str):
    rows = []
    t = 0
    with open(path) as f:
        for line in f:
            line = line.strip().replace(',', ' ').replace('\t',' ')
            p = line.split()
            if len(p) < 3:
                continue
            uid = int(p[0])
            iid = int(p[1])
            rating = float(p[2])
            if rating > THRESHOLDS['filmtrust']:
                rows.append((uid, iid, t))
                t += 1
    return rows


def _load_amazon(path: str):
    """CSV or JSON lines. Returns (user_id, item_id, timestamp) for rating>=4."""
    rows = []
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.json', '.jsonl'):
        with open(path) as f:
            for line in f:
                try:
                    obj  = json.loads(line.strip())
                    uid  = obj.get('reviewerID','')
                    iid  = obj.get('asin','')
                    rate = float(obj.get('overall', 0))
                    ts   = int(obj.get('unixReviewTime', 0))
                    if rate >= THRESHOLDS['amazon']:
                        rows.append((uid, iid, ts))
                except: continue
    else:
        import csv
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    uid  = row.get('reviewerID', row.get('user_id',''))
                    iid  = row.get('asin', row.get('item_id',''))
                    rate = float(row.get('overall', row.get('rating', 0)))
                    ts   = int(row.get('unixReviewTime', row.get('timestamp', 0)))
                    if rate >= THRESHOLDS['amazon']:
                        rows.append((uid, iid, ts))
                except: continue
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  BUNDLE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _build_bundle(interactions: list, name: str,
                  n_negatives: int,
                  use_timestamp: bool = False,
                  kcore: int = 5) -> DataBundle:
    """
    Build DataBundle from (user, item, timestamp) triples.

    FIX 1: kcore=0 → skip k-core filtering for curated datasets
    FIX 2: use_timestamp=True → sort by time for leave-one-out split
    """
    # ── Dedup (keep latest timestamp if duplicate) ────────────────────────────
    seen = {}
    for u, i, ts in interactions:
        key = (u, i)
        if key not in seen or ts > seen[key]:
            seen[key] = ts
    interactions = [(u, i, ts) for (u,i), ts in seen.items()]

    # ── K-core filtering (FIX 1: skipped for ml100k, ml1m) ───────────────────
    if kcore >= 2:
        interactions = _k_core(interactions, k=kcore)

    # ── Re-index to 0-based integers ──────────────────────────────────────────
    raw_users = sorted(set(u for u,i,ts in interactions))
    raw_items = sorted(set(i for u,i,ts in interactions))
    user2id   = {u: idx for idx,u in enumerate(raw_users)}
    item2id   = {i: idx for idx,i in enumerate(raw_items)}
    indexed   = [(user2id[u], item2id[i], ts) for u,i,ts in interactions]

    n_users = len(user2id)
    n_items = len(item2id)

    # ── Group by user, sort by timestamp (FIX 2) ──────────────────────────────
    user_items = defaultdict(list)  # uid → [(iid, ts), ...]
    for u, i, ts in indexed:
        user_items[u].append((i, ts))

    if use_timestamp:
        # Temporal sort: last interaction by time = test item
        for u in user_items:
            user_items[u].sort(key=lambda x: x[1])
    else:
        # Random reproducible shuffle for datasets without timestamps
        for u in user_items:
            random.shuffle(user_items[u])

    # ── Leave-one-out split: last item per user → test ────────────────────────
    train_dict = {}
    test_dict  = {}
    for u, item_ts_list in user_items.items():
        items_ordered = [iid for iid, ts in item_ts_list]
        test_dict[u]  = items_ordered[-1]
        train_dict[u] = items_ordered[:-1]

    # ── Adjacency + degree ────────────────────────────────────────────────────
    adj_user = defaultdict(list)
    adj_item = defaultdict(list)
    for u, items in train_dict.items():
        for i in items:
            adj_user[u].append(i)
            adj_item[i].append(u)

    deg_user = np.array([len(adj_user[u]) for u in range(n_users)], dtype=np.float32)
    deg_item = np.array([len(adj_item[i]) for i in range(n_items)], dtype=np.float32)
    deg_user[deg_user == 0] = 1
    deg_item[deg_item == 0] = 1

    # ── Negative sampling for evaluation ──────────────────────────────────────
    all_items = set(range(n_items))
    neg_dict  = {}
    for u in user_items:
        seen_u    = set(iid for iid,_ in user_items[u])
        neg_pool  = list(all_items - seen_u)
        n_neg     = min(n_negatives, len(neg_pool))
        neg_dict[u] = random.sample(neg_pool, n_neg)

    # ── Assemble ──────────────────────────────────────────────────────────────
    bundle            = DataBundle()
    bundle.name       = name
    bundle.n_users    = n_users
    bundle.n_items    = n_items
    bundle.train_dict = dict(train_dict)
    bundle.test_dict  = test_dict
    bundle.neg_dict   = neg_dict
    bundle.adj_user   = dict(adj_user)
    bundle.adj_item   = dict(adj_item)
    bundle.deg_user   = deg_user
    bundle.deg_item   = deg_item
    bundle.all_items  = all_items
    bundle._item2id   = item2id   # keep for name loading

    print(repr(bundle))
    n_train = sum(len(v) for v in train_dict.values())
    print(f"  train={n_train}  test={len(test_dict)}  "
          f"neg_per_user={n_negatives}  "
          f"kcore={'off' if kcore<2 else kcore}  "
          f"split={'timestamp' if use_timestamp else 'random'}")
    return bundle


def _k_core(interactions, k=5):
    """Iterative k-core: remove users/items with fewer than k interactions."""
    from collections import Counter
    while True:
        uc = Counter(u for u,i,ts in interactions)
        ic = Counter(i for u,i,ts in interactions)
        filtered = [(u,i,ts) for u,i,ts in interactions if uc[u]>=k and ic[i]>=k]
        if len(filtered) == len(interactions): break
        interactions = filtered
    return interactions


# ══════════════════════════════════════════════════════════════════════════════
#  EDGE INDEX BUILDER  (for LightGCN sparse propagation)
# ══════════════════════════════════════════════════════════════════════════════
def build_edge_index(bundle: DataBundle):
    """
    Build bidirectional sparse edge index for the bipartite user-item graph.
    Users: nodes 0..n_users-1
    Items: nodes n_users..n_users+n_items-1

    Returns:
        edge_index  [2, 2E]  LongTensor
        edge_weight [2E]     FloatTensor  (LightGCN norm: 1/sqrt(du*di))
    """
    import torch
    rows, cols, weights = [], [], []
    for u, items in bundle.adj_user.items():
        du = bundle.deg_user[u]
        for i in items:
            di = bundle.deg_item[i]
            w  = 1.0 / math.sqrt(float(du) * float(di))
            rows.append(u);              cols.append(bundle.n_users + i)
            rows.append(bundle.n_users+i); cols.append(u)
            weights.extend([w, w])
    return (torch.tensor([rows, cols], dtype=torch.long),
            torch.tensor(weights,      dtype=torch.float32))


# ══════════════════════════════════════════════════════════════════════════════
#  NEGATIVE SAMPLER  (per training batch)
# ══════════════════════════════════════════════════════════════════════════════
def sample_negatives_batch(users, train_dict, n_items):
    """Sample one random unseen negative item per user in the batch."""
    neg = []
    for u in users:
        seen = set(train_dict.get(u, []))
        while True:
            j = random.randint(0, n_items-1)
            if j not in seen:
                neg.append(j); break
    return neg
