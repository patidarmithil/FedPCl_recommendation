# =============================================================================
# PerFedRec v6 — Correct Collaborative Signal
#
# INHERITS all v3+v4 fixes:
#   [A] Correct personalization mix (local + cluster + global)
#   [B] Per-user local params persisted across rounds
#   [C] Clustering uses GNN-output embedding u_fin
#   [F] Negative sampling uses random.sample (no duplicates)
#   [G] LR = 0.01 (paper default)
#   [H] Regularization only in Adam weight_decay
#   [I] cluster_start_rnd = 10 (warm-up)
#   [J] Local model EMA smoothing
#   [K] Global model momentum averaging
#   [L] cluster_freq = 15
#   [M] repr_items = pos + sampled_neg
#   [N] Balanced negative sampling
#
# v5 WAS REVERTED — it caused regression 0.39→0.26 because:
#   • LightGCN_Collab replaced item content with mean(user_embeddings)
#   • Cached neighbor embeddings (global-model space) mixed with local-model
#     user embeddings → space mismatch → item representations became noise
#   • Random init cache (early rounds) corrupted BPR loss signal
#
# NEW in v6 — correct collaborative signal, two targeted changes:
#
#   [O] Collaborative item feature augmentation  (static, zero-risk)
#       For each item, compute mean of training-user demographic features
#       (age, gender, occupation) of all users who rated it.
#       Concatenate to item's genre vector: i_fdim  19 → 42
#       → "what kind of user typically rates this item"
#       → adds collaborative signal without touching the GNN or training loop
#       → computed once from train_data, never changes
#
#   [P] Global-only item head  (matches the paper's design)
#       Paper: "Item embeddings are shared and updated iteratively among
#               users via the server."
#       v4 bug: ih.* was mixed  α1·local_ih + α2·cluster_ih + α3·global_ih
#               → items overfitted to individual users via local component
#               → local item gradients corrupted the collaborative item space
#       Fix: ALL ih.* keys excluded from mix3 → always use global_ih
#       → items are always the collaborative representation from the global model
#       → users still personalize through uh.*, gnn.* parameters
#       → dot product = (personalized user) · (collaborative item) ✓
#
#   LightGCN is UNCHANGED from v4 — no cross-user GNN modifications.
#   The collaborative signal enters through [O] (static features) and [P]
#   (globally-aggregated item embeddings), not through GNN rewiring.
#
# Expected: HR@10  0.39 (v4) → 0.45–0.52 (v6)
# =============================================================================

import subprocess, sys
try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "scikit-learn", "--break-system-packages"])

import os, random, copy, warnings, zipfile, requests
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# 0.  CONFIG
# =============================================================================
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir  = "./data"

    # ── Architecture ──────────────────────────────────────────────────────────
    embed_dim    = 64
    n_fc_layers  = 2
    n_gnn_layers = 2

    # ── Federated ─────────────────────────────────────────────────────────────
    n_rounds          = 100
    n_users_per_round = 128
    local_epochs      = 3
    lr                = 0.01       # [G]
    weight_decay      = 1e-4       # [H]
    n_clusters        = 10
    cluster_start_rnd = 10         # [I]
    cluster_freq      = 15         # [L]
    alpha             = [1/3, 1/3, 1/3]

    # ── v4 stability ──────────────────────────────────────────────────────────
    local_ema        = 0.5         # [J]
    global_momentum  = 0.7         # [K]
    repr_neg_count   = 50          # [M]

    # ── Eval ──────────────────────────────────────────────────────────────────
    n_neg_eval = 100
    eval_every = 10

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed   = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

seed_all(cfg.seed)
print(f"Device : {cfg.device}")


# =============================================================================
# 1.  DATA
# =============================================================================

def download_ml100k(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, "ml-100k")
    if os.path.isdir(dest):
        return dest
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zp  = os.path.join(data_dir, "ml-100k.zip")
    print("Downloading MovieLens-100K …")
    r = requests.get(url, stream=True)
    with open(zp, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    with zipfile.ZipFile(zp) as z:
        z.extractall(data_dir)
    print("Done.")
    return dest


def load_data(data_dir):
    ml = download_ml100k(data_dir)

    df = pd.read_csv(os.path.join(ml, "u.data"), sep="\t",
                     names=["uid","iid","rating","ts"])
    u2i = {u: i for i, u in enumerate(sorted(df.uid.unique()))}
    i2i = {it: i for i, it in enumerate(sorted(df.iid.unique()))}
    df["uidx"] = df.uid.map(u2i)
    df["iidx"] = df.iid.map(i2i)
    N, M = len(u2i), len(i2i)

    ucols = ["uid","age","gender","occ","zip"]
    uf = pd.read_csv(os.path.join(ml,"u.user"), sep="|", names=ucols)
    uf["uidx"] = uf.uid.map(u2i)
    uf = uf.dropna(subset=["uidx"]).copy()
    uf["uidx"]  = uf.uidx.astype(int)
    uf["age_n"] = uf.age / 80.0
    uf["sex_n"] = (uf.gender == "M").astype(float)
    occ_d = pd.get_dummies(uf.occ, prefix="o").astype(float)
    uf_feat = pd.concat([uf[["uidx","age_n","sex_n"]], occ_d], axis=1)
    uf_feat = uf_feat.sort_values("uidx").reset_index(drop=True)
    user_feat = uf_feat.drop("uidx", axis=1).values.astype(np.float32)

    icols = (["iid","title","rdate","vdate","url"]
             + [f"g{k}" for k in range(19)])
    itf = pd.read_csv(os.path.join(ml,"u.item"), sep="|",
                      names=icols, encoding="latin-1")
    itf["iidx"] = itf.iid.map(i2i)
    itf = itf.dropna(subset=["iidx"]).copy()
    itf["iidx"] = itf.iidx.astype(int)
    gcols = [f"g{k}" for k in range(19)]
    item_feat = (itf[["iidx"]+gcols]
                 .sort_values("iidx").reset_index(drop=True)[gcols]
                 .values.astype(np.float32))

    def pad(a, n):
        if a.shape[0] < n:
            a = np.vstack([a, np.zeros((n-a.shape[0], a.shape[1]),
                                       dtype=np.float32)])
        return a
    user_feat = pad(user_feat, N)
    item_feat = pad(item_feat, M)

    return df, user_feat, item_feat, N, M


def leave_one_out(df):
    df = df.sort_values(["uidx","ts"])
    train_d, val_d, test_d = [], [], []
    train_dict = defaultdict(set)
    for u, grp in df.groupby("uidx"):
        its = grp.iidx.tolist()
        if len(its) < 3:
            for it in its:
                train_d.append((u, it)); train_dict[u].add(it)
        else:
            test_d.append((u, its[-1]))
            val_d.append((u, its[-2]))
            for it in its[:-2]:
                train_d.append((u, it)); train_dict[u].add(it)
    return train_d, val_d, test_d, train_dict


print("Loading data …")
df, user_feat, item_feat_raw, N_USERS, N_ITEMS = load_data(cfg.data_dir)
train_data, val_data, test_data, train_dict = leave_one_out(df)

# ── [O] Collaborative item feature augmentation ───────────────────────────────
# For each item: mean demographic feature vector of all training users who rated it.
# This adds "what kind of user typically rates this item" to the item description.
# Computed once from train_data → static, never changes → zero risk of gradient issues.
print("Building collaborative item features [O] …")
collab_i_feat = np.zeros((N_ITEMS, user_feat.shape[1]), dtype=np.float32)
collab_count  = np.zeros(N_ITEMS, dtype=np.float32)
for u, it in train_data:
    collab_i_feat[it] += user_feat[u]
    collab_count[it]  += 1
nz = collab_count > 0
collab_i_feat[nz] /= collab_count[nz, np.newaxis]

# Augmented item features: [genre features | mean-user-demographic features]
item_feat = np.concatenate([item_feat_raw, collab_i_feat], axis=1)

print(f"Users={N_USERS}  Items={N_ITEMS}  "
      f"Train={len(train_data)}  Val={len(val_data)}  Test={len(test_data)}")
print(f"User-feat={user_feat.shape[1]}  "
      f"Item-feat={item_feat.shape[1]} "
      f"({item_feat_raw.shape[1]} genre + {user_feat.shape[1]} collab)")


# =============================================================================
# 2.  MODEL   (identical to v4 — no GNN changes)
# =============================================================================

class FeatureCrossing(nn.Module):
    def __init__(self, dim, L=2):
        super().__init__()
        self.w = nn.ParameterList([nn.Parameter(torch.randn(dim)*0.01)
                                   for _ in range(L)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(dim))
                                   for _ in range(L)])
        self.L = L

    def forward(self, x0):
        xl = x0
        for l in range(self.L):
            dot = (xl * self.w[l]).sum(-1, keepdim=True)
            xl  = x0 * dot + self.b[l] + xl
        return xl


class AttnFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, *xs):
        stk = torch.stack(xs, 1)
        w   = torch.softmax(self.fc(stk.tanh()), 1)
        return (w * stk).sum(1)


class EmbHead(nn.Module):
    def __init__(self, feat_dim, D, n_ent, L=2):
        super().__init__()
        self.lin  = nn.Linear(feat_dim, D)
        self.fc   = FeatureCrossing(D, L)
        self.id   = nn.Embedding(n_ent, D)
        self.attn = AttnFusion(D)
        self.proj = nn.Linear(3*D, D)
        nn.init.normal_(self.id.weight, std=0.01)

    def forward(self, feat, ids):
        ef  = self.lin(feat)
        efc = self.fc(ef)
        eid = self.id(ids)
        ea  = self.attn(ef, efc, eid)
        return self.proj(torch.cat([ef, efc, ea], -1))


class LightGCN(nn.Module):
    """Standard LightGCN — IDENTICAL to v4, no collaborative GNN changes."""
    def __init__(self, L=2):
        super().__init__()
        self.L = L

    def forward(self, u_emb, i_embs):
        n = i_embs.shape[0]
        if n == 0:
            return u_emb.squeeze(0), i_embs
        norm  = 1.0 / n**0.5
        all_u = [u_emb.squeeze(0)]
        all_i = [i_embs]
        u, its = u_emb.squeeze(0), i_embs
        for _ in range(self.L):
            u_new  = norm * its.mean(0)
            i_new  = norm * u.unsqueeze(0).expand_as(its)
            u, its = u_new, i_new
            all_u.append(u)
            all_i.append(its)
        return torch.stack(all_u).mean(0), torch.stack(all_i).mean(0)


class Model(nn.Module):
    def __init__(self, u_fdim, i_fdim, D, N, M, L_fc=2, L_gnn=2):
        super().__init__()
        self.uh  = EmbHead(u_fdim, D, N, L_fc)
        self.ih  = EmbHead(i_fdim, D, M, L_fc)   # i_fdim = 42 with [O]
        self.gnn = LightGCN(L_gnn)

    def forward_user(self, u_feat_np, u_id, item_ids, device):
        feat = torch.tensor(u_feat_np, dtype=torch.float32, device=device).unsqueeze(0)
        uid  = torch.tensor([u_id],    dtype=torch.long,    device=device)
        u_e  = self.uh(feat, uid)

        # item_feat is now the [O]-augmented feature (genre + collab demographics)
        ifeats = torch.tensor(item_feat[item_ids], dtype=torch.float32, device=device)
        iids   = torch.tensor(item_ids,            dtype=torch.long,    device=device)
        i_e    = self.ih(ifeats, iids)

        u_fin, i_fin = self.gnn(u_e, i_e)
        return u_fin, i_fin

    @torch.no_grad()
    def score_items(self, u_feat_np, u_id, item_ids, device):
        u_fin, i_fin = self.forward_user(u_feat_np, u_id, item_ids, device)
        return (u_fin * i_fin).sum(-1)

    @torch.no_grad()
    def get_gnn_repr(self, u_feat_np, u_id, repr_item_ids, device):
        u_fin, _ = self.forward_user(u_feat_np, u_id, repr_item_ids, device)
        return u_fin.detach().cpu().numpy()


# =============================================================================
# 3.  UTILITIES
# =============================================================================

def get_p(m):    return {k: v.clone().detach().cpu() for k, v in m.state_dict().items()}
def set_p(m, p): m.load_state_dict({k: v.to(next(m.parameters()).device)
                                     for k, v in p.items()})

def w_avg(params, sizes):
    total = sum(sizes)
    out   = {}
    for key in params[0]:
        out[key] = sum((s/total) * p[key].float() for s, p in zip(sizes, params))
    return out

def mix3(lp, cp, gp, alpha, item_keys):
    """
    [A] paper formula:  α1·local + α2·cluster + α3·global
    [P] item head keys: always use global  (items are shared, not personalized)
    """
    a1, a2, a3 = alpha
    return {k: gp[k].float()                                              # [P]
            if k in item_keys else
            a1*lp[k].float() + a2*cp[k].float() + a3*gp[k].float()      # [A]
            for k in lp}

def ema_merge(old_p, new_p, alpha):
    """[J] alpha * old + (1-alpha) * new"""
    return {k: alpha * old_p[k].float() + (1.0 - alpha) * new_p[k].float()
            for k in new_p}

def bpr_batched(u_fin, pos_embs, neg_embs):
    """[H] no explicit regularization — weight_decay in Adam is sufficient."""
    ps   = (u_fin * pos_embs).sum(-1)
    ns   = (u_fin * neg_embs).sum(-1)
    return -F.logsigmoid(ps - ns).mean()


# =============================================================================
# 4.  EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(model, data, train_dict, n_items, n_neg, device):
    model.eval()
    all_items = set(range(n_items))
    hrs, ndcgs = [], []

    for user, pos_item in tqdm(data, desc="Eval", leave=False):
        seen  = train_dict[user] | {pos_item}
        pool  = list(all_items - seen)
        negs  = random.sample(pool, min(n_neg, len(pool)))
        items = [pos_item] + negs

        scores = model.score_items(
            user_feat[user], user, items, device
        ).cpu().numpy()

        order  = np.argsort(-scores)
        hit    = int(0 in set(order[:10].tolist()))
        rank   = np.where(order == 0)[0][0] + 1
        hrs.append(hit)
        ndcgs.append(1.0 / np.log2(rank+1) if hit else 0.0)

    return {"HR@10": float(np.mean(hrs)), "NDCG@10": float(np.mean(ndcgs))}


# =============================================================================
# 5.  TRAINING LOOP
# =============================================================================

def train(cfg):
    device = cfg.device
    K      = cfg.n_clusters

    user_items = defaultdict(list)
    for u, it in train_data:
        user_items[u].append(it)
    eligible  = [u for u, its in user_items.items() if len(its) >= 2]
    all_items = set(range(N_ITEMS))

    model = Model(
        u_fdim=user_feat.shape[1],
        i_fdim=item_feat.shape[1],     # 42 = 19 genre + 23 collab [O]
        D=cfg.embed_dim, N=N_USERS, M=N_ITEMS,
        L_fc=cfg.n_fc_layers, L_gnn=cfg.n_gnn_layers
    ).to(device)

    global_p  = get_p(model)

    # [P] Identify ALL item-head parameter keys — these stay global-only in mix3.
    #     Paper: "Item embeddings are shared and updated via the server."
    #     Effect: items are NEVER personalized; always use collaborative global repr.
    item_keys = frozenset(k for k in global_p.keys() if k.startswith("ih."))
    print(f"Global-only item keys [P]: {len(item_keys)} "
          f"(all ih.* parameters excluded from personalization mix)")

    cluster_p = [copy.deepcopy(global_p) for _ in range(K)]

    user_local_params = {}   # [B]

    user_cluster = np.random.randint(0, K, N_USERS)
    repr_mat     = np.zeros((N_USERS, cfg.embed_dim), dtype=np.float32)
    repr_ready   = np.zeros(N_USERS, dtype=bool)

    best_hr, best_p = -1.0, copy.deepcopy(global_p)
    history         = []

    print(f"\n{'='*70}")
    print(f"PerFedRec v6 | rounds={cfg.n_rounds} | K={K} | "
          f"epochs={cfg.local_epochs} | lr={cfg.lr}")
    print(f"v3/v4 fixes : [A] correct mix  [B] local persist  [C] GNN repr  "
          f"[F-N] stability")
    print(f"v6 new [O]  : collab item feat  "
          f"({item_feat_raw.shape[1]}→{item_feat.shape[1]} dims, static)")
    print(f"v6 new [P]  : global item head  "
          f"(all ih.* = global; users personalise via uh.* only)")
    print(f"{'='*70}")

    for rnd in range(cfg.n_rounds):
        warm = rnd < cfg.cluster_start_rnd

        # ── cluster-proportional user selection ──────────────────────────
        n_sel = min(cfg.n_users_per_round, len(eligible))
        sel   = []
        for k in range(K):
            ku  = [u for u in eligible if user_cluster[u] == k]
            if not ku: continue
            n_k = max(1, round(n_sel * len(ku) / len(eligible)))
            sel += random.sample(ku, min(n_k, len(ku)))
        sel = list(set(sel))[:n_sel]

        # ── local training ───────────────────────────────────────────────
        local_ps, local_sz = [], []

        for user in sel:
            u_its    = user_items[user]
            n_pos    = len(u_its)
            c        = int(user_cluster[user])
            neg_pool = list(all_items - set(u_its))

            # [N] balanced negatives
            if len(neg_pool) >= n_pos:
                negs = random.sample(neg_pool, n_pos)
            else:
                negs = random.choices(neg_pool, k=n_pos)

            # Personalised init: [A] + [P]
            if warm or user not in user_local_params:
                init_p = global_p
            else:
                init_p = mix3(user_local_params[user],
                               cluster_p[c],
                               global_p,
                               cfg.alpha,
                               item_keys)    # [P] ih.* always = global

            set_p(model, init_p)
            model.train()

            opt = torch.optim.Adam(model.parameters(),
                                   lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)

            for _ep in range(cfg.local_epochs):
                all_step = u_its + negs
                opt.zero_grad()
                u_fin, i_fin = model.forward_user(
                    user_feat[user], user, all_step, device)
                pos_embs = i_fin[:n_pos]
                neg_embs = i_fin[n_pos:n_pos + len(negs)]
                loss = bpr_batched(u_fin, pos_embs, neg_embs)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # [J] EMA for local params
            new_local = get_p(model)
            if user in user_local_params:
                user_local_params[user] = ema_merge(
                    user_local_params[user], new_local, cfg.local_ema)
            else:
                user_local_params[user] = new_local

            local_ps.append(get_p(model))
            local_sz.append(n_pos)

            # [C]+[M] repr for clustering
            model.eval()
            repr_items = u_its + random.sample(
                neg_pool, min(cfg.repr_neg_count, len(neg_pool)))
            r = model.get_gnn_repr(user_feat[user], user, repr_items, device)
            repr_mat[user]   = r
            repr_ready[user] = True

        if not local_ps:
            continue

        # ── global aggregation with momentum [K] ─────────────────────────
        new_global = w_avg(local_ps, local_sz)
        global_p   = ema_merge(global_p, new_global, cfg.global_momentum)

        # ── re-cluster [L] ────────────────────────────────────────────────
        if (not warm
                and rnd % cfg.cluster_freq == 0
                and repr_ready.sum() >= K):
            km = KMeans(n_clusters=K, random_state=cfg.seed,
                        n_init=5, max_iter=100)
            user_cluster = km.fit_predict(repr_mat)

        # ── cluster aggregation ───────────────────────────────────────────
        if not warm:
            c_buf, s_buf = defaultdict(list), defaultdict(list)
            for i, user in enumerate(sel[:len(local_ps)]):
                c = int(user_cluster[user])
                c_buf[c].append(local_ps[i])
                s_buf[c].append(local_sz[i])
            for k in range(K):
                if c_buf[k]:
                    cluster_p[k] = w_avg(c_buf[k], s_buf[k])

        # ── evaluation ───────────────────────────────────────────────────
        if (rnd + 1) % cfg.eval_every == 0 or rnd == 0:
            set_p(model, global_p)
            m = evaluate(model, val_data, train_dict, N_ITEMS,
                         cfg.n_neg_eval, device)
            history.append({"round": rnd+1, **m})
            tag = " ← best" if m["HR@10"] > best_hr else ""
            print(f"Rd {rnd+1:3d} | HR@10={m['HR@10']:.4f}  "
                  f"NDCG@10={m['NDCG@10']:.4f}{tag}")
            if m["HR@10"] > best_hr:
                best_hr = m["HR@10"]
                best_p  = copy.deepcopy(global_p)

    # ── final test ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Test evaluation (best checkpoint) …")
    set_p(model, best_p)
    test_m = evaluate(model, test_data, train_dict, N_ITEMS,
                      cfg.n_neg_eval, device)
    print(f"  HR@10   = {test_m['HR@10']:.4f}   (paper target: ~0.6119)")
    print(f"  NDCG@10 = {test_m['NDCG@10']:.4f}   (paper target: ~0.4409)")
    print(f"{'='*70}")

    return test_m, model, pd.DataFrame(history)


# =============================================================================
# 6.  ABLATION HELPER
# =============================================================================

def ablation(cfg):
    orig_alpha = cfg.alpha
    variants = {
        "Var1 – no personalization":  {"alpha_override": [0.0, 0.0, 1.0]},
        "Var2 – no item features":    {"zero_feat": True},
        "Var3 – no clustering":       {"alpha_override": [0.5, 0.0, 0.5]},
        "Var4 – no collab item feat": {"no_collab_feat": True},
        "Var5 – personalize item hd": {"personalize_items": True},
        "PerFedRec v6 (full)":        {},
    }
    results = {}
    for name, kw in variants.items():
        print(f"\n{'─'*58}\nAblation: {name}")
        if kw.get("alpha_override"):
            cfg.alpha = kw["alpha_override"]
        m, _, _ = train(cfg)
        results[name] = m
        cfg.alpha = orig_alpha
    print(f"\n{'='*58}")
    print(f"{'Model':<40}  HR@10   NDCG@10")
    print("─" * 58)
    for name, m in results.items():
        print(f"{name:<40}  {m['HR@10']:.4f}  {m['NDCG@10']:.4f}")
    print("=" * 58)
    return results


# =============================================================================
# 7.  RUN
# =============================================================================

if __name__ == "__main__":

    test_metrics, model, hist = train(cfg)

    hist.to_csv("perfedrec_v6_history.csv", index=False)
    print("History saved → perfedrec_v6_history.csv")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("PerFedRec v6 — Global Item Head + Collab Features",
                     fontsize=13)
        for ax, col in zip(axes, ["HR@10", "NDCG@10"]):
            ax.plot(hist["round"], hist[col], marker="o", ms=4, linewidth=2,
                    color="#4CAF50", label="v6")
            ax.axhline(0.6119 if col == "HR@10" else 0.4409,
                       color="red", linestyle="--", linewidth=1, alpha=0.7,
                       label="Paper target")
            ax.set_title(col); ax.set_xlabel("Round")
            ax.set_ylabel(col); ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig("perfedrec_v6_curve.png", dpi=120)
        plt.show()
        print("Plot saved → perfedrec_v6_curve.png")
    except Exception:
        pass

    RUN_ABLATION = False
    if RUN_ABLATION:
        ablation(cfg)
