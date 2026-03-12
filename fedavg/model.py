"""
model.py
════════
LightGCN implementation faithful to He et al. (2020).
"LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"

Architecture:
  • Embedding layer:  E_u ∈ R^{N×d},  E_i ∈ R^{M×d}
  • Propagation:      K rounds of message passing on bipartite user-item graph
  • Aggregation:      e_u_final = (1/K+1) * sum(e_u^0 + e_u^1 + ... + e_u^K)
  • Prediction:       r̂_ui = e_u_final · e_i_final  (inner product)
  • Loss:             BPR (Bayesian Personalized Ranking)

Key design choices from paper:
  • NO feature transformation (no weight matrices)
  • NO non-linear activation
  • Symmetric normalisation: 1/sqrt(deg_u * deg_i) per edge
  • Mean pool across all K+1 layers as final embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LightGCN(nn.Module):
    """
    Centralized LightGCN for recommendation.

    Args:
        n_users:      total number of users
        n_items:      total number of items
        embed_dim:    embedding dimension d  (paper: 64)
        n_layers:     number of GNN propagation layers K  (paper: 2)
        edge_index:   [2, 2E] LongTensor — bidirectional edges
                      rows[0] = source nodes, rows[1] = target nodes
                      Users indexed 0..N-1, Items indexed N..N+M-1
        edge_weight:  [2E] FloatTensor — LightGCN norm 1/sqrt(du*di)
    """

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int, n_layers: int,
                 edge_index: torch.Tensor,
                 edge_weight: torch.Tensor):
        super().__init__()

        self.n_users    = n_users
        self.n_items    = n_items
        self.n_nodes    = n_users + n_items
        self.embed_dim  = embed_dim
        self.n_layers   = n_layers

        # ── Learnable embeddings  (Eq.1 in LightGCN paper) ───────────────────
        # Xavier uniform init: scale = sqrt(6/(fan_in + fan_out))
        # Initialised as a single matrix [n_users+n_items, d] for efficiency
        self.embedding = nn.Embedding(self.n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # ── Adjacency structure (fixed, not learned) ──────────────────────────
        # Store as sparse tensor for efficient propagation
        self.register_buffer('edge_index',  edge_index)
        self.register_buffer('edge_weight', edge_weight)

        # Pre-build sparse adjacency for fast propagation
        self._build_sparse_adj()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_sparse_adj(self):
        """
        Build sparse adjacency matrix A of shape [n_nodes, n_nodes].
        A[u, N+i] = 1/sqrt(du*di)   (user→item)
        A[N+i, u] = 1/sqrt(du*di)   (item→user, symmetric)
        """
        n = self.n_nodes
        idx = self.edge_index                                # [2, 2E]
        val = self.edge_weight                               # [2E]

        # torch.sparse_coo_tensor for efficient sparse-dense multiply
        self.adj = torch.sparse_coo_tensor(
            idx, val, size=(n, n)
        ).coalesce()

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self):
        """
        Full forward pass — propagate through K layers and return
        final embeddings for ALL users and items.

        LightGCN propagation (Eq.7 in He et al. 2020):
            E^(k+1) = A_hat · E^(k)
            A_hat   = D^{-1/2} A D^{-1/2}   (already encoded in edge_weight)

        Final embedding (Eq.8):
            e_u_final = (1/K+1) * sum_{k=0}^{K} e_u^(k)

        Returns:
            user_embs  [n_users, d]  — final user embeddings
            item_embs  [n_items, d]  — final item embeddings
        """
        # Layer-0: raw learned embeddings
        E0 = self.embedding.weight                          # [n_nodes, d]

        # Accumulate all layer representations
        all_embs = [E0]
        E_k = E0

        for _ in range(self.n_layers):
            # Sparse message passing:  E^(k+1) = A_hat * E^(k)
            # Each node aggregates from its normalized neighbours
            E_k = torch.sparse.mm(self.adj, E_k)           # [n_nodes, d]
            all_embs.append(E_k)

        # Mean pool across all K+1 layers  (alpha_k = 1/(K+1))
        # This is the key design choice of LightGCN vs vanilla GCN
        E_final = torch.stack(all_embs, dim=0).mean(dim=0) # [n_nodes, d]

        # Split into user and item embeddings
        user_embs = E_final[:self.n_users]                 # [n_users, d]
        item_embs = E_final[self.n_users:]                 # [n_items, d]

        return user_embs, item_embs

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, user_ids: torch.Tensor,
                item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for (user, item) pairs via inner product.

        Args:
            user_ids  [B]  — user indices
            item_ids  [B]  — item indices

        Returns:
            scores [B] — predicted interaction scores r̂_ui = e_u · e_i
        """
        user_embs, item_embs = self.forward()
        e_u = user_embs[user_ids]     # [B, d]
        e_i = item_embs[item_ids]     # [B, d]
        return (e_u * e_i).sum(dim=1) # [B]

    # ─────────────────────────────────────────────────────────────────────────
    def bpr_loss(self, users: torch.Tensor,
                 pos_items: torch.Tensor,
                 neg_items: torch.Tensor) -> torch.Tensor:
        """
        BPR (Bayesian Personalized Ranking) loss.

        L_BPR = -sum_{(u,i,j) in D} log sigma(r̂_ui - r̂_uj)

        Args:
            users      [B] — user indices
            pos_items  [B] — positive item indices
            neg_items  [B] — negative item indices (randomly sampled)

        Returns:
            scalar loss
        """
        user_embs, item_embs = self.forward()

        e_u   = user_embs[users]      # [B, d]
        e_pos = item_embs[pos_items]  # [B, d]
        e_neg = item_embs[neg_items]  # [B, d]

        pos_scores = (e_u * e_pos).sum(dim=1)  # [B]
        neg_scores = (e_u * e_neg).sum(dim=1)  # [B]

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

    # ─────────────────────────────────────────────────────────────────────────
    def get_user_ratings(self, user_id: int) -> torch.Tensor:
        """
        Get predicted scores for user_id against ALL items.
        Used for top-K recommendation and evaluation.

        Returns: scores [n_items]
        """
        with torch.no_grad():
            user_embs, item_embs = self.forward()
            e_u = user_embs[user_id]                       # [d]
            scores = item_embs @ e_u                       # [n_items]
        return scores

    # ─────────────────────────────────────────────────────────────────────────
    def get_all_ratings_matrix(self) -> torch.Tensor:
        """
        Compute full user-item score matrix.
        Efficient: one matrix multiply instead of N separate calls.

        Returns: R [n_users, n_items]
        """
        with torch.no_grad():
            user_embs, item_embs = self.forward()
            return user_embs @ item_embs.t()               # [n_users, n_items]


# ══════════════════════════════════════════════════════════════════════════════
#  L2 REGULARISATION HELPER
# ══════════════════════════════════════════════════════════════════════════════
def l2_reg_loss(model: LightGCN,
                users: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: torch.Tensor,
                weight: float = 1e-4) -> torch.Tensor:
    """
    L2 regularisation on the INITIAL (layer-0) embeddings only.
    This is the standard practice for LightGCN — only regularise
    the base embeddings, not the propagated ones.

    L_reg = weight * ( ||e_u^(0)||^2 + ||e_i^(0)||^2 + ||e_j^(0)||^2 )
    """
    n_u = model.n_users
    # Initial embeddings for users and items in this batch
    e_u0   = model.embedding.weight[users]
    e_pos0 = model.embedding.weight[n_u + pos_items]
    e_neg0 = model.embedding.weight[n_u + neg_items]

    reg = (e_u0 ** 2).sum() + (e_pos0 ** 2).sum() + (e_neg0 ** 2).sum()
    return weight * reg / users.shape[0]   # normalize by batch size


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
def hit_rate_and_ndcg(scores_all_items: torch.Tensor,
                      test_item: int,
                      neg_items: list,
                      k: int = 10) -> tuple:
    """
    Compute HR@K and NDCG@K for one user.

    Protocol (paper Section IV-A3):
      • Candidate set = {test_item} + neg_items  (101 total)
      • Rank test_item among candidates
      • HR@K   = 1 if rank ≤ K else 0
      • NDCG@K = 1/log2(rank+1) if rank ≤ K else 0

    Args:
        scores_all_items  [n_items] — scores for all items
        test_item         int       — held-out positive item
        neg_items         list      — 100 negative item IDs
        k                 int       — cutoff (paper: 10)

    Returns:
        (hr, ndcg)  both floats
    """
    candidates = [test_item] + list(neg_items)          # [101]
    cand_scores = scores_all_items[candidates]           # [101]

    # Rank of test item (index 0): count how many negatives score higher
    test_score = cand_scores[0]
    n_higher   = int((cand_scores[1:] > test_score).sum())
    rank       = n_higher + 1   # 1-indexed

    hr   = 1.0 if rank <= k else 0.0
    ndcg = (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
    return hr, ndcg


@torch.no_grad()
def evaluate_model(model: LightGCN,
                   test_dict: dict,
                   neg_dict: dict,
                   k: int = 10) -> dict:
    """
    Full evaluation across all test users.
    Computes HR@K and NDCG@K.

    Args:
        model:     LightGCN (in eval mode)
        test_dict: {user_id: test_item_id}
        neg_dict:  {user_id: [100 negative item ids]}
        k:         cutoff

    Returns:
        {'HR@K': float, 'NDCG@K': float, 'n_users': int}
    """
    model.eval()

    # Compute full rating matrix once — much faster than per-user calls
    R = model.get_all_ratings_matrix()    # [n_users, n_items]

    total_hr, total_ndcg, n = 0.0, 0.0, 0

    for uid, test_item in test_dict.items():
        if uid not in neg_dict:
            continue
        scores = R[uid]     # [n_items]
        hr, ndcg = hit_rate_and_ndcg(scores, test_item, neg_dict[uid], k=k)
        total_hr   += hr
        total_ndcg += ndcg
        n += 1

    return {
        f'HR@{k}':   total_hr   / max(n, 1),
        f'NDCG@{k}': total_ndcg / max(n, 1),
        'n_users':   n
    }


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import load_dataset, build_edge_index

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build tiny synthetic dataset for sanity check
    print("\n── Synthetic sanity check ──")
    n_u, n_i, d = 100, 200, 64

    # Random edge index
    row = list(range(n_u)) * 5
    col = [n_u + (i * 3 % n_i) for i in row]
    rows_all = row + col
    cols_all = col + row
    weights  = [0.3] * len(rows_all)
    ei = torch.tensor([rows_all, cols_all], dtype=torch.long, device=device)
    ew = torch.tensor(weights, dtype=torch.float32, device=device)

    model = LightGCN(n_u, n_i, d, n_layers=2,
                     edge_index=ei, edge_weight=ew).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    u_embs, i_embs = model.forward()
    print(f"user_embs: {u_embs.shape}   item_embs: {i_embs.shape}")

    # BPR loss
    users    = torch.randint(0, n_u, (32,), device=device)
    pos_i    = torch.randint(0, n_i, (32,), device=device)
    neg_i    = torch.randint(0, n_i, (32,), device=device)
    loss     = model.bpr_loss(users, pos_i, neg_i)
    reg      = l2_reg_loss(model, users, pos_i, neg_i)
    print(f"BPR loss: {loss.item():.4f}   L2 reg: {reg.item():.6f}")
    assert abs(loss.item() - math.log(2)) < 0.1, "BPR loss should be ~0.693 initially"

    # Evaluation metrics
    test_dict = {u: (u*3) % n_i for u in range(20)}
    neg_dict  = {u: list(range(5, 105)) for u in range(20)}
    metrics   = evaluate_model(model, test_dict, neg_dict, k=10)
    print(f"Metrics: {metrics}")
    print("\nSelf-test PASSED")
