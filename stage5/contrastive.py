"""
contrastive.py
══════════════
Pure contrastive loss functions for FedPCL (paper Eq.5 and Eq.6).
No model state — call these from the client during local training.

── User-side (Eq.5) ────────────────────────────────────────────────────
InfoNCE pulling anchor e_u^(0) toward positive e_u^(even_layer),
using other users' layer-0 embeddings as negatives.

L_Con^U = mean over u of:
  -log( exp(cos(e_u^0, e_u^l)/τ) / Σ_{v≠u} exp(cos(e_u^0, e_v^0)/τ) )

── Item-side (Eq.6) ────────────────────────────────────────────────────
InfoNCE on two dropout-augmented views of the same item.

L_Con^V = mean over i of:
  -log( exp(cos(v1_i, v2_i)/τ) / Σ_{j} exp(cos(v1_i, v2_j)/τ) )

── Temperature ─────────────────────────────────────────────────────────
τ = 0.3  (paper Section IV-A)
"""

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
def user_contrastive_loss(e0_all: torch.Tensor,
                          el_anchor: torch.Tensor,
                          tau: float = 0.3) -> torch.Tensor:
    """
    User-side structural contrastive loss (paper Eq.5).

    For the anchor user (index 0 by convention):
      anchor   = e0_all[0]        — layer-0 embedding of user u
      positive = el_anchor        — even-layer embedding of user u (e_u^l)
      negatives= e0_all[1:]       — layer-0 of all other users in subgraph

    Args:
        e0_all:    [N, d]  layer-0 embeddings of ALL users in local subgraph
                           e0_all[0] = anchor user u, e0_all[1:] = neighbours
        el_anchor: [d]     even-layer embedding of anchor user u only
        tau:       float   temperature (paper: 0.3)

    Returns:
        scalar loss tensor
    """
    N = e0_all.shape[0]
    if N < 3:
        # Need at least 3 users (anchor + 2 negatives) for meaningful signal.
        # Return 0 so training continues — BPR still runs.
        return torch.tensor(0.0, device=e0_all.device, requires_grad=True)

    # L2-normalise for cosine similarity
    e0_norm  = F.normalize(e0_all,    dim=1)   # [N, d]
    el_norm  = F.normalize(el_anchor.unsqueeze(0), dim=1)  # [1, d]

    anchor = e0_norm[0]          # [d]
    negs   = e0_norm[1:]         # [N-1, d]

    # Positive similarity: anchor · positive  (cosine, both normalised)
    pos_sim = (anchor * el_norm.squeeze(0)).sum()        # scalar

    # Negative similarities: anchor · each other user's e^(0)
    neg_sims = (anchor.unsqueeze(0) * negs).sum(dim=1)  # [N-1]

    # InfoNCE: log( exp(pos/τ) / Σ exp(neg/τ) )
    # = pos/τ - log( Σ exp(neg/τ) )
    # For numerical stability use logsumexp
    pos_term = pos_sim / tau
    neg_term = torch.logsumexp(neg_sims / tau, dim=0)

    loss = -(pos_term - neg_term)
    return loss


# ══════════════════════════════════════════════════════════════════════════════
def item_contrastive_loss(E_pos: torch.Tensor,
                          drop_rate: float = 0.3,
                          tau: float = 0.3) -> torch.Tensor:
    """
    Item-side structural contrastive loss (paper Eq.6).

    Two independent dropout augmentations of the same item embeddings
    form two views. Each item's view1 should be closer to its own view2
    than to other items' view2.

    Args:
        E_pos:     [n_pos, d]  layer-0 item embeddings (positive items only)
        drop_rate: float       dropout probability for augmentation (paper not
                               specified; we use 0.1 — small to preserve signal)
        tau:       float       temperature (paper: 0.3)

    Returns:
        scalar loss tensor
    """
    n = E_pos.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=E_pos.device, requires_grad=True)

    # Two independent dropout masks → two views of the same embedding matrix
    # training=True forces dropout to actually drop (not identity at eval)
    view1 = F.dropout(E_pos, p=drop_rate, training=True)   # [n, d]
    view2 = F.dropout(E_pos, p=drop_rate, training=True)   # [n, d]

    # L2-normalise
    v1 = F.normalize(view1, dim=1)   # [n, d]
    v2 = F.normalize(view2, dim=1)   # [n, d]

    # Similarity matrix: [n, n]  — v1[i] · v2[j] for all i,j
    sim_matrix = torch.mm(v1, v2.t()) / tau   # [n, n]

    # Diagonal = positive pairs (same item, two views)
    # Off-diagonal = negative pairs (different items)
    # InfoNCE loss = mean of -log( exp(sim_ii) / Σ_j exp(sim_ij) )
    #              = mean of cross_entropy(sim_matrix, identity labels)
    labels = torch.arange(n, device=E_pos.device)
    loss   = F.cross_entropy(sim_matrix, labels)
    return loss


# ══════════════════════════════════════════════════════════════════════════════
def structural_contrastive_loss(e0_all: torch.Tensor,
                                el_anchor: torch.Tensor,
                                E_pos: torch.Tensor,
                                beta1: float = 0.1,
                                lam: float   = 1.0,
                                tau: float   = 0.3,
                                drop_rate: float = 0.3) -> tuple:
    """
    Full structural contrastive loss (paper Eq.7, contrastive terms only).

    L_con = β₁ · (L_Con^U + λ · L_Con^V)

    Args:
        e0_all:    [N, d]  layer-0 embeddings of all users in subgraph
        el_anchor: [d]     even-layer embedding of anchor user
        E_pos:     [n_pos, d] layer-0 item embeddings
        beta1:     float   weight of contrastive loss (paper β₁=0.1)
        lam:       float   item-side weight (paper λ=1.0)
        tau:       float   temperature (paper τ=0.3)
        drop_rate: float   item augmentation dropout rate

    Returns:
        (total_con_loss, l_user, l_item)  — all scalar tensors
    """
    l_user = user_contrastive_loss(e0_all, el_anchor, tau)
    l_item = item_contrastive_loss(E_pos,  drop_rate,  tau)

    total  = beta1 * (l_user + lam * l_item)
    return total, l_user, l_item
