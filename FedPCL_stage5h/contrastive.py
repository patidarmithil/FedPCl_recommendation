"""
contrastive.py  FIXED
══════════════════════
BUG FIXED in user_contrastive_loss:

Paper Eq.5:
  L_Con^U = -log( exp(e_u^(l) · e_u^(0)/τ) / Σ_{j∈U} exp(e_u^(l) · e_j^(0)/τ) )

The QUERY is e_u^(l) (even-layer embedding) throughout both numerator AND denominator.
Original code used e_u^(0) as the query for negative similarities — WRONG.

BUG:   neg_sims = e_u^(0) · e_v^(0)   ← wrong query
FIX:   neg_sims = e_u^(l) · e_v^(0)   ← correct per paper Eq.5

Also: denominator in paper Eq.5 sums over ALL j∈U (including positive j=i).
Standard InfoNCE includes the positive in the denominator.
"""

import torch
import torch.nn.functional as F


def user_contrastive_loss(e0_all: torch.Tensor,
                          el_anchor: torch.Tensor,
                          tau: float = 0.2) -> torch.Tensor:
    """
    User-side structural contrastive loss (paper Eq.5) — FIXED.

    Paper Eq.5:
      L = -log( exp(e_u^(l) · e_u^(0)/τ) / Σ_{j∈U} exp(e_u^(l) · e_j^(0)/τ) )

    Query  = e_u^(l)  (even-layer, el_anchor)
    Positive key = e_u^(0)  (layer-0 of same user)
    Negative keys = e_v^(0) for v≠u  (layer-0 of other users)
    Denominator = ALL j including positive (standard InfoNCE)

    Args:
        e0_all:    [N, d]  layer-0 embeddings: e0_all[0]=anchor, e0_all[1:]=neighbours
        el_anchor: [d]     even-layer embedding of anchor user u
        tau:       float   temperature
    """
    N = e0_all.shape[0]
    if N < 3:
        return torch.tensor(0.0, device=e0_all.device, requires_grad=True)

    # L2-normalise
    e0_norm = F.normalize(e0_all, dim=1)              # [N, d]
    el_norm = F.normalize(el_anchor.unsqueeze(0), dim=1).squeeze(0)  # [d]

    # el_norm is the QUERY throughout (paper Eq.5)
    e0_anchor = e0_norm[0]   # [d]  — e_u^(0), the positive key
    negs_e0   = e0_norm[1:]  # [N-1, d] — e_v^(0), the negative keys

    # Positive: e_u^(l) · e_u^(0)
    pos_sim = (el_norm * e0_anchor).sum()                          # scalar

    # Negatives: e_u^(l) · e_v^(0) for v≠u  ← KEY FIX
    neg_sims = (el_norm.unsqueeze(0) * negs_e0).sum(dim=1)        # [N-1]

    # Denominator includes ALL j∈U (positive + negatives) per paper Eq.5
    all_logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / tau  # [N]

    loss = -(pos_sim / tau - torch.logsumexp(all_logits, dim=0))
    return loss


def item_contrastive_loss(E_pos: torch.Tensor,
                          drop_rate: float = 0.3,
                          tau: float = 0.2) -> torch.Tensor:
    """
    Item-side structural contrastive loss (paper Eq.6). Unchanged — was correct.
    """
    n = E_pos.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=E_pos.device, requires_grad=True)

    view1 = F.dropout(E_pos, p=drop_rate, training=True)
    view2 = F.dropout(E_pos, p=drop_rate, training=True)

    v1 = F.normalize(view1, dim=1)
    v2 = F.normalize(view2, dim=1)

    sim_matrix = torch.mm(v1, v2.t()) / tau
    labels = torch.arange(n, device=E_pos.device)
    loss   = F.cross_entropy(sim_matrix, labels)
    return loss


def structural_contrastive_loss(e0_all: torch.Tensor,
                                el_anchor: torch.Tensor,
                                E_pos: torch.Tensor,
                                beta1: float = 0.1,
                                lam: float   = 1.0,
                                tau: float   = 0.2,
                                drop_rate: float = 0.3) -> tuple:
    """
    Full structural contrastive loss (paper Eq.7).
    L_con = β₁ · (L_Con^U + λ · L_Con^V)
    """
    l_user = user_contrastive_loss(e0_all, el_anchor, tau)
    l_item = item_contrastive_loss(E_pos,  drop_rate,  tau)
    total  = beta1 * (l_user + lam * l_item)
    return total, l_user, l_item
