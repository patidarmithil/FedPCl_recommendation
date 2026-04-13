"""
client_stage5.py
════════════════
Stage 5 client — adds Local Differential Privacy (LDP) on top of Stage 4.

Changes from Stage 4 (client_stage4.py):
  - Inherits everything: BPR + contrastive loss + expanded LightGCN
  - Overrides local_train() to apply LDP AFTER computing deltas/user_emb
    but BEFORE returning them to the server

LDP mechanism (paper Section III-D3, Eq.9):
  g̃ = clip(g, σ) + Laplacian(0, λ)

  Step 1 — Per-coordinate clipping:
    clip(g, σ) = torch.clamp(g, -σ, σ)
    Bounds each coordinate so sensitivity is controlled.

  Step 2 — Laplacian noise:
    noise ~ Lap(0, λ)  sampled independently per coordinate
    Satisfies ε-LDP with ε = σ / λ  (smaller λ = stronger privacy, lower ε)

What gets LDP-protected:
  item_deltas  — dict {item_id: delta [d]}   sent to server for FedAvg
  user_emb     — [d] tensor                  sent to server for K-means

What stays private (never leaves client, no LDP needed):
  Adam m, v states
  Gradient w.r.t. user_emb (only the embedding update is kept locally)

Privacy-utility tradeoff:
  σ=0.1, λ=0.001  → strong privacy, small utility hit
  σ=1.0, λ=0.01   → weaker privacy, less utility hit
  σ=inf, λ=0      → no privacy (= Stage 4)
"""

import torch
from client_stage4 import ClientStage4


# ══════════════════════════════════════════════════════════════════════════════
#  LDP MECHANISM   (paper Eq.9)
# ══════════════════════════════════════════════════════════════════════════════

def apply_ldp(grad: torch.Tensor,
              clip_sigma: float,
              lambda_laplace: float) -> torch.Tensor:
    """
    Apply Local Differential Privacy to a gradient / delta tensor.

    g̃ = clip(g, σ) + Laplace(0, λ)

    Args:
        grad:           gradient or delta tensor (any shape)
        clip_sigma:     σ  — per-coordinate clipping bound
        lambda_laplace: λ  — Laplacian noise scale

    Returns:
        LDP-protected tensor (same shape as input)
    """
    # Step 1: per-coordinate clipping
    g_clipped = torch.clamp(grad, -clip_sigma, clip_sigma)

    # Step 2: Laplacian noise
    noise = torch.distributions.Laplace(
        loc=torch.zeros_like(g_clipped),
        scale=lambda_laplace
    ).sample()

    return g_clipped + noise


def apply_ldp_to_deltas(item_deltas: dict,
                         clip_sigma: float,
                         lambda_laplace: float) -> dict:
    """Apply LDP independently to each item's delta vector."""
    return {
        iid: apply_ldp(delta, clip_sigma, lambda_laplace)
        for iid, delta in item_deltas.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class ClientStage5(ClientStage4):
    """
    Stage 5 federated client: Stage 4 + Local Differential Privacy.

    Inherits ALL Stage 4 logic unchanged:
      - Expanded LightGCN subgraph
      - BPR loss
      - Structural contrastive loss (L_Con^U + λ·L_Con^V)
      - Adam optimiser for user_emb

    Adds:
      - LDP on item_deltas before upload (clip + Laplacian noise)
      - LDP on user_emb before upload for K-means clustering

    Extra args vs Stage 4:
        use_ldp:        bool    toggle LDP on/off (False = Stage 4 behaviour)
        clip_sigma:     float   σ, per-coordinate clipping bound
        lambda_laplace: float   λ, Laplacian noise scale
    """

    def local_train(self,
                    E_personal:     torch.Tensor,
                    neigh_embs:     dict,
                    n_layers:       int,
                    local_epochs:   int,
                    lr_item:        float,
                    lr_user:        float,
                    weight_decay:   float,
                    use_cl:         bool  = True,
                    beta1:          float = 0.1,
                    lam:            float = 1.0,
                    tau:            float = 0.3,
                    drop_rate:      float = 0.1,
                    # ── LDP args (new in Stage 5) ─────────────────────────────
                    use_ldp:        bool  = True,
                    clip_sigma:     float = 0.1,
                    lambda_laplace: float = 0.001) -> tuple:
        """
        Stage 4 local training + LDP applied to outgoing tensors.

        Calls super().local_train() to get raw deltas, then protects them.

        Returns:
            item_deltas:   {item_id: LDP-delta [d]}
            avg_loss:      float
            user_emb_ldp:  [d] LDP-protected user embedding for clustering
        """
        # ── Run full Stage 4 training ─────────────────────────────────────────
        item_deltas, avg_loss, user_emb = super().local_train(
            E_personal   = E_personal,
            neigh_embs   = neigh_embs,
            n_layers     = n_layers,
            local_epochs = local_epochs,
            lr_item      = lr_item,
            lr_user      = lr_user,
            weight_decay = weight_decay,
            use_cl       = use_cl,
            beta1        = beta1,
            lam          = lam,
            tau          = tau,
            drop_rate    = drop_rate,
        )

        if not use_ldp:
            # LDP disabled — identical to Stage 4
            return item_deltas, avg_loss, user_emb

        # ── Apply LDP to item deltas ──────────────────────────────────────────
        item_deltas_ldp = apply_ldp_to_deltas(
            item_deltas, clip_sigma, lambda_laplace
        )

        # ── Apply LDP to user embedding (sent for K-means) ───────────────────
        user_emb_ldp = apply_ldp(user_emb, clip_sigma, lambda_laplace)

        return item_deltas_ldp, avg_loss, user_emb_ldp
