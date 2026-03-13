"""
client_stage5.py
════════════════
Stage 5 client — adds Local Differential Privacy (LDP) to all uploads.

Changes from Stage 4 (client_stage4.py):
  UPLOAD item_deltas:  apply clip + Laplace noise  (paper Eq. 9)
  UPLOAD user_emb:     apply clip + Laplace noise  (for clustering privacy)

Everything else (local LightGCN, BPR, contrastive loss, Adam) is identical
to Stage 4.

Privacy guarantee (per upload):
  ε = clip_norm / noise_scale    (Laplace mechanism, basic composition)
  δ = 0                          (pure ε-LDP)

Paper Section III-D-3:
  g̃u = clip(gu, σ) + Laplacian(0, λ)
  where gu ∈ {g^u_i, g^u_m}  (item embedding grad + model grad)

In our weight-delta formulation:
  g^u_i → item_deltas[iid]   (weight delta, not raw gradient)
  g^u_m → user_emb           (local private embedding, shared for clustering)

LDP is applied AFTER local training, BEFORE returning to the server.
The local model (user_emb, Adam state) is NOT perturbed — only uploads are.
"""

import math
import random
import torch
import torch.nn.functional as F

from contrastive import structural_contrastive_loss
from ldp import apply_ldp_to_deltas, apply_ldp


class ClientStage5:
    """
    Stage 5 federated client: Stage 4 + LDP on all uploads.

    Args:
        uid:             user integer ID
        train_items:     list of item IDs (private)
        neighbour_users: dict {uid_v: [item_ids]} for 2-hop neighbours
        n_items:         total items
        embed_dim:       d
        device:          torch device
        clip_norm:       σ — LDP clipping threshold  (paper: not specified,
                         default 1.0 is standard in federated DP literature)
        noise_scale:     λ — Laplace noise scale (default 0.01, weak LDP for
                         performance; set 0.1 for stronger privacy)
    """

    def __init__(self, uid: int,
                 train_items: list,
                 neighbour_users: dict,
                 n_items: int,
                 embed_dim: int,
                 device: torch.device,
                 clip_norm:   float = 1.0,
                 noise_scale: float = 0.01):
        self.uid             = uid
        self.train_items     = train_items
        self._train_set      = set(train_items)
        self.neighbour_users = neighbour_users
        self.n_items         = n_items
        self.d               = embed_dim
        self.dev             = device
        self.n               = len(train_items)

        # LDP parameters
        self.clip_norm   = clip_norm
        self.noise_scale = noise_scale

        # neighbour uid list (stable order)
        self.neigh_uids = list(neighbour_users.keys())

        # User embedding (private — NEVER shared raw, only LDP-protected copy)
        limit = math.sqrt(6.0 / (1 + embed_dim))
        self.user_emb = torch.empty(embed_dim, device=device).uniform_(-limit, limit)

        # Neighbour embeddings cache (refreshed each round)
        self.neigh_embs: dict = {}

        # Adam state for user_emb (local only)
        self.m = torch.zeros(embed_dim, device=device)
        self.v = torch.zeros(embed_dim, device=device)
        self.t = 0

    # ──────────────────────────────────────────────────────────────────────────
    def local_train(self,
                    E_personal:    torch.Tensor,
                    neigh_embs:    dict,
                    n_layers:      int,
                    local_epochs:  int,
                    lr_item:       float,
                    lr_user:       float,
                    weight_decay:  float,
                    use_cl:        bool   = True,
                    beta1:         float  = 0.1,
                    lam:           float  = 1.0,
                    tau:           float  = 0.2,
                    drop_rate:     float  = 0.3) -> tuple:
        """
        Local training (identical to Stage 4) + LDP on uploads.

        Returns:
            item_deltas_ldp: dict {item_id: LDP-protected delta [d]}
            avg_loss:        float
            user_emb_ldp:    [d] tensor — LDP-protected user embedding
        """
        if self.n < 2:
            # Even on skip, apply LDP to the zero delta and noisy user_emb
            empty_ldp = {}
            u_ldp     = apply_ldp(self.user_emb, self.clip_norm, self.noise_scale)
            return empty_ldp, float('inf'), u_ldp

        self.neigh_embs = neigh_embs
        pos_ids = self.train_items
        E_local = E_personal[pos_ids].detach().clone()   # [n_pos, d]

        total_loss = 0.0

        for _ in range(local_epochs):
            # ── Sample negatives ───────────────────────────────────────────────
            neg_ids = []
            for _ in pos_ids:
                while True:
                    j = random.randint(0, self.n_items - 1)
                    if j not in self._train_set:
                        neg_ids.append(j)
                        break

            # ── Leaf tensors ───────────────────────────────────────────────────
            E_pos  = E_local.clone().requires_grad_(True)
            E_neg  = E_personal[neg_ids].detach().clone().requires_grad_(True)
            e_u    = self.user_emb.clone().requires_grad_(True)

            neigh_e0 = self._get_neigh_e0()

            # ── Expanded LightGCN ──────────────────────────────────────────────
            layers_u, layers_i, e_u_agg = self._lightgcn_expanded(
                e_u, E_pos, neigh_e0, n_layers
            )

            # ── BPR loss ───────────────────────────────────────────────────────
            n_pairs    = min(self.n, len(neg_ids))
            pos_scores = (e_u_agg * E_pos[:n_pairs]).sum(dim=1)
            neg_scores = (e_u_agg * E_neg[:n_pairs]).sum(dim=1)
            loss_bpr   = -F.logsigmoid(pos_scores - neg_scores).mean()

            # ── L2 reg ─────────────────────────────────────────────────────────
            loss_reg = weight_decay * (
                (e_u   ** 2).sum() +
                (E_pos ** 2).sum() / max(self.n, 1)
            )

            # ── Contrastive loss ───────────────────────────────────────────────
            loss_cl = torch.tensor(0.0, device=self.dev)
            if use_cl and len(layers_u) > 2:
                e_u_l2   = layers_u[-1]
                e0_all   = self._build_e0_all(e_u, neigh_e0)
                loss_cl, _, _ = structural_contrastive_loss(
                    e0_all    = e0_all,
                    el_anchor = e_u_l2,
                    E_pos     = E_pos,
                    beta1     = beta1,
                    lam       = lam,
                    tau       = tau,
                    drop_rate = drop_rate,
                )

            loss = loss_bpr + loss_reg + loss_cl
            loss.backward()
            total_loss += float(loss.detach())

            # ── SGD on local item copy ─────────────────────────────────────────
            if E_pos.grad is not None:
                E_local = (E_local - lr_item * E_pos.grad.detach()).detach()

            # ── Adam on user_emb ───────────────────────────────────────────────
            if e_u.grad is not None:
                g = e_u.grad.detach()
                self.t += 1
                b1, b2, eps = 0.9, 0.999, 1e-8
                self.m = b1 * self.m + (1 - b1) * g
                self.v = b2 * self.v + (1 - b2) * g ** 2
                m_hat  = self.m / (1 - b1 ** self.t)
                v_hat  = self.v / (1 - b2 ** self.t)
                self.user_emb = (
                    self.user_emb - lr_user * m_hat / (v_hat.sqrt() + eps)
                ).detach()

        # ── Compute raw item deltas ────────────────────────────────────────────
        E_orig  = E_personal[pos_ids].detach()
        deltas  = E_local - E_orig
        item_deltas_raw = {iid: deltas[idx].clone()
                           for idx, iid in enumerate(pos_ids)}

        # ── Apply LDP to item deltas (paper Eq. 9) ─────────────────────────────
        item_deltas_ldp = apply_ldp_to_deltas(
            item_deltas_raw,
            clip_norm   = self.clip_norm,
            noise_scale = self.noise_scale,
        )

        # ── Apply LDP to user_emb before sharing for clustering ────────────────
        user_emb_ldp = apply_ldp(
            self.user_emb,
            clip_norm   = self.clip_norm,
            noise_scale = self.noise_scale,
        )

        return item_deltas_ldp, total_loss / max(local_epochs, 1), user_emb_ldp

    # ──────────────────────────────────────────────────────────────────────────
    def _get_neigh_e0(self) -> torch.Tensor:
        embs = [self.neigh_embs[v] for v in self.neigh_uids if v in self.neigh_embs]
        if not embs:
            return None
        return torch.stack(embs).to(self.dev)

    # ──────────────────────────────────────────────────────────────────────────
    def _build_e0_all(self, e_u: torch.Tensor,
                      neigh_e0) -> torch.Tensor:
        if neigh_e0 is None or neigh_e0.shape[0] == 0:
            return e_u.unsqueeze(0)
        return torch.cat([e_u.unsqueeze(0), neigh_e0.detach()], dim=0)

    # ──────────────────────────────────────────────────────────────────────────
    def _lightgcn_expanded(self, e_u, E_pos, neigh_e0, n_layers):
        """Expanded LightGCN — identical to Stage 4."""
        n_pos = E_pos.shape[0]
        if n_pos == 0:
            return [e_u], [E_pos], e_u

        norm = 1.0 / math.sqrt(float(n_pos))
        layers_u = [e_u]
        layers_i = [E_pos]
        e_uk, E_ik = e_u, E_pos

        for layer in range(n_layers):
            new_eu = norm * E_ik.sum(dim=0)
            new_Ei = (norm * e_uk).unsqueeze(0).expand(n_pos, -1)

            if (layer + 1) % 2 == 0 and neigh_e0 is not None and neigh_e0.shape[0] > 0:
                M           = neigh_e0.shape[0]
                u_norm      = 1.0 / math.sqrt(float(1 + M))
                neigh_contrib = u_norm * neigh_e0.detach().mean(dim=0)
                new_eu      = u_norm * new_eu + neigh_contrib

            e_uk, E_ik = new_eu, new_Ei
            layers_u.append(e_uk)
            layers_i.append(E_ik)

        e_u_agg = torch.stack(layers_u, dim=0).mean(dim=0)
        return layers_u, layers_i, e_u_agg

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_scores(self, E_personal: torch.Tensor,
                   n_layers: int) -> torch.Tensor:
        """Score ALL items for evaluation using personalised embeddings."""
        if self.n == 0:
            return torch.zeros(self.n_items, device=self.dev)
        E_pos = E_personal[self.train_items]
        _, _, e_u_agg = self._lightgcn_expanded(
            self.user_emb, E_pos, None, n_layers
        )
        return (e_u_agg.unsqueeze(0) * E_personal).sum(dim=1)
