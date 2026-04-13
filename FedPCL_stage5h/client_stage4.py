"""
client_stage4.py
════════════════
Stage 4 client — adds graph expansion + structural contrastive loss.

Changes from Stage 3 (client_stage3.py):
  1. Receives neighbour_users dict from server (2-hop graph expansion)
  2. LightGCN now runs on EXPANDED subgraph with multiple users
  3. Extracts layer-0 AND even-layer embeddings for contrastive loss
  4. Adds L_Con^U (user InfoNCE) + L_Con^V (item InfoNCE) to BPR
  5. Respects warmup: contrastive loss only after round > warmup_rounds

Graph expansion detail:
  Original subgraph (Stage 2-3):  u ↔ {i : i in train_items}
  Expanded subgraph (Stage 4):    u ↔ I_u ↔ U_2hop ↔ I_2hop
  where U_2hop = other users who share ≥1 item with u

  Multiple users in subgraph → enables user-side contrastive negatives.

Even-layer selection for K=2 GNN layers:
  Layer 0: e_u^(0) = initial  [anchor]
  Layer 1: e_u^(1) = agg(items)
  Layer 2: e_u^(2) = agg(users via items)  [positive — even layer]

BPR still uses mean-pool of layers 0,1,2 (unchanged from Stage 3).
Contrastive uses layer-0 as anchor, layer-2 as positive.
"""

import math
import random
import torch
import torch.nn.functional as F

from contrastive import structural_contrastive_loss


class ClientStage4:
    """
    Stage 4 federated client with expanded subgraph + contrastive learning.

    Args:
        uid:             user integer ID
        train_items:     list of item IDs (private)
        neighbour_users: dict {uid_v: [item_ids of v]} for 2-hop neighbours
        n_items:         total items
        embed_dim:       d
        device:          torch device
    """

    def __init__(self, uid: int,
                 train_items: list,
                 neighbour_users: dict,
                 n_items: int,
                 embed_dim: int,
                 device: torch.device):
        self.uid             = uid
        self.train_items     = train_items
        self._train_set      = set(train_items)
        self.neighbour_users = neighbour_users  # {v_uid: [item_ids]}
        self.n_items         = n_items
        self.d               = embed_dim
        self.dev             = device
        self.n               = len(train_items)

        # neighbour uid list (stable order for embedding lookup)
        self.neigh_uids = list(neighbour_users.keys())

        # User embedding (private)
        limit = math.sqrt(6.0 / (1 + embed_dim))
        self.user_emb = torch.empty(embed_dim, device=device).uniform_(-limit, limit)

        # Neighbour user embeddings — RANDOMLY INITIALISED on server side,
        # sent to client as anonymised embeddings for contrastive negatives.
        # In our simulation the server holds all user_emb; here we store
        # a local copy that gets refreshed each round via local_train args.
        self.neigh_embs: dict = {}   # {v_uid: [d] tensor} — filled by server

        # Adam state
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
                    tau:           float  = 0.3,
                    drop_rate:     float  = 0.1) -> tuple:
        """
        Local training with BPR + structural contrastive loss.

        Args:
            E_personal:   [n_items, d]  personalised item embeddings from server
            neigh_embs:   {v_uid: [d]}  neighbour user embeddings (for CL negatives)
            n_layers:     K GNN layers
            local_epochs: E
            lr_item:      SGD lr for item copy
            lr_user:      Adam lr for user_emb
            weight_decay: L2 reg
            use_cl:       False during warmup (first 20 rounds)
            beta1:        contrastive weight β₁=0.1
            lam:          item CL weight λ=1.0
            tau:          temperature τ=0.3
            drop_rate:    item augmentation dropout

        Returns:
            item_deltas, avg_loss, user_emb  (same contract as Stage 3)
        """
        if self.n < 2:
            return {}, float('inf'), self.user_emb.detach().clone()

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

            # ── Build leaf tensors ─────────────────────────────────────────────
            E_pos  = E_local.clone().requires_grad_(True)                       # [n_pos,d]
            E_neg  = E_personal[neg_ids].detach().clone().requires_grad_(True)  # [n_neg,d]
            e_u    = self.user_emb.clone().requires_grad_(True)                 # [d]

            # Neighbour user embeddings (detached — not updated locally)
            neigh_e0 = self._get_neigh_e0()   # [M, d] or None

            # ── Expanded LightGCN — returns all layer outputs ──────────────────
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

            # ── Contrastive loss (skipped during warmup) ───────────────────────
            loss_cl = torch.tensor(0.0, device=self.dev)
            if use_cl and len(layers_u) > 2:
                # Even-layer: with K=2, even layers are 0 and 2
                # anchor = layer-0 of anchor user
                # positive = layer-2 of anchor user
                e_u_l2 = layers_u[-1]   # layer-K = layer-2 (last, which is even for K=2)

                # Stack ALL users' layer-0 embeddings: [anchor + neighbours, d]
                e0_all = self._build_e0_all(e_u, neigh_e0)  # [1+M, d]

                loss_cl, _, _ = structural_contrastive_loss(
                    e0_all    = e0_all,
                    el_anchor = e_u_l2,
                    E_pos     = E_pos,   # NOT detached — gradients must flow to E_local
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

        # ── Item deltas ────────────────────────────────────────────────────────
        E_orig  = E_personal[pos_ids].detach()
        deltas  = E_local - E_orig
        item_deltas = {iid: deltas[idx].clone()
                       for idx, iid in enumerate(pos_ids)}

        return item_deltas, total_loss / max(local_epochs, 1), \
               self.user_emb.detach().clone()

    # ──────────────────────────────────────────────────────────────────────────
    def _get_neigh_e0(self) -> torch.Tensor:
        """
        Stack available neighbour user embeddings into [M, d].
        Returns None if no neighbours available.
        """
        embs = []
        for v in self.neigh_uids:
            if v in self.neigh_embs:
                embs.append(self.neigh_embs[v])
        if not embs:
            return None
        return torch.stack(embs).to(self.dev)   # [M, d]

    # ──────────────────────────────────────────────────────────────────────────
    def _build_e0_all(self, e_u: torch.Tensor,
                      neigh_e0) -> torch.Tensor:
        """
        Build [1+M, d] matrix of layer-0 user embeddings.
        Row 0 = anchor user, rows 1..M = neighbours.
        """
        if neigh_e0 is None or neigh_e0.shape[0] == 0:
            return e_u.unsqueeze(0)   # [1, d] — no negatives available
        return torch.cat([e_u.unsqueeze(0),
                          neigh_e0.detach()], dim=0)   # [1+M, d]

    # ──────────────────────────────────────────────────────────────────────────
    def _lightgcn_expanded(self,
                           e_u:      torch.Tensor,
                           E_pos:    torch.Tensor,
                           neigh_e0, # [M,d] or None — neighbour layer-0 embs
                           n_layers: int) -> tuple:
        """
        LightGCN on expanded subgraph (autograd-compatible).

        Subgraph structure:
          anchor user u ↔ its items  (n_pos nodes)
          neighbour users v ↔ their items  (included in E_pos via shared items)

        For simplicity and efficiency we use the SAME normalisation as Stage 3
        for the BPR path, and compute the extra user-level aggregation only
        for the contrastive positive (layer 2).

        Layer propagation:
          e_u^(k+1) = (1/√n_pos) Σ e_i^(k)      — user aggregates from items
          e_i^(k+1) = (1/√n_pos) e_u^(k)         — items aggregate from user
          e_u^(2)   = (1/√(1+M)) [e_u^(1) + Σ_v e_v^(0) * 1/√n_v]
                      (users at even layer aggregate from their item-side)

        Final BPR embedding: mean of layers 0..K (same as before).
        Contrastive uses layer-0 (anchor) and layer-K (positive).
        """
        n_pos = E_pos.shape[0]
        if n_pos == 0:
            return [e_u], [E_pos], e_u

        norm = 1.0 / math.sqrt(float(n_pos))
        layers_u = [e_u]
        layers_i = [E_pos]
        e_uk, E_ik = e_u, E_pos

        for layer in range(n_layers):
            # Standard bipartite propagation
            new_eu = norm * E_ik.sum(dim=0)
            new_Ei = (norm * e_uk).unsqueeze(0).expand(n_pos, -1)

            # At even layers (layer+1 is even, i.e. layer=1 gives layer-2 output):
            # incorporate neighbour user signals for richer representation.
            if (layer + 1) % 2 == 0 and neigh_e0 is not None and neigh_e0.shape[0] > 0:
                # Each neighbour v contributes 1/√(1+M) * e_v^(0)
                M        = neigh_e0.shape[0]
                u_norm   = 1.0 / math.sqrt(float(1 + M))
                neigh_contrib = u_norm * neigh_e0.detach().mean(dim=0)
                new_eu   = u_norm * new_eu + neigh_contrib

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
