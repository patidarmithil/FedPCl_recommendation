"""
client.py  v2
═════════════
Proper FedAvg client — fixes 3 bugs from v1:

BUG 1 FIXED — E local epochs (not 1 gradient step):
  FedAvg paper Algorithm 1: client runs E full epochs over local data
  before returning to server. E=5 gives 3.5x speedup over E=1 (FedSGD).

BUG 2 FIXED — Return weight DELTA not raw gradient:
  Paper: client returns updated weights w_k
  Server: averages w_k weighted by dataset size
  We send delta = E_local - E_global for each touched item.
  Server averages deltas, applies to global table.

BUG 3 FIXED (consequence of 1+2):
  With E local epochs, embeddings train properly -> scores in 5-15 range.

What client owns (private):
  user_emb  -- personal user embedding [d], never shared
  Adam m,v  -- optimiser state for user_emb

What client receives from server:
  E_global  -- global item embedding table [n_items, d]

What client sends back:
  item_deltas -- dict {item_id: delta [d]} = updated_local - original_global
  n           -- dataset size for weighted FedAvg
"""

import math
import random
import torch
import torch.nn.functional as F


class Client:
    def __init__(self, uid: int, train_items: list,
                 n_items: int, embed_dim: int,
                 device: torch.device):
        self.uid         = uid
        self.train_items = train_items
        self.n_items     = n_items
        self.d           = embed_dim
        self.dev         = device
        self.n           = len(train_items)
        self._train_set  = set(train_items)

        # User embedding -- local only, never shared
        limit = math.sqrt(6.0 / (1 + embed_dim))
        self.user_emb = torch.empty(embed_dim, device=device).uniform_(-limit, limit)

        # Adam state for user_emb
        self.m = torch.zeros(embed_dim, device=device)
        self.v = torch.zeros(embed_dim, device=device)
        self.t = 0

    # ------------------------------------------------------------------
    def local_train(self, E_global: torch.Tensor,
                    n_layers: int,
                    local_epochs: int,
                    lr_item: float,
                    lr_user: float,
                    weight_decay: float) -> tuple:
        """
        FedAvg ClientUpdate: run E local epochs, return weight deltas.

        Per epoch:
          1. Sample 1 negative per positive
          2. Local LightGCN on subgraph
          3. BPR + L2 loss -> backprop
          4. SGD step on local item copy
          5. Adam step on user_emb

        Returns:
            item_deltas: dict {item_id: delta [d]}  (local_final - global_start)
            avg_loss:    float
        """
        if self.n < 2:
            return {}, float('inf')

        # Local copy of this client's item embeddings (the 'w_k' in the paper)
        pos_ids  = self.train_items
        E_local  = E_global[pos_ids].detach().clone()   # [n_pos, d]

        total_loss = 0.0

        for _ in range(local_epochs):
            # Sample negatives
            neg_ids = []
            for _ in pos_ids:
                while True:
                    j = random.randint(0, self.n_items - 1)
                    if j not in self._train_set:
                        neg_ids.append(j)
                        break

            # Leaf tensors for autograd
            E_pos = E_local.clone().requires_grad_(True)                        # [n_pos,d]
            E_neg = E_global[neg_ids].detach().clone().requires_grad_(True)    # [n_neg,d]
            e_u   = self.user_emb.clone().requires_grad_(True)                 # [d]

            # Local LightGCN
            _, _, e_u_agg = self._lightgcn(e_u, E_pos, n_layers)

            # BPR loss
            n_pairs    = min(self.n, len(neg_ids))
            pos_scores = (e_u_agg * E_pos[:n_pairs]).sum(dim=1)
            neg_scores = (e_u_agg * E_neg[:n_pairs]).sum(dim=1)
            loss_bpr   = -F.logsigmoid(pos_scores - neg_scores).mean()

            # L2 reg on layer-0 embeddings
            loss_reg = weight_decay * (
                (e_u   ** 2).sum() +
                (E_pos ** 2).sum() / max(self.n, 1)
            )

            loss = loss_bpr + loss_reg
            loss.backward()
            total_loss += float(loss.detach())

            # SGD step on local item copy (like paper's w <- w - eta * grad)
            if E_pos.grad is not None:
                E_local = (E_local - lr_item * E_pos.grad.detach()).detach()

            # Adam step on user_emb (local only)
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

        # Delta = local_final - global_start (what gets sent to server)
        E_orig  = E_global[pos_ids].detach()
        deltas  = E_local - E_orig          # [n_pos, d]
        item_deltas = {iid: deltas[idx].clone()
                       for idx, iid in enumerate(pos_ids)}

        return item_deltas, total_loss / max(local_epochs, 1)

    # ------------------------------------------------------------------
    def _lightgcn(self, e_u, E_pos, n_layers):
        """LightGCN on local subgraph. Supports autograd."""
        n_pos = E_pos.shape[0]
        if n_pos == 0:
            return [e_u], [E_pos], e_u

        norm = 1.0 / math.sqrt(float(n_pos))
        layers_u, layers_i = [e_u], [E_pos]
        e_uk, E_ik = e_u, E_pos

        for _ in range(n_layers):
            new_eu = norm * E_ik.sum(dim=0)
            new_Ei = (norm * e_uk).unsqueeze(0).expand(n_pos, -1)
            e_uk, E_ik = new_eu, new_Ei
            layers_u.append(e_uk)
            layers_i.append(E_ik)

        e_u_agg = torch.stack(layers_u, dim=0).mean(dim=0)
        return layers_u, layers_i, e_u_agg

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_scores(self, E_global: torch.Tensor, n_layers: int) -> torch.Tensor:
        """Score ALL items for evaluation."""
        if self.n == 0:
            return torch.zeros(self.n_items, device=self.dev)
        E_pos = E_global[self.train_items]
        _, _, e_u_agg = self._lightgcn(self.user_emb, E_pos, n_layers)
        return (e_u_agg.unsqueeze(0) * E_global).sum(dim=1)
