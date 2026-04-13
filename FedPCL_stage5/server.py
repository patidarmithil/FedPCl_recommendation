"""
server.py  v2
═════════════
Server for FedAvg — averages weight DELTAS (not gradients).

FedAvg paper Algorithm 1:
  w_{t+1} = Σ_k (n_k / Σn_k) * w_k_{t+1}

In recommendation setting (items only, not full model weights):
  For each item i that appeared in at least one client's update:
    E_global[i] += weighted_avg(delta_k[i])
    where weight = n_k / Σ_{k: i in D_k} n_k

Key difference from v1:
  v1: server did SGD:  E[i] -= lr * avg_gradient     (FedSGD)
  v2: server averages: E[i] += weighted_avg(delta)    (FedAvg)

No server lr needed — the lr was already applied by the client
during its E local SGD steps.
"""

import math
import random
from collections import defaultdict
import torch


class Server:
    """
    Central FedAvg server.

    Args:
        n_items:   total items
        embed_dim: d
        device:    torch device
    """

    def __init__(self, n_items: int, embed_dim: int,
                 device: torch.device):
        self.n_items = n_items
        self.d       = embed_dim
        self.dev     = device
        self.round   = 0

        # Global item embeddings -- shared with all clients each round
        limit = math.sqrt(6.0 / (1 + embed_dim))
        self.E_global = torch.empty(
            n_items, embed_dim, device=device
        ).uniform_(-limit, limit)

    # ------------------------------------------------------------------
    def select_clients(self, all_ids: list, n_clients: int) -> list:
        """Random subset of n_clients from all_ids."""
        n = min(n_clients, len(all_ids))
        return random.sample(all_ids, n)

    # ------------------------------------------------------------------
    def get_global_embeddings(self) -> torch.Tensor:
        """Return detached copy of E_global to broadcast to clients."""
        return self.E_global.detach()

    # ------------------------------------------------------------------
    def aggregate(self, selected_ids: list,
                  delta_list: list,
                  sizes: dict) -> dict:
        """
        FedAvg aggregation of weight deltas.

        For each item i:
          E_global[i] += Σ_k (n_k * delta_k[i]) / Σ_k n_k
          where sum is over contributing clients only (those with item i)

        Args:
            selected_ids: client IDs in this round
            delta_list:   [{item_id: delta_tensor}] one per selected client
            sizes:        {client_id: dataset_size}

        Returns:
            stats dict
        """
        # Accumulate weighted deltas per item
        d_sum  = defaultdict(lambda: torch.zeros(self.d, device=self.dev))
        sz_sum = defaultdict(float)

        for uid, item_deltas in zip(selected_ids, delta_list):
            m_u = float(sizes.get(uid, 1))
            for iid, delta in item_deltas.items():
                d_sum[iid]  += m_u * delta
                sz_sum[iid] += m_u

        # Apply weighted average delta to global embeddings
        n_updated = 0
        for iid, d_total in d_sum.items():
            avg_delta = d_total / sz_sum[iid]
            self.E_global[iid] += avg_delta          # += delta (not -= gradient)
            n_updated += 1

        self.round += 1
        return {'n_items_updated': n_updated}

    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        with torch.no_grad():
            norms = self.E_global.norm(dim=1)
            return {
                'round': self.round,
                'emb_norm_mean': float(norms.mean()),
                'emb_norm_std':  float(norms.std()),
            }
