"""
server_stage3.py
════════════════
Stage 3 server — adds K-means clustering + per-cluster item tables.

New state vs Stage 2:
  E_clusters      list of K tensors [n_items, d] — one per cluster
  assignments     dict {uid: cluster_id}
  centroids       [K, d] tensor — K-means cluster centres (user-space)

New methods:
  get_personal_embeddings(uid)    → μ1*E_cluster[k] + μ2*E_global
  update_clusters(user_embs)      → run K-means, update assignments
  aggregate_stage3(...)           → update both E_global and E_clusters[k]

Aggregation logic (paper Eq. 10–13):
  E_global[i]    += weighted_avg(deltas from ALL contributing clients)
  E_clusters[k][i]+= weighted_avg(deltas from cluster-k clients only)
"""

import math
import random
from collections import defaultdict
import torch
import numpy as np
from sklearn.cluster import KMeans


class ServerStage3:
    """
    Stage 3 server with K-means clustering and per-cluster item embeddings.

    Args:
        n_items:         total items
        embed_dim:       d
        n_clusters:      K = 5 (paper)
        mu1:             cluster model weight (paper: 0.5)
        mu2:             global model weight  (paper: 0.5)
        device:          torch device
    """

    def __init__(self, n_items: int, embed_dim: int,
                 n_clusters: int = 5,
                 mu1: float = 0.5, mu2: float = 0.5,
                 device: torch.device = None):
        self.n_items    = n_items
        self.d          = embed_dim
        self.K          = n_clusters
        self.mu1        = mu1     # cluster model weight
        self.mu2        = mu2     # global model weight
        self.dev        = device or torch.device('cpu')
        self.round      = 0

        limit = math.sqrt(6.0 / (1 + embed_dim))

        # ── Global item embeddings (same as Stage 2) ──────────────────────────
        self.E_global = torch.empty(
            n_items, embed_dim, device=self.dev
        ).uniform_(-limit, limit)

        # ── Per-cluster item embeddings (K tables, same init as global) ────────
        # Each cluster starts as a copy of E_global so personalisation
        # begins from the same point as the global model.
        self.E_clusters = [
            self.E_global.clone() for _ in range(n_clusters)
        ]

        # ── Cluster assignments {uid: cluster_id} ─────────────────────────────
        self.assignments: dict = {}     # populated after first K-means run
        self.centroids          = None  # [K, d] numpy array

    # ──────────────────────────────────────────────────────────────────────────
    def select_clients(self, all_ids: list, n_clients: int) -> list:
        n = min(n_clients, len(all_ids))
        return random.sample(all_ids, n)

    # ──────────────────────────────────────────────────────────────────────────
    def get_personal_embeddings(self, uid: int) -> torch.Tensor:
        """
        Build personalised item table for user uid.

        Formula (paper Eq. 8):
          E_personal = μ1 * E_cluster[k] + μ2 * E_global
          where k = assignments[uid]

        If user has no cluster assignment yet (first rounds before first
        K-means run), fall back to E_global.
        """
        if uid not in self.assignments:
            return self.E_global.detach()

        k     = self.assignments[uid]
        E_p   = self.mu1 * self.E_clusters[k] + self.mu2 * self.E_global
        return E_p.detach()

    # ──────────────────────────────────────────────────────────────────────────
    def update_clusters(self, uid_emb_pairs: list) -> dict:
        """
        Run K-means on collected user embeddings → update assignments.

        Called every cluster_every rounds with embeddings from selected clients.

        Algorithm:
          1. Stack user embeddings into matrix [M, d]
          2. Run KMeans(K) → get labels + centroids
          3. Assign selected clients to their new clusters
          4. Assign ALL other clients to nearest centroid
             (they weren't selected this round but still need an assignment)

        Args:
            uid_emb_pairs: [(uid, user_emb_tensor), ...]

        Returns:
            stats dict: cluster sizes after assignment
        """
        if len(uid_emb_pairs) < self.K:
            return {}   # not enough users to cluster yet

        uids = [p[0] for p in uid_emb_pairs]
        embs = torch.stack([p[1] for p in uid_emb_pairs]).cpu().numpy()

        # K-means with k-means++ init for stability
        km = KMeans(n_clusters=self.K, init='k-means++',
                    n_init=3, random_state=42, max_iter=100)
        km.fit(embs)

        self.centroids = km.cluster_centers_   # [K, d] numpy

        # Assign selected clients
        for uid, label in zip(uids, km.labels_):
            self.assignments[uid] = int(label)

        # Assign unselected clients to nearest centroid
        all_unassigned = [uid for uid in self.assignments
                          if uid not in set(uids)]
        # Also newly seen users that were never selected
        # (they stay at default until selected)

        # Cluster size stats
        from collections import Counter
        counts = Counter(self.assignments.values())
        return {f'cluster_{k}': counts.get(k, 0) for k in range(self.K)}

    # ──────────────────────────────────────────────────────────────────────────
    def aggregate(self, selected_ids: list,
                  delta_list: list,
                  sizes: dict) -> dict:
        """
        FedAvg aggregation — updates BOTH E_global and E_clusters.

        For E_global:
          E_global[i] += weighted_avg(delta from ALL contributing clients)

        For E_clusters[k]:
          E_clusters[k][i] += weighted_avg(delta from cluster-k clients only)

        Args:
            selected_ids: client IDs this round
            delta_list:   [{item_id: delta_tensor}] per client
            sizes:        {uid: dataset_size}

        Returns:
            stats dict
        """
        # ── Accumulators for global ───────────────────────────────────────────
        g_sum  = defaultdict(lambda: torch.zeros(self.d, device=self.dev))
        g_sz   = defaultdict(float)

        # ── Accumulators for each cluster ─────────────────────────────────────
        c_sum  = [defaultdict(lambda: torch.zeros(self.d, device=self.dev))
                  for _ in range(self.K)]
        c_sz   = [defaultdict(float) for _ in range(self.K)]

        for uid, item_deltas in zip(selected_ids, delta_list):
            m_u = float(sizes.get(uid, 1))
            k   = self.assignments.get(uid, None)

            for iid, delta in item_deltas.items():
                # Global accumulator
                g_sum[iid]  += m_u * delta
                g_sz[iid]   += m_u

                # Cluster accumulator (only if assigned)
                if k is not None:
                    c_sum[k][iid] += m_u * delta
                    c_sz[k][iid]  += m_u

        # ── Apply global update ───────────────────────────────────────────────
        n_global = 0
        for iid, d_total in g_sum.items():
            self.E_global[iid] += d_total / g_sz[iid]
            n_global += 1

        # ── Apply per-cluster updates ─────────────────────────────────────────
        n_cluster_updates = 0
        for k in range(self.K):
            for iid, d_total in c_sum[k].items():
                self.E_clusters[k][iid] += d_total / c_sz[k][iid]
                n_cluster_updates += 1

        self.round += 1
        return {
            'n_global_updated':  n_global,
            'n_cluster_updated': n_cluster_updates,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        with torch.no_grad():
            norms = self.E_global.norm(dim=1)
            from collections import Counter
            cluster_sizes = Counter(self.assignments.values())
            return {
                'round':           self.round,
                'emb_norm_mean':   float(norms.mean()),
                'cluster_sizes':   dict(cluster_sizes),
                'n_assigned':      len(self.assignments),
            }
