"""
server_stage5.py
════════════════
Stage 5 server — Stage 4 + privacy accounting + aggregation robustness.

Changes from Stage 4 (server_stage4.py):
  1. Tracks cumulative privacy budget (ε) across rounds
  2. Stores per-round privacy spend in audit log
  3. Optional: median-of-means aggregation as Byzantine-robust alternative
     to standard FedAvg (activated when robust_aggregation=True)
  4. Reports privacy stats in get_stats()

Everything else is inherited from ServerStage4:
  - E_global, E_clusters[K]
  - K-means clustering / assignments
  - get_personal_embeddings()
  - get_neighbours() / get_neigh_embs()
  - aggregate() (global + per-cluster)

Privacy accounting:
  Each round, selected clients upload LDP-protected deltas.
  Per-client ε-spend per round = clip_norm / noise_scale.
  Total ε after T rounds = T * ε_per_round  (basic composition).
  Advanced composition (Kairouz et al. 2015) can reduce this but we
  use basic composition for simplicity and correctness.

Robust aggregation (optional):
  Standard FedAvg is sensitive to outlier deltas (e.g. from Byzantine or
  poorly trained clients). Coordinate-wise trimmed mean discards the top
  and bottom `trim_frac` fraction of values before averaging, giving
  resilience to a bounded fraction of adversarial clients.
"""

import math
import random
from collections import defaultdict
import torch

from server_stage4 import ServerStage4


class ServerStage5(ServerStage4):
    """
    Stage 5 server: Stage 4 + privacy accounting + robust aggregation option.

    Extra args:
        clip_norm:           σ — LDP clipping threshold (must match client)
        noise_scale:         λ — Laplace noise scale (must match client)
        robust_aggregation:  if True, use trimmed-mean instead of FedAvg
        trim_frac:           fraction to trim from each end (default 0.1 = 10%)
    """

    def __init__(self,
                 n_items: int,
                 embed_dim: int,
                 train_dict: dict,
                 n_clusters: int          = 5,
                 mu1: float               = 0.5,
                 mu2: float               = 0.5,
                 max_neigh: int           = 20,
                 max_items_neigh: int     = 10,
                 clip_norm: float         = 1.0,
                 noise_scale: float       = 0.01,
                 robust_aggregation: bool = False,
                 trim_frac: float         = 0.1,
                 device: torch.device     = None):

        super().__init__(
            n_items         = n_items,
            embed_dim       = embed_dim,
            train_dict      = train_dict,
            n_clusters      = n_clusters,
            mu1             = mu1,
            mu2             = mu2,
            max_neigh       = max_neigh,
            max_items_neigh = max_items_neigh,
            device          = device,
        )

        # LDP params (server keeps these for accounting; clients apply the noise)
        self.clip_norm   = clip_norm
        self.noise_scale = noise_scale

        # Robust aggregation
        self.robust_aggregation = robust_aggregation
        self.trim_frac          = trim_frac

        # Privacy accounting
        self._epsilon_per_round = (clip_norm / noise_scale
                                   if noise_scale > 0 else float('inf'))
        self._cumulative_epsilon = 0.0
        self._privacy_log: list  = []   # [{round, epsilon_spent, cumulative}]

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def cumulative_epsilon(self) -> float:
        """Total ε spent so far across all rounds (basic composition)."""
        return self._cumulative_epsilon

    # ──────────────────────────────────────────────────────────────────────────
    def aggregate(self,
                  selected_ids: list,
                  delta_list: list,
                  sizes: dict) -> dict:
        """
        Aggregation with optional robust (trimmed-mean) mode.

        When robust_aggregation=False: standard weighted FedAvg (inherited).
        When robust_aggregation=True:  coordinate-wise trimmed mean,
          ignoring client dataset sizes (uniform weights for robustness).

        Also updates privacy accounting log.
        """
        if self.robust_aggregation:
            stats = self._trimmed_mean_aggregate(selected_ids, delta_list)
        else:
            stats = super().aggregate(selected_ids, delta_list, sizes)

        # Privacy accounting — count only rounds where clients actually uploaded
        n_uploaders = sum(1 for d in delta_list if d)
        if n_uploaders > 0:
            self._cumulative_epsilon += self._epsilon_per_round
            self._privacy_log.append({
                'round':             self.round,
                'epsilon_spent':     round(self._epsilon_per_round, 4),
                'cumulative_epsilon':round(self._cumulative_epsilon, 4),
                'n_uploaders':       n_uploaders,
            })

        return stats

    # ──────────────────────────────────────────────────────────────────────────
    def _trimmed_mean_aggregate(self,
                                selected_ids: list,
                                delta_list: list) -> dict:
        """
        Coordinate-wise trimmed mean aggregation.

        For each item i with k contributing clients:
          1. Stack deltas into [k, d] matrix
          2. Sort each coordinate
          3. Trim `trim_frac` from each end (floor(k * trim_frac) entries)
          4. Average the remaining values

        More robust than FedAvg against outliers / Byzantine clients.
        Uses uniform weights (ignores dataset sizes) since trimmed mean
        is not straightforwardly compatible with weighted averaging.

        Returns:
            stats dict  (same schema as standard aggregate)
        """
        # ── Collect deltas per item ───────────────────────────────────────────
        item_deltas_all: dict = defaultdict(list)  # {iid: [[d], [d], ...]}

        for uid, item_deltas in zip(selected_ids, delta_list):
            for iid, delta in item_deltas.items():
                item_deltas_all[iid].append(delta.to(self.dev))

        # ── Global trimmed-mean update ─────────────────────────────────────────
        n_global_updated  = 0
        n_cluster_updated = 0

        # Also collect per-cluster for cluster tables
        cluster_item_all: list = [defaultdict(list) for _ in range(self.K)]
        uid_to_cluster = {uid: self.assignments.get(uid)
                          for uid in selected_ids}

        for uid, item_deltas in zip(selected_ids, delta_list):
            k = uid_to_cluster.get(uid)
            for iid, delta in item_deltas.items():
                if k is not None:
                    cluster_item_all[k][iid].append(delta.to(self.dev))

        # Apply global trimmed mean
        for iid, deltas in item_deltas_all.items():
            avg_delta = self._trimmed_mean(deltas, self.trim_frac)
            self.E_global[iid] += avg_delta
            n_global_updated   += 1

        # Apply per-cluster trimmed mean
        for k in range(self.K):
            for iid, deltas in cluster_item_all[k].items():
                avg_delta = self._trimmed_mean(deltas, self.trim_frac)
                self.E_clusters[k][iid] += avg_delta
                n_cluster_updated       += 1

        self.round += 1
        return {
            'n_global_updated':  n_global_updated,
            'n_cluster_updated': n_cluster_updated,
            'aggregation':       'trimmed_mean',
        }

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _trimmed_mean(tensors: list, trim_frac: float) -> torch.Tensor:
        """
        Coordinate-wise trimmed mean of a list of same-shape tensors.

        Args:
            tensors:   list of [d] tensors
            trim_frac: fraction to trim from each end

        Returns:
            [d] tensor — trimmed mean
        """
        k = len(tensors)
        if k == 1:
            return tensors[0].clone()

        stacked = torch.stack(tensors, dim=0)   # [k, d]
        n_trim  = max(0, int(math.floor(k * trim_frac)))

        if n_trim == 0 or 2 * n_trim >= k:
            # No trimming possible (too few clients)
            return stacked.mean(dim=0)

        # Sort each coordinate; trim top and bottom n_trim
        sorted_vals, _ = stacked.sort(dim=0)                  # [k, d]
        trimmed        = sorted_vals[n_trim: k - n_trim]      # [k-2t, d]
        return trimmed.mean(dim=0)

    # ──────────────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Extended stats including privacy accounting."""
        base = super().get_stats()
        base.update({
            'cumulative_epsilon':  round(self._cumulative_epsilon, 4),
            'epsilon_per_round':   round(self._epsilon_per_round, 4),
            'clip_norm':           self.clip_norm,
            'noise_scale':         self.noise_scale,
            'robust_aggregation':  self.robust_aggregation,
        })
        return base

    # ──────────────────────────────────────────────────────────────────────────
    def get_privacy_log(self) -> list:
        """Return full privacy accounting log."""
        return self._privacy_log
