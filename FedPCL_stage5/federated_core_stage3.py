"""
federated_core_stage3.py
════════════════════════
Stage 3 training loop — FedAvg + Clustering + Personalization.

Changes from Stage 2:
  1. Server builds E_personal = μ1*E_cluster + μ2*E_global per user
  2. Client trains on E_personal (not raw E_global)
  3. Client returns (item_deltas, user_emb) — user_emb used for K-means
  4. Server runs K-means every cluster_every rounds
  5. Server aggregates deltas into BOTH E_global and E_clusters[k]
  6. Evaluation uses E_personal per user (not E_global)

Imports from Stage 2 (unchanged):
  data_loader.py  — load_dataset()
  client_stage3.py — ClientStage3
  server_stage3.py — ServerStage3

Target (Table I, FedPCL paper):
  Steam: HR@10=80.36%  NDCG@10=65.55%  (FedPCL, the full method)
  Intermediate target (PerFedRec): HR@10=76.61% NDCG@10=62.63%
"""

import math
import random
import time
import json
import torch

from data_loader import load_dataset, load_item_names
from client_stage3 import ClientStage3
from server_stage3 import ServerStage3


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
HPARAMS = {
    # ── Architecture ──────────────────────────────────────────────────────────
    'embed_dim':         64,
    'n_gnn_layers':      2,

    # ── Federated training ────────────────────────────────────────────────────
    'n_rounds':          400,
    'clients_per_round': 128,
    'local_epochs':      5,       # E=5 (paper)

    # ── Personalization (paper Section III-C) ─────────────────────────────────
    'n_clusters':        5,       # K=5 clusters (paper)
    'mu1':               0.5,     # cluster model weight (paper)
    'mu2':               0.5,     # global model weight  (paper)
    'cluster_every':     10,      # run K-means every N rounds

    # ── Learning rates ────────────────────────────────────────────────────────
    'lr_item':           0.1,
    'lr_user':           0.001,
    'weight_decay':      1e-6,

    # ── Evaluation ────────────────────────────────────────────────────────────
    'eval_every':        10,
    'top_k':             10,
}

# Targets from Table I — Stage 3 should approach PerFedRec/FPFR/FedPCL
TARGETS = {
    'steam':     {'FedAvg': (71.21, 50.22),
                  'PerFedRec': (76.61, 62.63),
                  'FedPCL':   (80.36, 65.55)},
    'ml100k':    {'FedAvg': (42.70, 23.87),
                  'PerFedRec': (61.87, 43.51),
                  'FedPCL':   (63.81, 45.03)},
    'ml1m':      {'FedAvg': (44.70, 24.90),
                  'PerFedRec': (61.31, 42.83),
                  'FedPCL':   (62.86, 44.12)},
    'filmtrust': {'FedAvg': (10.81,  4.83),
                  'PerFedRec': (15.12,  8.01),
                  'FedPCL':   (16.81,  8.61)},
    'amazon':    {'FedAvg': (26.53, 14.53),
                  'PerFedRec': (32.64, 21.39),
                  'FedPCL':   (34.04, 22.93)},
}

DATASET_PATHS = {
    'steam':     'steam_processed.json',
    'ml100k':    'u.data',
    'ml1m':      'ratings.dat',
    'filmtrust': 'ratings.txt',
    'amazon':    'amazon_electronics.csv',
}

ITEM_NAME_FILES = {
    'steam':  'steam_processed.json',
    'ml100k': 'u.item',
    'ml1m':   'movies.dat',
}


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION — uses personalised embeddings per user
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(clients, server, test_dict, neg_dict,
             n_gnn_layers, top_k=10):
    """
    HR@K and NDCG@K — each user scored with their personalised E_personal.

    Key difference from Stage 2: each user gets their own
    E_personal = μ1*E_cluster[k] + μ2*E_global instead of raw E_global.
    """
    total_hr, total_ndcg, n = 0.0, 0.0, 0

    for uid, test_item in test_dict.items():
        if uid not in clients or uid not in neg_dict:
            continue

        E_personal = server.get_personal_embeddings(uid)
        scores     = clients[uid].get_scores(E_personal, n_gnn_layers)
        candidates = [test_item] + neg_dict[uid]
        cand_sc    = scores[candidates]
        n_higher   = int((cand_sc[1:] > cand_sc[0]).sum())
        rank       = n_higher + 1

        if rank <= top_k:
            total_hr   += 1.0
            total_ndcg += 1.0 / math.log2(rank + 1)
        n += 1

    return {
        f'HR@{top_k}':   total_hr   / max(n, 1),
        f'NDCG@{top_k}': total_ndcg / max(n, 1),
        'n_users': n,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def train_stage3(dataset_name, data_path,
                 hparams=None, device=None, verbose=True):
    """
    Stage 3: FedAvg + K-means Clustering + Personalized Models.
    Runs exactly hp['n_rounds'] rounds (no early stopping).
    """
    if hparams is None:
        hparams = {}
    hp = {**HPARAMS, **hparams}

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bar = "=" * 72
    k   = hp['top_k']

    # ── Load data ─────────────────────────────────────────────────────────────
    if verbose:
        print(bar)
        print(f"  Stage 3: FedAvg + Clustering — {dataset_name.upper()}")
        print(bar)
        print(f"  device={device}  d={hp['embed_dim']}  K_gnn={hp['n_gnn_layers']}")
        print(f"  lr_item={hp['lr_item']}  lr_user={hp['lr_user']}  "
              f"wd={hp['weight_decay']}  local_epochs={hp['local_epochs']}")
        print(f"  rounds={hp['n_rounds']}  clients/round={hp['clients_per_round']}")
        print(f"  clusters K={hp['n_clusters']}  μ1={hp['mu1']}  μ2={hp['mu2']}  "
              f"cluster_every={hp['cluster_every']}")
        tgts = TARGETS.get(dataset_name, {})
        if tgts:
            fa = tgts.get('FedAvg', (0,0))
            fp = tgts.get('FedPCL', (0,0))
            print(f"  FedAvg baseline: HR@10={fa[0]:.2f}%  NDCG@10={fa[1]:.2f}%")
            print(f"  FedPCL target:   HR@10={fp[0]:.2f}%  NDCG@10={fp[1]:.2f}%")
        print(bar)

    bundle = load_dataset(dataset_name, data_path)

    # ── Server ────────────────────────────────────────────────────────────────
    server = ServerStage3(
        n_items    = bundle.n_items,
        embed_dim  = hp['embed_dim'],
        n_clusters = hp['n_clusters'],
        mu1        = hp['mu1'],
        mu2        = hp['mu2'],
        device     = device,
    )

    # ── Clients ───────────────────────────────────────────────────────────────
    clients = {}
    sizes   = {}
    for uid, items in bundle.train_dict.items():
        if len(items) < 1:
            continue
        clients[uid] = ClientStage3(
            uid         = uid,
            train_items = items,
            n_items     = bundle.n_items,
            embed_dim   = hp['embed_dim'],
            device      = device,
        )
        sizes[uid] = len(items)

    all_ids = list(clients.keys())

    if verbose:
        print(f"  Clients: {len(clients)}  |  Cluster every {hp['cluster_every']} rounds")
        print(bar)
        print(f"\n  {'Round':>6} | {'Loss':>8} | "
              f"{'HR@10':>7} | {'NDCG@10':>8} | "
              f"{'Clusters':>9} | {'Time':>6}")
        print(f"  {'-'*62}")

    # ── Collect ALL user embeddings once at start (for initial clustering) ─────
    # Before round 1, initialise cluster assignments with initial user embeddings
    all_uid_embs = [(uid, c.user_emb.detach().clone())
                    for uid, c in clients.items()]
    server.update_clusters(all_uid_embs)

    best_hr   = 0.0
    best_ndcg = 0.0
    best_rnd  = 0
    log       = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for rnd in range(1, hp['n_rounds'] + 1):
        t0 = time.time()

        # Step 1: select clients
        sel_ids = server.select_clients(all_ids, hp['clients_per_round'])

        # Step 2: each client trains on personalised embeddings
        delta_list = []
        losses     = []
        uid_embs   = []   # collect for clustering

        for uid in sel_ids:
            E_personal = server.get_personal_embeddings(uid)

            item_deltas, loss, user_emb = clients[uid].local_train(
                E_personal   = E_personal,
                n_layers     = hp['n_gnn_layers'],
                local_epochs = hp['local_epochs'],
                lr_item      = hp['lr_item'],
                lr_user      = hp['lr_user'],
                weight_decay = hp['weight_decay'],
            )
            delta_list.append(item_deltas)
            uid_embs.append((uid, user_emb))
            if math.isfinite(loss):
                losses.append(loss)

        # Step 3: FedAvg aggregation (global + per-cluster)
        stats = server.aggregate(sel_ids, delta_list, sizes)

        # Step 4: re-cluster every cluster_every rounds
        cluster_info = ""
        if rnd % hp['cluster_every'] == 0:
            cluster_stats = server.update_clusters(uid_embs)
            if cluster_stats and verbose:
                sizes_str = " ".join(
                    f"C{k}:{v}" for k, v in sorted(cluster_stats.items())
                )
                cluster_info = f" [{sizes_str}]"

        avg_loss = sum(losses) / max(len(losses), 1)
        dt       = time.time() - t0

        # Step 5: evaluate periodically
        if rnd % hp['eval_every'] == 0 or rnd == 1 or rnd == hp['n_rounds']:
            metrics = evaluate(
                clients      = clients,
                server       = server,
                test_dict    = bundle.test_dict,
                neg_dict     = bundle.neg_dict,
                n_gnn_layers = hp['n_gnn_layers'],
                top_k        = k,
            )
            hr   = metrics[f'HR@{k}']
            ndcg = metrics[f'NDCG@{k}']

            if hr > best_hr:
                best_hr, best_ndcg, best_rnd = hr, ndcg, rnd

            if verbose:
                marker = " ★" if rnd == best_rnd else ""
                n_assigned = len(server.assignments)
                print(f"  {rnd:>6} | {avg_loss:>8.4f} | "
                      f"{hr*100:>6.2f}% | {ndcg*100:>7.2f}% | "
                      f"{n_assigned:>8} | "
                      f"{dt:>4.1f}s{marker}{cluster_info}")

            log.append({
                'round':      rnd,
                'loss':       round(avg_loss, 5),
                f'HR@{k}':   round(hr * 100, 3),
                f'NDCG@{k}': round(ndcg * 100, 3),
            })

    # ── Results ───────────────────────────────────────────────────────────────
    tgts = TARGETS.get(dataset_name, {})
    if verbose:
        print(f"\n{bar}")
        print(f"  RESULT  ({dataset_name.upper()})")
        print(f"  Best HR@{k}:   {best_hr*100:.2f}%")
        print(f"  Best NDCG@{k}: {best_ndcg*100:.2f}%")
        print(f"  Best round: {best_rnd}")
        print()
        for method, (hr_t, ndcg_t) in tgts.items():
            hr_gap   = best_hr*100   - hr_t
            ndcg_gap = best_ndcg*100 - ndcg_t
            print(f"  vs {method:<10}: "
                  f"HR@10 {hr_gap:+.2f}%  NDCG@10 {ndcg_gap:+.2f}%")

        # Final cluster assignment summary
        stats = server.get_stats()
        if stats['cluster_sizes']:
            print(f"\n  Cluster sizes: {stats['cluster_sizes']}")
        print(bar)

    # Save log
    log_path = f'stage3_log_{dataset_name}.json'
    with open(log_path, 'w') as f:
        json.dump({'dataset': dataset_name, 'hparams': hp,
                   'best_hr': round(best_hr*100, 3),
                   'best_ndcg': round(best_ndcg*100, 3),
                   'best_round': best_rnd, 'log': log}, f, indent=2)
    if verbose:
        print(f"  Log → {log_path}")

    # ── Recommendations for user 0 ────────────────────────────────────────────
    if verbose and 0 in clients:
        import os
        names_file = ITEM_NAME_FILES.get(dataset_name)
        id2name = {}
        if names_file and os.path.exists(names_file):
            id2name = load_item_names(
                dataset_name, names_file,
                item2id=getattr(bundle, '_item2id', None)
            )
        E_personal = server.get_personal_embeddings(0)
        scores     = clients[0].get_scores(E_personal, hp['n_gnn_layers'])
        seen       = set(bundle.train_dict.get(0, []))
        scores[list(seen)] = -1e9
        topk = torch.topk(scores, k).indices.tolist()
        cluster_id = server.assignments.get(0, '?')
        print(f"\n  TOP-10 FOR USER 0  (cluster {cluster_id}):")
        for rank, iid in enumerate(topk, 1):
            name = id2name.get(iid, f'item_{iid}')
            print(f"    {rank:2d}. {str(name)[:55]:<55}  "
                  f"score={float(scores[iid]):.4f}")
        if 0 in bundle.test_dict:
            held = bundle.test_dict[0]
            print(f"\n  Held-out: {id2name.get(held, f'item_{held}')}")

    return {
        'dataset':    dataset_name,
        f'HR@{k}':   round(best_hr * 100, 3),
        f'NDCG@{k}': round(best_ndcg * 100, 3),
        'best_round': best_rnd,
        'server':     server,
        'clients':    clients,
        'bundle':     bundle,
    }
