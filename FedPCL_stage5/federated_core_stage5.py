"""
federated_core_stage5.py
════════════════════════
Stage 5 training loop: Full FedPCL with Local Differential Privacy (LDP).

Changes from Stage 4:
  1. Clients apply LDP (clip + Laplace noise) to ALL uploads:
       - item deltas  (paper Eq. 9, g̃^u_i)
       - user embeddings for clustering
  2. Server tracks cumulative privacy budget ε per round
  3. Optional: robust aggregation (trimmed mean) for Byzantine resilience
  4. Privacy budget is reported in the training log and results

Paper reference (Section III-D-3):
  g̃u = clip(gu, σ) + Laplacian(0, λ)

Privacy parameters (not specified in paper; we use common defaults):
  clip_norm   = 1.0   (σ — L2 clipping norm)
  noise_scale = 0.01  (λ — Laplace scale, gives ε=100 per round, weak LDP)

Tradeoff:
  Larger noise_scale → stronger privacy (smaller ε) → lower accuracy
  Smaller noise_scale → weaker privacy (larger ε)   → higher accuracy
  Set noise_scale=0.001 for near-lossless, noise_scale=0.1 for tighter DP.

Targets (Table I, FedPCL paper):
  FedPCL (full, incl. LDP): HR@10=80.36%, NDCG@10=65.55% on Steam
  LDP adds slight noise → small accuracy drop vs Stage 4 is expected.
"""

import math
import random
import time
import json
import torch

from data_loader import load_dataset, load_item_names
from client_stage5 import ClientStage5
from server_stage5 import ServerStage5
from ldp import privacy_budget


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
    'local_epochs':      5,

    # ── Personalization (same as Stage 3/4) ───────────────────────────────────
    'n_clusters':        5,
    'mu1':               0.5,
    'mu2':               0.5,
    'cluster_every':     10,

    # ── Contrastive learning (same as Stage 4) ─────────────────────────────────
    'beta1':             0.1,
    'lam':               1.0,
    'tau':               0.2,   # paper Fig.3 shows best at τ∈{0.175,0.2}
    'drop_rate':         0.3,   # see stage4 comment — 0.3 correct for d=64
    'warmup_rounds':     20,
    'max_neigh':         20,
    'max_items_neigh':   10,

    # ── LDP (NEW in Stage 5) ──────────────────────────────────────────────────
    # clip_norm   = σ (L2 sensitivity bound for each delta/embedding upload)
    # noise_scale = λ (Laplace noise scale; ε = clip_norm / noise_scale)
    # Set noise_scale=0.001 for near-zero noise (almost no privacy cost).
    # Set noise_scale=0.1   for strong privacy (ε≈10/round, notable accuracy drop).
    'clip_norm':             1.0,
    'noise_scale':           0.01,    # ε = 100 per round (weak LDP, high accuracy)
    'robust_aggregation':    False,   # True → trimmed-mean instead of FedAvg
    'trim_frac':             0.1,     # trimmed-mean: discard top/bottom 10%

    # ── Learning rates ────────────────────────────────────────────────────────
    'lr_item':           0.1,
    'lr_user':           0.001,
    'weight_decay':      1e-6,

    # ── Evaluation ────────────────────────────────────────────────────────────
    'eval_every':        10,
    'top_k':             10,
}

TARGETS = {
    'steam':     {'Stage4_noLDP': (80.36, 65.55),
                  'FedPCL_paper': (80.36, 65.55)},
    'ml100k':    {'Stage4_noLDP': (63.81, 45.03),
                  'FedPCL_paper': (63.81, 45.03)},
    'ml1m':      {'Stage4_noLDP': (62.86, 44.12),
                  'FedPCL_paper': (62.86, 44.12)},
    'filmtrust': {'Stage4_noLDP': (16.81,  8.61),
                  'FedPCL_paper': (16.81,  8.61)},
    'amazon':    {'Stage4_noLDP': (34.04, 22.93),
                  'FedPCL_paper': (34.04, 22.93)},
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
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(clients, server, test_dict, neg_dict, n_gnn_layers, top_k=10):
    """HR@K and NDCG@K using personalised embeddings per user."""
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
def train_stage5(dataset_name, data_path,
                 hparams=None, device=None, verbose=True):
    """
    Stage 5: Full FedPCL with Local Differential Privacy.
    Runs exactly n_rounds rounds (no early stopping).
    """
    if hparams is None:
        hparams = {}
    hp = {**HPARAMS, **hparams}

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bar = "=" * 76
    k   = hp['top_k']

    # Compute privacy budget
    eps_round = privacy_budget(hp['clip_norm'], hp['noise_scale'])
    eps_total = eps_round * hp['n_rounds']

    if verbose:
        print(bar)
        print(f"  Stage 5: Full FedPCL + LDP — {dataset_name.upper()}")
        print(bar)
        print(f"  device={device}  d={hp['embed_dim']}  K_gnn={hp['n_gnn_layers']}")
        print(f"  lr_item={hp['lr_item']}  lr_user={hp['lr_user']}  "
              f"wd={hp['weight_decay']}  local_epochs={hp['local_epochs']}")
        print(f"  rounds={hp['n_rounds']}  clients/round={hp['clients_per_round']}")
        print(f"  clusters K={hp['n_clusters']}  μ1={hp['mu1']}  μ2={hp['mu2']}")
        print(f"  β₁={hp['beta1']}  λ={hp['lam']}  τ={hp['tau']}  "
              f"warmup={hp['warmup_rounds']}  drop={hp['drop_rate']}")
        print(f"  ── LDP ──────────────────────────────────────────────────")
        print(f"  clip_norm σ={hp['clip_norm']}  "
              f"noise_scale λ={hp['noise_scale']}  "
              f"ε/round={eps_round:.2f}  ε_total={eps_total:.1f}")
        if hp['robust_aggregation']:
            print(f"  robust_agg=trimmed_mean  trim_frac={hp['trim_frac']}")
        tgts = TARGETS.get(dataset_name, {})
        if 'FedPCL_paper' in tgts:
            fp = tgts['FedPCL_paper']
            print(f"  FedPCL target (no LDP): HR@10={fp[0]:.2f}%  "
                  f"NDCG@10={fp[1]:.2f}%")
        print(bar)

    bundle = load_dataset(dataset_name, data_path)

    # ── Server ────────────────────────────────────────────────────────────────
    server = ServerStage5(
        n_items             = bundle.n_items,
        embed_dim           = hp['embed_dim'],
        train_dict          = bundle.train_dict,
        n_clusters          = hp['n_clusters'],
        mu1                 = hp['mu1'],
        mu2                 = hp['mu2'],
        max_neigh           = hp['max_neigh'],
        max_items_neigh     = hp['max_items_neigh'],
        clip_norm           = hp['clip_norm'],
        noise_scale         = hp['noise_scale'],
        robust_aggregation  = hp['robust_aggregation'],
        trim_frac           = hp['trim_frac'],
        device              = device,
    )

    # ── Clients ───────────────────────────────────────────────────────────────
    clients = {}
    sizes   = {}
    if verbose:
        print("  Building clients + computing 2-hop neighbourhoods...")

    for uid, items in bundle.train_dict.items():
        if len(items) < 1:
            continue
        neighbours = server.get_neighbours(uid)
        clients[uid] = ClientStage5(
            uid             = uid,
            train_items     = items,
            neighbour_users = neighbours,
            n_items         = bundle.n_items,
            embed_dim       = hp['embed_dim'],
            device          = device,
            clip_norm       = hp['clip_norm'],
            noise_scale     = hp['noise_scale'],
        )
        sizes[uid] = len(items)

    all_ids = list(clients.keys())

    # ── Initial clustering (same as Stage 3/4) ────────────────────────────────
    # Initial user_embs are NOT yet LDP-protected (no gradient yet — purely random)
    # so privacy cost is negligible here; the first round's upload IS protected.
    all_uid_embs = [(uid, c.user_emb.detach().clone())
                    for uid, c in clients.items()]
    server.update_clusters(all_uid_embs)

    if verbose:
        avg_neigh = sum(len(c.neigh_uids) for c in clients.values()) / max(len(clients), 1)
        print(f"  Clients: {len(clients)}  |  Avg neighbours: {avg_neigh:.1f}")
        print(f"  Privacy: ε/round={eps_round:.2f}  "
              f"ε after {hp['n_rounds']} rounds = {eps_total:.1f}")
        print(bar)
        print(f"\n  {'Round':>6} | {'Loss':>8} | "
              f"{'HR@10':>7} | {'NDCG@10':>8} | {'CL':>5} | "
              f"{'ε_cum':>8} | {'Time':>6}")
        print(f"  {'-'*66}")

    best_hr, best_ndcg, best_rnd = 0.0, 0.0, 0
    log = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for rnd in range(1, hp['n_rounds'] + 1):
        t0     = time.time()
        use_cl = (rnd > hp['warmup_rounds'])

        sel_ids    = server.select_clients(all_ids, hp['clients_per_round'])
        delta_list = []
        losses     = []
        uid_embs   = []   # LDP-protected user embeddings for clustering

        for uid in sel_ids:
            E_personal = server.get_personal_embeddings(uid)

            # Neighbour embeddings for CL negatives
            neigh_embs = server.get_neigh_embs(uid, clients) if use_cl else {}

            # local_train returns LDP-protected deltas and user_emb
            item_deltas_ldp, loss, user_emb_ldp = clients[uid].local_train(
                E_personal   = E_personal,
                neigh_embs   = neigh_embs,
                n_layers     = hp['n_gnn_layers'],
                local_epochs = hp['local_epochs'],
                lr_item      = hp['lr_item'],
                lr_user      = hp['lr_user'],
                weight_decay = hp['weight_decay'],
                use_cl       = use_cl,
                beta1        = hp['beta1'],
                lam          = hp['lam'],
                tau          = hp['tau'],
                drop_rate    = hp['drop_rate'],
            )
            delta_list.append(item_deltas_ldp)    # LDP-protected
            uid_embs.append((uid, user_emb_ldp))  # LDP-protected
            if math.isfinite(loss):
                losses.append(loss)

        # Aggregate (standard FedAvg or trimmed mean)
        # server.aggregate also updates privacy accounting
        server.aggregate(sel_ids, delta_list, sizes)

        # Re-cluster using LDP-protected user embeddings
        if rnd % hp['cluster_every'] == 0:
            server.update_clusters(uid_embs)

        avg_loss   = sum(losses) / max(len(losses), 1)
        dt         = time.time() - t0
        eps_so_far = server.cumulative_epsilon

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
                marker   = " ★" if rnd == best_rnd else ""
                cl_label = "ON " if use_cl else "off"
                print(f"  {rnd:>6} | {avg_loss:>8.4f} | "
                      f"{hr*100:>6.2f}% | {ndcg*100:>7.2f}% | "
                      f"{cl_label} | {eps_so_far:>7.1f} | "
                      f"{dt:>4.1f}s{marker}")

            log.append({
                'round':             rnd,
                'loss':              round(avg_loss, 5),
                f'HR@{k}':          round(hr * 100, 3),
                f'NDCG@{k}':        round(ndcg * 100, 3),
                'cl_active':         use_cl,
                'cumulative_epsilon':round(eps_so_far, 2),
            })

    # ── Results ───────────────────────────────────────────────────────────────
    tgts = TARGETS.get(dataset_name, {})
    if verbose:
        print(f"\n{bar}")
        print(f"  RESULT  ({dataset_name.upper()})")
        print(f"  Best HR@{k}:   {best_hr*100:.2f}%")
        print(f"  Best NDCG@{k}: {best_ndcg*100:.2f}%")
        print(f"  Best round: {best_rnd}")
        print(f"  Total ε spent: {server.cumulative_epsilon:.2f}  "
              f"(ε/round={eps_round:.2f}, {hp['n_rounds']} rounds)")
        print()
        for method, (hr_t, ndcg_t) in tgts.items():
            if hr_t == 0:
                continue
            print(f"  vs {method:<14}: "
                  f"HR@10 {best_hr*100-hr_t:+.2f}%  "
                  f"NDCG@10 {best_ndcg*100-ndcg_t:+.2f}%")
        print()
        pstats = server.get_stats()
        print(f"  Privacy: σ={pstats['clip_norm']}  "
              f"λ={pstats['noise_scale']}  "
              f"ε_total={pstats['cumulative_epsilon']:.2f}")
        if hp['robust_aggregation']:
            print(f"  Aggregation: trimmed-mean  (trim_frac={hp['trim_frac']})")
        else:
            print(f"  Aggregation: FedAvg (standard)")
        print(bar)

    # Save log
    log_path = f'stage5_log_{dataset_name}.json'
    with open(log_path, 'w') as f:
        json.dump({
            'dataset':    dataset_name,
            'hparams':    hp,
            'best_hr':    round(best_hr * 100, 3),
            'best_ndcg':  round(best_ndcg * 100, 3),
            'best_round': best_rnd,
            'privacy': {
                'clip_norm':         hp['clip_norm'],
                'noise_scale':       hp['noise_scale'],
                'epsilon_per_round': round(eps_round, 4),
                'total_epsilon':     round(server.cumulative_epsilon, 4),
                'n_rounds':          hp['n_rounds'],
            },
            'log': log,
            'privacy_log': server.get_privacy_log(),
        }, f, indent=2)
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
        'dataset':           dataset_name,
        f'HR@{k}':          round(best_hr * 100, 3),
        f'NDCG@{k}':        round(best_ndcg * 100, 3),
        'best_round':        best_rnd,
        'total_epsilon':     server.cumulative_epsilon,
        'server':            server,
        'clients':           clients,
        'bundle':            bundle,
    }
