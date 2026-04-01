"""
federated_core_stage5.py
════════════════════════
Stage 5 training loop: Stage 4 + Local Differential Privacy (LDP).

Changes from Stage 4:
  1. ClientStage5 instead of ClientStage4
  2. LDP hyperparameters (use_ldp, clip_sigma, lambda_laplace) passed
     through to client.local_train() every round
  3. Evaluation unchanged (LDP only affects training uploads)
  4. Log includes LDP settings for reproducibility

Privacy-utility analysis printed after training:
  - Performance WITH LDP vs WITHOUT LDP (Stage 4 result)
  - ε = clip_sigma / lambda_laplace  (approximate privacy budget)

Paper hyperparameters (Section IV-A):
  clip_sigma     = 0.1   (σ)
  lambda_laplace = 0.001 (λ)
  → ε = 0.1 / 0.001 = 100  (loose privacy; paper uses small λ)
"""

import math
import time
import json
import torch

from data_loader          import load_dataset, load_item_names
from client_stage5        import ClientStage5
from server_stage5        import ServerStage5


# ══════════════════════════════════════════════════════════════════════════════
HPARAMS = {
    # ── Architecture ──────────────────────────────────────────────────────────
    'embed_dim':         64,
    'n_gnn_layers':      2,

    # ── Federated training ────────────────────────────────────────────────────
    'n_rounds':          400,
    'clients_per_round': 128,
    'local_epochs':      5,

    # ── Personalisation (unchanged from Stage 3/4) ────────────────────────────
    'n_clusters':        5,
    'mu1':               0.5,
    'mu2':               0.5,
    'cluster_every':     10,

    # ── Contrastive learning (unchanged from Stage 4) ─────────────────────────
    'beta1':             0.1,
    'lam':               1.0,
    'tau':               0.3,
    'drop_rate':         0.3,
    'warmup_rounds':     20,
    'max_neigh':         20,
    'max_items_neigh':   10,

    # ── Learning rates ────────────────────────────────────────────────────────
    'lr_item':           0.1,
    'lr_user':           0.001,
    'weight_decay':      1e-6,

    # ── LDP (Stage 5 NEW) ─────────────────────────────────────────────────────
    'use_ldp':           True,
    'clip_sigma':        0.1,    # σ  per-coordinate clipping bound
    'lambda_laplace':    0.001,  # λ  Laplacian noise scale

    # ── Evaluation ────────────────────────────────────────────────────────────
    'eval_every':        10,
    'top_k':             10,
}

# Stage 4 results used as baseline for privacy-utility comparison
STAGE4_RESULTS = {
    'steam':     {'HR@10': 78.84, 'NDCG@10': 55.28},
    'ml100k':    {'HR@10':  0.0,  'NDCG@10':  0.0},
    'ml1m':      {'HR@10':  0.0,  'NDCG@10':  0.0},
    'filmtrust': {'HR@10':  0.0,  'NDCG@10':  0.0},
    'amazon':    {'HR@10':  0.0,  'NDCG@10':  0.0},
}

PAPER_TARGETS = {
    'steam':     {'HR@10': 80.36, 'NDCG@10': 65.55},
    'ml100k':    {'HR@10': 63.81, 'NDCG@10': 45.03},
    'ml1m':      {'HR@10': 62.86, 'NDCG@10': 44.12},
    'filmtrust': {'HR@10': 16.81, 'NDCG@10':  8.61},
    'amazon':    {'HR@10': 34.04, 'NDCG@10': 22.93},
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
def train_stage5(dataset_name, data_path,
                 hparams=None, device=None, verbose=True):
    """
    Stage 5: FedPCL + Local Differential Privacy.
    Runs exactly n_rounds rounds.
    """
    if hparams is None:
        hparams = {}
    hp = {**HPARAMS, **hparams}

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bar = "=" * 72
    k   = hp['top_k']

    # ── Privacy budget ε = σ / λ ──────────────────────────────────────────────
    if hp['use_ldp'] and hp['lambda_laplace'] > 0:
        epsilon = hp['clip_sigma'] / hp['lambda_laplace']
    else:
        epsilon = float('inf')

    if verbose:
        print(bar)
        print(f"  Stage 5: FedPCL + LDP — {dataset_name.upper()}")
        print(bar)
        print(f"  device={device}  d={hp['embed_dim']}  K_gnn={hp['n_gnn_layers']}")
        print(f"  rounds={hp['n_rounds']}  clients/round={hp['clients_per_round']}")
        print(f"  local_epochs={hp['local_epochs']}  lr_item={hp['lr_item']}")
        print(f"  clusters K={hp['n_clusters']}  mu1={hp['mu1']}  mu2={hp['mu2']}")
        print(f"  beta1={hp['beta1']}  lam={hp['lam']}  tau={hp['tau']}")
        print()
        if hp['use_ldp']:
            print(f"  LDP ENABLED:")
            print(f"    clip_sigma     σ = {hp['clip_sigma']}")
            print(f"    lambda_laplace λ = {hp['lambda_laplace']}")
            print(f"    privacy budget ε = σ/λ = {epsilon:.1f}")
            print(f"    (lower ε = stronger privacy; typical range 1–100)")
        else:
            print(f"  LDP DISABLED  (identical to Stage 4)")
        tgt = PAPER_TARGETS.get(dataset_name, {})
        if tgt:
            print(f"\n  FedPCL paper target: "
                  f"HR@10={tgt['HR@10']:.2f}%  NDCG@10={tgt['NDCG@10']:.2f}%")
        s4 = STAGE4_RESULTS.get(dataset_name, {})
        if s4 and s4['HR@10'] > 0:
            print(f"  Stage 4 baseline:    "
                  f"HR@10={s4['HR@10']:.2f}%  NDCG@10={s4['NDCG@10']:.2f}%")
        print(bar)

    bundle = load_dataset(dataset_name, data_path)

    # ── Server (identical to Stage 4) ────────────────────────────────────────
    server = ServerStage5(
        n_items         = bundle.n_items,
        embed_dim       = hp['embed_dim'],
        train_dict      = bundle.train_dict,
        n_clusters      = hp['n_clusters'],
        mu1             = hp['mu1'],
        mu2             = hp['mu2'],
        max_neigh       = hp['max_neigh'],
        max_items_neigh = hp['max_items_neigh'],
        device          = device,
    )

    # ── Clients (Stage 5 — same init as Stage 4) ──────────────────────────────
    clients = {}
    sizes   = {}
    if verbose:
        print("  Building clients + 2-hop neighbourhoods ...")

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
        )
        sizes[uid] = len(items)

    all_ids = list(clients.keys())

    # Initial clustering
    all_uid_embs = [(uid, c.user_emb.detach().clone())
                    for uid, c in clients.items()]
    server.update_clusters(all_uid_embs)

    if verbose:
        avg_neigh = sum(len(c.neigh_uids) for c in clients.values()) / max(len(clients), 1)
        print(f"  Clients: {len(clients)}  |  Avg neighbours: {avg_neigh:.1f}")
        ldp_tag = f"LDP(σ={hp['clip_sigma']},λ={hp['lambda_laplace']})" \
                  if hp['use_ldp'] else "LDP=OFF"
        print(f"  {ldp_tag}")
        print(bar)
        print(f"\n  {'Round':>6} | {'Loss':>8} | "
              f"{'HR@10':>7} | {'NDCG@10':>8} | {'CL':>5} | {'LDP':>5} | {'Time':>6}")
        print(f"  {'-'*62}")

    best_hr, best_ndcg, best_rnd = 0.0, 0.0, 0
    log = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for rnd in range(1, hp['n_rounds'] + 1):
        t0     = time.time()
        use_cl = (rnd > hp['warmup_rounds'])

        sel_ids    = server.select_clients(all_ids, hp['clients_per_round'])
        delta_list = []
        losses     = []
        uid_embs   = []

        for uid in sel_ids:
            E_personal = server.get_personal_embeddings(uid)
            neigh_embs = server.get_neigh_embs(uid, clients) if use_cl else {}

            # ── Stage 5: pass LDP args to local_train ────────────────────────
            item_deltas, loss, user_emb = clients[uid].local_train(
                E_personal     = E_personal,
                neigh_embs     = neigh_embs,
                n_layers       = hp['n_gnn_layers'],
                local_epochs   = hp['local_epochs'],
                lr_item        = hp['lr_item'],
                lr_user        = hp['lr_user'],
                weight_decay   = hp['weight_decay'],
                use_cl         = use_cl,
                beta1          = hp['beta1'],
                lam            = hp['lam'],
                tau            = hp['tau'],
                drop_rate      = hp['drop_rate'],
                use_ldp        = hp['use_ldp'],
                clip_sigma     = hp['clip_sigma'],
                lambda_laplace = hp['lambda_laplace'],
            )
            delta_list.append(item_deltas)
            uid_embs.append((uid, user_emb))
            if math.isfinite(loss):
                losses.append(loss)

        server.aggregate(sel_ids, delta_list, sizes)

        if rnd % hp['cluster_every'] == 0:
            server.update_clusters(uid_embs)

        avg_loss = sum(losses) / max(len(losses), 1)
        dt       = time.time() - t0

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
                ldp_label = "ON " if hp['use_ldp'] else "off"
                print(f"  {rnd:>6} | {avg_loss:>8.4f} | "
                      f"{hr*100:>6.2f}% | {ndcg*100:>7.2f}% | "
                      f"{cl_label} | {ldp_label} | {dt:>4.1f}s{marker}")

            log.append({
                'round':      rnd,
                'loss':       round(avg_loss, 5),
                f'HR@{k}':   round(hr * 100, 3),
                f'NDCG@{k}': round(ndcg * 100, 3),
                'cl_active':  use_cl,
                'ldp_active': hp['use_ldp'],
            })

    # ── Results + privacy-utility analysis ───────────────────────────────────
    tgt = PAPER_TARGETS.get(dataset_name, {})
    s4  = STAGE4_RESULTS.get(dataset_name, {})

    if verbose:
        print(f"\n{bar}")
        print(f"  RESULT  ({dataset_name.upper()})")
        print(f"  Best HR@{k}:   {best_hr*100:.2f}%")
        print(f"  Best NDCG@{k}: {best_ndcg*100:.2f}%")
        print(f"  Best round:    {best_rnd}")
        print()

        # Privacy-utility table
        print(f"  PRIVACY-UTILITY ANALYSIS:")
        print(f"  {'Method':<25} {'HR@10':>7}  {'NDCG@10':>8}")
        print(f"  {'-'*42}")
        if tgt:
            print(f"  {'Paper FedPCL':<25} {tgt['HR@10']:>6.2f}%  {tgt['NDCG@10']:>7.2f}%")
        if s4 and s4['HR@10'] > 0:
            print(f"  {'Stage 4 (no LDP)':<25} {s4['HR@10']:>6.2f}%  {s4['NDCG@10']:>7.2f}%")
            ldp_label = f"Stage 5 (LDP ε={epsilon:.0f})" if hp['use_ldp'] else "Stage 5 (LDP off)"
            hr_drop   = best_hr*100 - s4['HR@10']
            ndcg_drop = best_ndcg*100 - s4['NDCG@10']
            print(f"  {ldp_label:<25} {best_hr*100:>6.2f}%  {best_ndcg*100:>7.2f}%")
            print(f"  {'LDP cost':<25} {hr_drop:>+6.2f}%  {ndcg_drop:>+7.2f}%")
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
            'epsilon':    round(epsilon, 3) if epsilon != float('inf') else None,
            'log':        log,
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
        'dataset':    dataset_name,
        f'HR@{k}':   round(best_hr * 100, 3),
        f'NDCG@{k}': round(best_ndcg * 100, 3),
        'best_round': best_rnd,
        'epsilon':    epsilon,
        'server':     server,
        'clients':    clients,
        'bundle':     bundle,
    }
