"""
federated_core.py
═════════════════
FedAvg federated training loop — fixed rounds version.
No early stopping: runs exactly n_rounds rounds.

Command:
  python train_fedavg.py --dataset steam \
      --data_path steam_processed.json \
      --n_rounds 400 --local_epochs 10

Targets (Table I, FedPCL paper):
  Steam:  HR@10=71.21%  NDCG@10=50.22%
"""

import math
import random
import time
import json
import torch

from data_loader import load_dataset, load_item_names
from client import Client
from server import Server


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
HPARAMS = {
    'embed_dim':         64,
    'n_gnn_layers':      2,       # K=2 for federated (paper)
    'n_rounds':          400,     # paper trains 400 rounds
    'clients_per_round': 128,     # paper: 128 clients per round
    'local_epochs':      5,       # E: local epochs (FedAvg paper: E=5)
    'lr_item':           0.1,     # SGD lr for item embeddings (paper η=0.1)
    'lr_user':           0.001,   # Adam lr for user embedding (local only)
    'weight_decay':      1e-6,    # L2 reg (paper β₂=1e-6)
    'eval_every':        10,      # evaluate every N rounds
    'top_k':             10,      # HR@10, NDCG@10
}

# FedAvg targets — Table I of FedPCL paper (Wang et al. IEEE TCSS 2025)
TARGETS = {
    'steam':     {'HR@10': 71.21, 'NDCG@10': 50.22},
    'ml100k':    {'HR@10': 42.70, 'NDCG@10': 23.87},
    'ml1m':      {'HR@10': 44.70, 'NDCG@10': 24.90},
    'filmtrust': {'HR@10': 10.81, 'NDCG@10':  4.83},
    'amazon':    {'HR@10': 26.53, 'NDCG@10': 14.53},
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
def evaluate(clients, server, test_dict, neg_dict,
             n_gnn_layers, top_k=10):
    """
    HR@K and NDCG@K over all test users.
    Candidates = {test_item} + 100 random negatives.
    """
    E_global = server.get_global_embeddings()
    total_hr, total_ndcg, n = 0.0, 0.0, 0

    for uid, test_item in test_dict.items():
        if uid not in clients or uid not in neg_dict:
            continue
        scores     = clients[uid].get_scores(E_global, n_gnn_layers)
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
def train_fedavg(dataset_name, data_path,
                 hparams=None, device=None, verbose=True):
    """
    Full FedAvg training — runs exactly hp['n_rounds'] rounds.
    No early stopping: deterministic, reproducible.
    """
    if hparams is None:
        hparams = {}
    hp = {**HPARAMS, **hparams}

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bar = "=" * 68
    k   = hp['top_k']

    # ── Load data ─────────────────────────────────────────────────────────────
    if verbose:
        print(bar)
        print(f"  FedAvg LightGCN — {dataset_name.upper()}")
        print(bar)
        print(f"  device={device}  d={hp['embed_dim']}  K={hp['n_gnn_layers']}")
        print(f"  lr_item={hp['lr_item']}  lr_user={hp['lr_user']}  "
              f"wd={hp['weight_decay']}  local_epochs={hp['local_epochs']}")
        print(f"  rounds={hp['n_rounds']}  clients/round={hp['clients_per_round']}")
        tgt = TARGETS.get(dataset_name, {})
        if tgt:
            print(f"  Target: HR@10={tgt['HR@10']:.2f}%  "
                  f"NDCG@10={tgt['NDCG@10']:.2f}%")
        print(bar)

    bundle = load_dataset(dataset_name, data_path)

    # ── Server ────────────────────────────────────────────────────────────────
    server = Server(
        n_items   = bundle.n_items,
        embed_dim = hp['embed_dim'],
        device    = device,
    )

    # ── Clients ───────────────────────────────────────────────────────────────
    clients = {}
    sizes   = {}
    for uid, items in bundle.train_dict.items():
        if len(items) < 1:
            continue
        clients[uid] = Client(
            uid         = uid,
            train_items = items,
            n_items     = bundle.n_items,
            embed_dim   = hp['embed_dim'],
            device      = device,
        )
        sizes[uid] = len(items)

    all_ids = list(clients.keys())

    if verbose:
        print(f"  Clients initialised: {len(clients)}")
        print(bar)
        print(f"\n  {'Round':>6} | {'Loss':>8} | "
              f"{'HR@10':>7} | {'NDCG@10':>8} | "
              f"{'Updated':>8} | {'Time':>6}")
        print(f"  {'-'*58}")

    # ── Training loop — fixed rounds, no early stopping ───────────────────────
    best_hr   = 0.0
    best_ndcg = 0.0
    best_rnd  = 0
    log       = []

    for rnd in range(1, hp['n_rounds'] + 1):
        t0 = time.time()

        sel_ids   = server.select_clients(all_ids, hp['clients_per_round'])
        E_global  = server.get_global_embeddings()
        delta_list = []
        losses    = []

        for uid in sel_ids:
            item_deltas, loss = clients[uid].local_train(
                E_global     = E_global,
                n_layers     = hp['n_gnn_layers'],
                local_epochs = hp['local_epochs'],
                lr_item      = hp['lr_item'],
                lr_user      = hp['lr_user'],
                weight_decay = hp['weight_decay'],
            )
            delta_list.append(item_deltas)
            if math.isfinite(loss):
                losses.append(loss)

        stats    = server.aggregate(sel_ids, delta_list, sizes)
        avg_loss = sum(losses) / max(len(losses), 1)
        dt       = time.time() - t0

        # Evaluate every eval_every rounds AND on the last round
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
                print(f"  {rnd:>6} | {avg_loss:>8.4f} | "
                      f"{hr*100:>6.2f}% | {ndcg*100:>7.2f}% | "
                      f"{stats['n_items_updated']:>7} | "
                      f"{dt:>4.1f}s{marker}")

            log.append({
                'round':      rnd,
                'loss':       round(avg_loss, 5),
                f'HR@{k}':   round(hr * 100, 3),
                f'NDCG@{k}': round(ndcg * 100, 3),
            })

    # ── Results ───────────────────────────────────────────────────────────────
    tgt = TARGETS.get(dataset_name, {})
    if verbose:
        print(f"\n{bar}")
        print(f"  RESULT  ({dataset_name.upper()})")
        print(f"  Best HR@{k}:   {best_hr*100:.2f}%  "
              f"(FedAvg target: {tgt.get('HR@10','N/A')}%)")
        print(f"  Best NDCG@{k}: {best_ndcg*100:.2f}%  "
              f"(FedAvg target: {tgt.get('NDCG@10','N/A')}%)")
        print(f"  Best round: {best_rnd}")
        if tgt:
            hr_gap   = best_hr*100   - tgt['HR@10']
            ndcg_gap = best_ndcg*100 - tgt['NDCG@10']
            print(f"  Gap vs paper: HR@10 {hr_gap:+.2f}%  "
                  f"NDCG@10 {ndcg_gap:+.2f}%")
        print(bar)

    # Save log
    log_path = f'fedavg_log_{dataset_name}.json'
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
        E_global = server.get_global_embeddings()
        scores   = clients[0].get_scores(E_global, hp['n_gnn_layers'])
        seen     = set(bundle.train_dict.get(0, []))
        scores[list(seen)] = -1e9
        topk = torch.topk(scores, k).indices.tolist()
        print(f"\n  TOP-10 RECOMMENDATIONS FOR USER 0:")
        for rank, iid in enumerate(topk, 1):
            name = id2name.get(iid, f'item_{iid}')
            print(f"    {rank:2d}. {str(name)[:55]:<55}  "
                  f"score={float(scores[iid]):.4f}")
        if 0 in bundle.test_dict:
            held = bundle.test_dict[0]
            print(f"\n  Held-out test item: {id2name.get(held, f'item_{held}')}")

    return {
        'dataset':    dataset_name,
        f'HR@{k}':   round(best_hr * 100, 3),
        f'NDCG@{k}': round(best_ndcg * 100, 3),
        'best_round': best_rnd,
        'server':     server,
        'clients':    clients,
        'bundle':     bundle,
    }
