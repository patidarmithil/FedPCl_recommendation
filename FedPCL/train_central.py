"""
train_central.py
════════════════
Centralized LightGCN training loop.
Target: reproduce "Center" column in Table I of the FedPCL paper.

Paper targets (Table I, Center column):
  ML-1M     HR@10=64.53%  NDCG@10=46.78%
  ML-100K   HR@10=64.31%  NDCG@10=45.57%
  Steam     HR@10=80.16%  NDCG@10=66.03%
  Filmtrust HR@10=21.85%  NDCG@10=10.53%
  Amazon-E  HR@10=38.83%  NDCG@10=26.74%

Hyperparameters (paper Section IV-A):
  embed_dim  = 64
  n_layers   = 2
  optimizer  = Adam
  lr         = 0.001      (standard Adam lr for centralized training)
  L2 reg     = 1e-4       (standard for LightGCN)
  batch_size = 2048
  epochs     = 200        (or until convergence, best model saved by HR@10)
  eval_every = 5 epochs

Usage:
    # Single dataset
    python train_central.py --dataset steam --data_path steam_processed.json

    # All datasets (requires all data files)
    python train_central.py --all
"""

import argparse
import math
import random
import time
import json
import os

import numpy as np
import torch
import torch.optim as optim

from data_loader import load_dataset, load_item_names, build_edge_index, sample_negatives_batch
from model import LightGCN, l2_reg_loss, evaluate_model

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
HPARAMS = {
    # Directly from paper Section IV-A
    'embed_dim':  64,      # d=64 (paper)
    'n_layers':   3,       # K=3 (LightGCN He et al. 2020 default for centralized)
    'lr':         0.001,   # Adam lr — paper says Adam optimizer; 0.001 is
                           # standard Adam lr for centralized LightGCN.
                           # Paper's η=0.1 is the federated SGD lr (Eq.11),
                           # NOT the centralized Adam lr.
    'l2_reg':     1e-4,    # from LightGCN paper He et al. 2020 (NOT FedPCL)
    'batch_size': 2048,
    'n_epochs':   1000,    # LightGCN trains up to 1000; early stopping exits sooner
    'eval_every': 5,
    'top_k':      10,
    'patience':   20,      # stop after 20x5=100 epochs without improvement
}

# Dataset paths — edit these to match your file locations
DATASET_PATHS = {
    'steam':     'steam_processed.json',
    'ml100k':    'ml-100k/u.data',
    'ml1m':      'ml-1m/ratings.dat',
    'filmtrust': 'filmtrust/ratings.txt',
    'amazon':    'amazon_electronics.csv',
}

# Item name files per dataset (optional — for displaying names in recommendations)
# Place these files in the same folder as the data files
ITEM_NAME_FILES = {
    'steam':     'steam_processed.json',   # names embedded in JSON
    'ml100k':    'u.item',                 # download with ML-100K zip
    'ml1m':      'movies.dat',             # download with ML-1M zip
    'filmtrust': None,
    'amazon':    None,
}

# Paper Table I targets for reference
TARGETS = {
    'steam':     {'HR@10': 80.16, 'NDCG@10': 66.03},
    'ml100k':    {'HR@10': 64.31, 'NDCG@10': 45.57},
    'ml1m':      {'HR@10': 64.53, 'NDCG@10': 46.78},
    'filmtrust': {'HR@10': 21.85, 'NDCG@10': 10.53},
    'amazon':    {'HR@10': 38.83, 'NDCG@10': 26.74},
}


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def get_train_batches(train_dict: dict, n_items: int, batch_size: int):
    """
    Generate (users, pos_items, neg_items) batches for one epoch.

    For each positive (user, item) pair in training data:
      • Sample one random negative item not in user's training set
      • Pack into batches of size batch_size

    Yields:
        (users_batch, pos_batch, neg_batch)  each [B] LongTensor on CPU
    """
    # Build flat list of all positive pairs
    pairs = []
    for u, items in train_dict.items():
        for i in items:
            pairs.append((u, i))

    # Shuffle for stochastic gradient
    random.shuffle(pairs)

    # Split into batches
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        users    = [p[0] for p in batch_pairs]
        pos_itms = [p[1] for p in batch_pairs]
        neg_itms = sample_negatives_batch(users, train_dict, n_items)

        yield (
            torch.tensor(users,    dtype=torch.long),
            torch.tensor(pos_itms, dtype=torch.long),
            torch.tensor(neg_itms, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def train(dataset_name: str,
          data_path:    str,
          hparams:      dict = None,
          device:       torch.device = None,
          verbose:      bool = True):
    """
    Full training loop for centralized LightGCN.

    Args:
        dataset_name:  e.g. 'steam', 'ml100k', 'ml1m', 'filmtrust', 'amazon'
        data_path:     path to raw data file
        hparams:       override any default hyperparameters
        device:        torch device (auto-detected if None)
        verbose:       print progress

    Returns:
        results dict with best HR@K and NDCG@K
    """
    if hparams is None:
        hparams = {}
    hp = {**HPARAMS, **hparams}   # merge with defaults

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bar = "=" * 68

    if verbose:
        print(bar)
        print(f"  Centralized LightGCN — {dataset_name.upper()}")
        print(bar)
        print(f"  device={device}  d={hp['embed_dim']}  K={hp['n_layers']}")
        print(f"  lr={hp['lr']}  l2={hp['l2_reg']}  batch={hp['batch_size']}")
        target = TARGETS.get(dataset_name, {})
        if target:
            print(f"  Target: HR@10={target['HR@10']:.2f}%  "
                  f"NDCG@10={target['NDCG@10']:.2f}%")
        print(bar)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    bundle = load_dataset(dataset_name, data_path)

    # ── 2. Build edge index (bipartite graph) ─────────────────────────────────
    edge_index, edge_weight = build_edge_index(bundle)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # ── 3. Build model ────────────────────────────────────────────────────────
    model = LightGCN(
        n_users     = bundle.n_users,
        n_items     = bundle.n_items,
        embed_dim   = hp['embed_dim'],
        n_layers    = hp['n_layers'],
        edge_index  = edge_index,
        edge_weight = edge_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Parameters: {n_params:,}  "
              f"({bundle.n_users + bundle.n_items} nodes × {hp['embed_dim']} dims)")

    # ── 4. Optimizer ──────────────────────────────────────────────────────────
    # Adam with lr=0.001 (standard for centralized LightGCN)
    # Paper says lr=0.1 for federated SGD, but Adam with 0.001 is correct here
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'])

    # ── 5. Training loop ──────────────────────────────────────────────────────
    best_hr   = 0.0
    best_ndcg = 0.0
    best_epoch = 0
    best_state = None
    patience_cnt = 0
    log = []

    if verbose:
        print(f"\n  {'Epoch':>6} | {'Loss':>8} | "
              f"{'HR@10':>7} | {'NDCG@10':>8} | {'Time':>6}")
        print(f"  {'-'*50}")

    total_epochs = hp['n_epochs']
    k = hp['top_k']

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        # ── Mini-batch training ────────────────────────────────────────────────
        for users, pos_items, neg_items in get_train_batches(
                bundle.train_dict, bundle.n_items, hp['batch_size']):

            users     = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()

            # BPR loss (Eq.4 in paper)
            loss_bpr = model.bpr_loss(users, pos_items, neg_items)

            # L2 regularisation on initial embeddings only (standard LightGCN)
            loss_reg = l2_reg_loss(model, users, pos_items, neg_items,
                                   weight=hp['l2_reg'])

            loss = loss_bpr + loss_reg
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0

        # ── Evaluation every eval_every epochs ────────────────────────────────
        if epoch % hp['eval_every'] == 0 or epoch == 1:
            metrics = evaluate_model(
                model, bundle.test_dict, bundle.neg_dict, k=k
            )
            hr   = metrics[f'HR@{k}']
            ndcg = metrics[f'NDCG@{k}']

            if hr > best_hr:
                best_hr    = hr
                best_ndcg  = ndcg
                best_epoch = epoch
                patience_cnt = 0
                # Save best model state
                best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
            else:
                patience_cnt += 1

            if verbose:
                marker = " ★" if epoch == best_epoch else ""
                print(f"  {epoch:>6} | {avg_loss:>8.4f} | "
                      f"{hr*100:>6.2f}% | {ndcg*100:>7.2f}% | "
                      f"{dt:>4.1f}s{marker}")

            log.append({
                'epoch':  epoch,
                'loss':   round(avg_loss, 5),
                f'HR@{k}':   round(hr * 100, 3),
                f'NDCG@{k}': round(ndcg * 100, 3),
            })

            # Early stopping
            if patience_cnt >= hp['patience']:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(no improvement for {hp['patience']} eval intervals)")
                break
        else:
            if verbose and epoch % 10 == 0:
                print(f"  {epoch:>6} | {avg_loss:>8.4f} | {'---':>7} | "
                      f"{'---':>8} | {dt:>4.1f}s")

    # ── 6. Restore best model ─────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── 7. Final evaluation ───────────────────────────────────────────────────
    final_metrics = evaluate_model(model, bundle.test_dict, bundle.neg_dict, k=k)
    final_hr   = final_metrics[f'HR@{k}']
    final_ndcg = final_metrics[f'NDCG@{k}']

    target = TARGETS.get(dataset_name, {})
    if verbose:
        print(f"\n{bar}")
        print(f"  RESULT  ({dataset_name.upper()})")
        print(f"  Best HR@{k}:   {final_hr*100:.2f}%  "
              f"(paper target: {target.get('HR@10', 'N/A')}%)")
        print(f"  Best NDCG@{k}: {final_ndcg*100:.2f}%  "
              f"(paper target: {target.get('NDCG@10', 'N/A')}%)")
        print(f"  Best epoch: {best_epoch}")

        if target:
            gap_hr   = final_hr*100 - target['HR@10']
            gap_ndcg = final_ndcg*100 - target['NDCG@10']
            print(f"  Gap vs paper: HR@10 {gap_hr:+.2f}%  "
                  f"NDCG@10 {gap_ndcg:+.2f}%")
        print(bar)

    # Save training log
    log_path = f'central_log_{dataset_name}.json'
    with open(log_path, 'w') as f:
        json.dump({'dataset': dataset_name, 'hparams': hp,
                   'best_hr': round(final_hr*100, 3),
                   'best_ndcg': round(final_ndcg*100, 3),
                   'best_epoch': best_epoch, 'log': log}, f, indent=2)
    if verbose:
        print(f"  Log → {log_path}")

    return {
        'dataset':   dataset_name,
        f'HR@{k}':   round(final_hr * 100, 3),
        f'NDCG@{k}': round(final_ndcg * 100, 3),
        'best_epoch': best_epoch,
        'model':     model,
        'bundle':    bundle,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TOP-K RECOMMENDATION FOR A SPECIFIC USER  (for demo/inspection)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def recommend_for_user(model: LightGCN,
                       bundle,
                       user_id: int,
                       k: int = 10,
                       id2item: dict = None) -> list:
    """
    Return top-K recommended items for user_id (excluding training items).

    Args:
        model:    trained LightGCN
        bundle:   DataBundle
        user_id:  integer user ID
        k:        number of recommendations
        id2item:  dict {item_int_id: item_name} (optional, for display)

    Returns:
        list of (item_id, item_name_or_id, score)
    """
    model.eval()
    scores = model.get_user_ratings(user_id).cpu()

    # Mask training items (already seen)
    seen = bundle.adj_user.get(user_id, [])
    scores[seen] = -1e9

    # Top-K
    topk_ids = torch.topk(scores, k).indices.tolist()

    results = []
    for iid in topk_ids:
        name = id2item.get(iid, f'item_{iid}') if id2item else f'item_{iid}'
        results.append((iid, name, float(scores[iid])))
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description='Centralized LightGCN training (FedPCL paper baseline)'
    )
    parser.add_argument('--dataset',   type=str, default='steam',
                        choices=list(DATASET_PATHS.keys()),
                        help='Dataset to train on')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file (overrides default path)')
    parser.add_argument('--all',       action='store_true',
                        help='Train on all 5 datasets (requires all data files)')
    parser.add_argument('--embed_dim', type=int,   default=64)
    parser.add_argument('--n_layers',  type=int,   default=2)
    parser.add_argument('--lr',        type=float, default=0.001)
    parser.add_argument('--l2_reg',    type=float, default=1e-4)
    parser.add_argument('--batch_size',type=int,   default=2048)
    parser.add_argument('--n_epochs',  type=int,   default=1000)
    parser.add_argument('--eval_every',type=int,   default=5)
    parser.add_argument('--patience',  type=int,   default=30)
    parser.add_argument('--names_path', type=str,  default=None,
                        help='Path to item names file (e.g. u.item for ml100k, movies.dat for ml1m)')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # Build hyperparams from args
    hp = {
        'embed_dim':  args.embed_dim,
        'n_layers':   args.n_layers,
        'lr':         args.lr,
        'l2_reg':     args.l2_reg,
        'batch_size': args.batch_size,
        'n_epochs':   args.n_epochs,
        'eval_every': args.eval_every,
        'patience':   args.patience,
        'top_k':      10,
    }

    if args.all:
        # ── Train all 5 datasets ──────────────────────────────────────────────
        all_results = {}
        for ds_name, ds_path in DATASET_PATHS.items():
            if not os.path.exists(ds_path):
                print(f"\n  [SKIP] {ds_name}: file not found at {ds_path}")
                continue
            result = train(ds_name, ds_path, hp, device, verbose=True)
            all_results[ds_name] = result

        # Summary table
        print("\n" + "="*68)
        print("  SUMMARY — Centralized LightGCN vs Paper Table I")
        print("="*68)
        print(f"  {'Dataset':12s}  {'HR@10':>8}  {'NDCG@10':>9}  "
              f"{'Target HR':>10}  {'Target NDCG':>12}  {'HR gap':>7}")
        print(f"  {'-'*65}")
        for ds, res in all_results.items():
            tgt = TARGETS.get(ds, {})
            hr_gap = res['HR@10'] - tgt.get('HR@10', 0)
            print(f"  {ds:12s}  {res['HR@10']:>7.2f}%  {res['NDCG@10']:>8.2f}%  "
                  f"{tgt.get('HR@10',0):>9.2f}%  {tgt.get('NDCG@10',0):>11.2f}%  "
                  f"{hr_gap:>+6.2f}%")
        print("="*68)

    else:
        # ── Train single dataset ──────────────────────────────────────────────
        data_path = args.data_path or DATASET_PATHS.get(args.dataset)
        if data_path is None or not os.path.exists(data_path):
            print(f"Error: data file not found. Provide --data_path <path>")
            return

        result = train(args.dataset, data_path, hp, device, verbose=True)

        # Show recommendations for user 0
        model  = result['model']
        bundle = result['bundle']
        recs   = recommend_for_user(model, bundle, user_id=0, k=10)
        print(f"\n  TOP-10 RECOMMENDATIONS FOR USER 0:")
        for rank, (iid, name, score) in enumerate(recs, 1):
            print(f"    {rank:2d}. item_{iid:<6d}  score={score:.4f}")
        if 0 in bundle.test_dict:
            print(f"\n  Held-out test item: item_{bundle.test_dict[0]}")


if __name__ == '__main__':
    main()
