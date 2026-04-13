"""
train_stage5.py
═══════════════
Entry point for Stage 5: Full FedPCL with Local Differential Privacy.
The ONLY file you run.

Usage:
    # Default: weak LDP (ε=100/round, near-lossless accuracy)
    python train_stage5.py --dataset steam --data_path steam_processed.json

    # Stronger privacy (ε=10/round, small accuracy drop)
    python train_stage5.py --dataset steam --data_path steam_processed.json \\
        --noise_scale 0.1

    # Very strong privacy (ε=1/round, notable accuracy drop)
    python train_stage5.py --dataset ml100k --data_path u.data \\
        --noise_scale 1.0 --clip_norm 1.0

    # Robust aggregation (Byzantine resilience) + weak LDP
    python train_stage5.py --dataset amazon --data_path amazon_electronics.csv \\
        --robust_aggregation --trim_frac 0.1

Privacy tradeoff guide:
    noise_scale=0.001  ε≈1000/round  near-lossless (essentially Stage 4)
    noise_scale=0.01   ε≈100/round   weak LDP, <1% accuracy drop expected
    noise_scale=0.1    ε≈10/round    moderate LDP, 1-3% accuracy drop
    noise_scale=1.0    ε≈1/round     strong LDP, 5-15% accuracy drop
"""

import argparse
import os
import random
import numpy as np
import torch

from federated_core_stage5 import train_stage5, DATASET_PATHS

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 5: Full FedPCL — Federated Personalized Contrastive '
                    'Learning with Local Differential Privacy'
    )
    # Dataset
    parser.add_argument('--dataset',           type=str,   default='steam',
                        choices=list(DATASET_PATHS.keys()))
    parser.add_argument('--data_path',         type=str,   default=None)

    # Training
    parser.add_argument('--n_rounds',          type=int,   default=400)
    parser.add_argument('--clients_per_round', type=int,   default=128)
    parser.add_argument('--embed_dim',         type=int,   default=64)
    parser.add_argument('--n_gnn_layers',      type=int,   default=2)
    parser.add_argument('--local_epochs',      type=int,   default=5)

    # Personalization
    parser.add_argument('--n_clusters',        type=int,   default=5)
    parser.add_argument('--mu1',               type=float, default=0.5)
    parser.add_argument('--mu2',               type=float, default=0.5)
    parser.add_argument('--cluster_every',     type=int,   default=10)

    # Contrastive
    parser.add_argument('--beta1',             type=float, default=0.1)
    parser.add_argument('--lam',               type=float, default=1.0)
    parser.add_argument('--tau',               type=float, default=0.2)
    parser.add_argument('--drop_rate',         type=float, default=0.3)
    parser.add_argument('--warmup_rounds',     type=int,   default=20)
    parser.add_argument('--max_neigh',         type=int,   default=20)

    # LDP (Stage 5 new args)
    parser.add_argument('--clip_norm',         type=float, default=1.0,
                        help='LDP clipping norm σ (L2). Default=1.0')
    parser.add_argument('--noise_scale',       type=float, default=0.01,
                        help='Laplace noise scale λ. ε=σ/λ. Default=0.01 '
                             '(weak LDP). Use 0.1 for moderate, 1.0 for strong.')
    parser.add_argument('--robust_aggregation',action='store_true',
                        help='Use trimmed-mean aggregation instead of FedAvg')
    parser.add_argument('--trim_frac',         type=float, default=0.1,
                        help='Fraction to trim from each end (trimmed-mean). '
                             'Only used if --robust_aggregation is set.')

    # Optimisation
    parser.add_argument('--lr_item',           type=float, default=0.1)
    parser.add_argument('--lr_user',           type=float, default=0.001)
    parser.add_argument('--weight_decay',      type=float, default=1e-6)
    parser.add_argument('--eval_every',        type=int,   default=10)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    data_path = args.data_path or DATASET_PATHS.get(args.dataset)
    if not data_path or not os.path.exists(data_path):
        print(f"Error: data file not found. Use --data_path <path>")
        return

    hp = {
        'embed_dim':            args.embed_dim,
        'n_gnn_layers':         args.n_gnn_layers,
        'n_rounds':             args.n_rounds,
        'clients_per_round':    args.clients_per_round,
        'local_epochs':         args.local_epochs,
        'n_clusters':           args.n_clusters,
        'mu1':                  args.mu1,
        'mu2':                  args.mu2,
        'cluster_every':        args.cluster_every,
        'beta1':                args.beta1,
        'lam':                  args.lam,
        'tau':                  args.tau,
        'drop_rate':            args.drop_rate,
        'warmup_rounds':        args.warmup_rounds,
        'max_neigh':            args.max_neigh,
        'max_items_neigh':      10,
        'clip_norm':            args.clip_norm,
        'noise_scale':          args.noise_scale,
        'robust_aggregation':   args.robust_aggregation,
        'trim_frac':            args.trim_frac,
        'lr_item':              args.lr_item,
        'lr_user':              args.lr_user,
        'weight_decay':         args.weight_decay,
        'eval_every':           args.eval_every,
        'top_k':                10,
    }

    result = train_stage5(args.dataset, data_path, hp, device, verbose=True)

    # Print final privacy summary
    eps = result['total_epsilon']
    print(f"\n[Privacy Summary]")
    print(f"  Total ε = {eps:.2f}")
    print(f"  Interpretation: each user's data is ε-LDP protected "
          f"(basic composition over {hp['n_rounds']} rounds)")
    print(f"  To tighten: increase --noise_scale "
          f"(current={hp['noise_scale']})")


if __name__ == '__main__':
    main()
