"""
show_results.py
═══════════════
Reads saved training logs and displays a clean results summary.
No model re-running needed.

Usage:
    python show_results.py --file stage4_log_steam.json
    python show_results.py --file stage4_log_steam.json --plot
    python show_results.py --file log1.json log2.json log3.json
    python show_results.py                              # auto-find all logs
    python show_results.py --dataset steam --stage 4   # filter auto-find
"""

import json
import os
import argparse

LOG_PATTERNS = {
    1: 'central_log_{dataset}.json',
    2: 'fedavg_log_{dataset}.json',
    3: 'stage3_log_{dataset}.json',
    4: 'stage4_log_{dataset}.json',
}

STAGE_NAMES = {
    1: 'Stage 1 — Centralized LightGCN',
    2: 'Stage 2 — FedAvg',
    3: 'Stage 3 — FedAvg + Clustering',
    4: 'Stage 4 — FedPCL (Full)',
}

DATASETS = ['steam', 'ml100k', 'ml1m', 'filmtrust', 'amazon']

PAPER_TARGETS = {
    1: {
        'steam':     (80.16, 66.03),
        'ml100k':    (64.31, 45.57),
        'ml1m':      (64.53, 46.78),
        'filmtrust': (21.85, 10.53),
        'amazon':    (38.83, 26.74),
    },
    2: {
        'steam':     (71.21, 50.22),
        'ml100k':    (42.70, 23.87),
        'ml1m':      (44.70, 24.90),
        'filmtrust': (10.81,  4.83),
        'amazon':    (26.53, 14.53),
    },
    3: {
        'steam':     (76.61, 62.63),
        'ml100k':    (61.87, 43.51),
        'ml1m':      (61.31, 42.83),
        'filmtrust': (15.12,  8.01),
        'amazon':    (32.64, 21.39),
    },
    4: {
        'steam':     (80.36, 65.55),
        'ml100k':    (63.81, 45.03),
        'ml1m':      (62.86, 44.12),
        'filmtrust': (16.81,  8.61),
        'amazon':    (34.04, 22.93),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
def detect_stage_dataset(filepath, data):
    """Guess stage and dataset from the filename or JSON contents."""
    fname = os.path.basename(filepath).lower()

    dataset = data.get('dataset', None)
    if not dataset:
        for ds in DATASETS:
            if ds in fname:
                dataset = ds
                break

    stage = None
    if 'central' in fname or 'stage1' in fname:
        stage = 1
    elif 'fedavg' in fname or 'stage2' in fname:
        stage = 2
    elif 'stage3' in fname:
        stage = 3
    elif 'stage4' in fname:
        stage = 4

    return stage, dataset


# ══════════════════════════════════════════════════════════════════════════════
def find_logs_auto(stage=None, dataset=None):
    """Auto-find log files in the current directory."""
    found = []
    stages   = [stage]   if stage   else [1, 2, 3, 4]
    datasets = [dataset] if dataset else DATASETS
    for s in stages:
        for ds in datasets:
            fname = LOG_PATTERNS[s].format(dataset=ds)
            if os.path.exists(fname):
                found.append(fname)
    return found


# ══════════════════════════════════════════════════════════════════════════════
def load_log(filepath):
    with open(filepath) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
def print_summary(filepath, data, stage, dataset):
    bar = "=" * 62
    sep = "-" * 62

    stage_label   = STAGE_NAMES.get(stage, f'Stage {stage}') if stage else 'Unknown Stage'
    dataset_label = dataset.upper() if dataset else 'Unknown Dataset'

    print(f"\n{bar}")
    print(f"  {stage_label}  —  {dataset_label}")
    print(f"  File: {os.path.basename(filepath)}")
    print(bar)

    best_hr   = data.get('best_hr',    0)
    best_ndcg = data.get('best_ndcg',  0)
    best_rnd  = data.get('best_round', '?')
    print(f"  Best HR@10  : {best_hr:.3f}%   (round {best_rnd})")
    print(f"  Best NDCG@10: {best_ndcg:.3f}%")

    tgt = PAPER_TARGETS.get(stage, {}).get(dataset) if (stage and dataset) else None
    if tgt:
        hr_gap   = best_hr   - tgt[0]
        ndcg_gap = best_ndcg - tgt[1]
        print(f"\n  Paper target : HR@10={tgt[0]:.2f}%   NDCG@10={tgt[1]:.2f}%")
        print(f"  Gap          : HR@10 {hr_gap:+.2f}%   NDCG@10 {ndcg_gap:+.2f}%")
        status = "Matches paper" if hr_gap >= -2.0 else "Below paper"
        print(f"  Status       : {status}")

    hp = data.get('hparams', {})
    if hp:
        print(f"\n  Hyperparameters used:")
        key_hps = ['embed_dim', 'n_gnn_layers', 'n_rounds', 'local_epochs',
                   'lr_item', 'lr_user', 'weight_decay',
                   'n_clusters', 'mu1', 'mu2',
                   'beta1', 'lam', 'tau', 'drop_rate', 'warmup_rounds']
        for k in key_hps:
            if k in hp:
                print(f"    {k:<20} = {hp[k]}")

    log = data.get('log', [])
    if log:
        print(f"\n  Training curve  (sampled every ~10% of rounds):")
        print(f"  {'Round':>6}  {'HR@10':>7}  {'NDCG@10':>8}  {'Loss':>8}")
        print(f"  {sep[:46]}")

        n       = len(log)
        step    = max(1, n // 10)
        indices = sorted(set(list(range(0, n, step)) + [n - 1]))
        for idx in indices:
            row  = log[idx]
            rnd  = row.get('round', '?')
            hr   = row.get('HR@10',   row.get('hr10',  0))
            ndcg = row.get('NDCG@10', row.get('ndcg10', 0))
            loss = row.get('loss', 0)
            mark = " *" if rnd == best_rnd else ""
            print(f"  {rnd:>6}  {hr:>6.2f}%  {ndcg:>7.2f}%  {loss:>8.5f}{mark}")

    print(bar)


# ══════════════════════════════════════════════════════════════════════════════
def print_comparison_table(results):
    if len(results) < 2:
        return
    print("\n" + "=" * 72)
    print("  COMPARISON TABLE")
    print("=" * 72)
    print(f"  {'File':<35} {'HR@10':>7}  {'NDCG@10':>8}  {'Round':>6}")
    print(f"  {'-' * 60}")
    for filepath, data, stage, dataset in results:
        fname = os.path.basename(filepath)
        hr    = data.get('best_hr',    0)
        ndcg  = data.get('best_ndcg',  0)
        rnd   = data.get('best_round', '?')
        print(f"  {fname:<35} {hr:>6.2f}%  {ndcg:>7.2f}%  {rnd:>6}")
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
def plot_curves(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [Plot] matplotlib not installed. Run: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FedPCL Training Curves', fontsize=14)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#a65628', '#f781bf', '#999999']

    for idx, (filepath, data, stage, dataset) in enumerate(results):
        log = data.get('log', [])
        if not log:
            continue
        label  = os.path.basename(filepath).replace('.json', '')
        color  = colors[idx % len(colors)]
        rounds = [r.get('round', 0) for r in log]
        hrs    = [r.get('HR@10',   r.get('hr10',  0)) for r in log]
        ndcgs  = [r.get('NDCG@10', r.get('ndcg10', 0)) for r in log]
        axes[0].plot(rounds, hrs,   label=label, color=color, linewidth=1.8)
        axes[1].plot(rounds, ndcgs, label=label, color=color, linewidth=1.8)

        tgt = PAPER_TARGETS.get(stage, {}).get(dataset) if (stage and dataset) else None
        if tgt:
            axes[0].axhline(tgt[0], linestyle='--', color=color, alpha=0.4,
                            label=f'Paper ({label})')
            axes[1].axhline(tgt[1], linestyle='--', color=color, alpha=0.4)

    for ax, title in [(axes[0], 'HR@10 (%)'), (axes[1], 'NDCG@10 (%)')]:
        ax.set_title(title)
        ax.set_xlabel('Round')
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    outfile = 'training_curves.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n  [Plot] Saved -> {outfile}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Show FedPCL training results from saved JSON log files',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python show_results.py --file stage4_log_steam.json
  python show_results.py --file stage4_log_steam.json fedavg_log_steam.json
  python show_results.py --file stage4_log_steam.json --plot
  python show_results.py                        (auto-find all logs)
  python show_results.py --dataset steam        (auto-find Steam logs)
  python show_results.py --stage 4              (auto-find Stage 4 logs)
        """
    )
    parser.add_argument(
        '--file', nargs='+', default=None, metavar='FILE',
        help='One or more JSON log files to display.\n'
             'Example: --file stage4_log_steam.json\n'
             'Multiple: --file log1.json log2.json'
    )
    parser.add_argument(
        '--dataset', type=str, default=None, choices=DATASETS,
        help='Filter auto-find by dataset (ignored if --file is used)'
    )
    parser.add_argument(
        '--stage', type=int, default=None, choices=[1, 2, 3, 4],
        help='Filter auto-find by stage (ignored if --file is used)'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Plot HR@10 and NDCG@10 training curves'
    )
    args = parser.parse_args()

    # ── Collect file list ─────────────────────────────────────────────────────
    if args.file:
        filepaths = args.file
        missing = [f for f in filepaths if not os.path.exists(f)]
        if missing:
            for m in missing:
                print(f"  File not found: {m}")
            return
    else:
        filepaths = find_logs_auto(stage=args.stage, dataset=args.dataset)
        if not filepaths:
            print("\n  No log files found in current directory.")
            print("  Specify a file directly with --file, for example:")
            print("    python show_results.py --file stage4_log_steam.json")
            return
        print(f"\n  Auto-found {len(filepaths)} log file(s):")
        for f in filepaths:
            print(f"    {f}")

    # ── Load and display ──────────────────────────────────────────────────────
    results = []
    for fp in filepaths:
        data = load_log(fp)
        stage, dataset = detect_stage_dataset(fp, data)
        print_summary(fp, data, stage, dataset)
        results.append((fp, data, stage, dataset))

    if len(results) > 1:
        print_comparison_table(results)

    if args.plot:
        plot_curves(results)


if __name__ == '__main__':
    main()
