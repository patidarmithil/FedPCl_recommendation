[Seed] 42
[Device] cuda
[LDP] σ=0.1  λ=0.001  ε=100.0
========================================================================
  Stage 5: FedPCL + LDP — STEAM
========================================================================
  device=cuda  d=64  K_gnn=2
  rounds=400  clients/round=128
  local_epochs=10  lr_item=0.1
  clusters K=5  mu1=0.5  mu2=0.5
  beta1=0.0  lam=0.0  tau=0.2

  LDP ENABLED: σ=0.1  λ=0.001  ε=100.0
  Expected convergence loss: 0.01 – 0.08
  Paper target: HR@10=80.36%  NDCG@10=65.55%
  Embedding snapshots will be saved at: [1, 400]
========================================================================
DataBundle(steam)  users=3757  items=5113  train=111382  density=0.580%
  train=111382  test=3757  neg_per_user=100  kcore=off  split=random
  Building clients + 2-hop neighbourhoods ...
  Clients: 3757  |  Avg neighbours: 20.0  |  LDP(σ=0.1,λ=0.001)
  Dataset stats: users=3757  items=5113  interactions=115139  density=0.5994%  kcore=off
========================================================================

   Round |     Loss |   HR@10 |  NDCG@10 |    CL |   LDP |   Time
  --------------------------------------------------------------
  [Emb saved] round=   1  →  emb_steam_round0001.npy
       1 |   0.5890 |   9.48% |    4.64% | off | ON  |  7.9s ★
      10 |   0.5720 |  11.31% |    5.56% | off | ON  |  7.4s ★
      20 |   0.5512 |  14.91% |    7.68% | off | ON  |  9.3s ★
      30 |   0.5532 |  20.23% |   11.32% | ON  | ON  | 13.5s ★
      40 |   0.5237 |  26.19% |   15.99% | ON  | ON  | 12.3s ★
      50 |   0.5051 |  32.31% |   20.96% | ON  | ON  | 12.9s ★
      60 |   0.4670 |  38.75% |   26.22% | ON  | ON  | 12.5s ★
      70 |   0.4514 |  46.58% |   32.32% | ON  | ON  | 13.7s ★
      80 |   0.3994 |  53.87% |   38.24% | ON  | ON  | 13.3s ★
      90 |   0.3869 |  58.34% |   41.88% | ON  | ON  | 15.0s ★
     100 |   0.3416 |  63.30% |   45.52% | ON  | ON  | 14.7s ★
     110 |   0.3047 |  66.49% |   47.80% | ON  | ON  | 13.5s ★
     120 |   0.2988 |  69.68% |   49.76% | ON  | ON  | 14.6s ★
     130 |   0.2557 |  71.65% |   51.22% | ON  | ON  | 12.5s ★
     140 |   0.2563 |  73.57% |   52.32% | ON  | ON  | 14.7s ★
     150 |   0.2412 |  74.58% |   53.02% | ON  | ON  | 14.2s ★
     160 |   0.1992 |  75.65% |   53.75% | ON  | ON  | 13.1s ★
     170 |   0.2165 |  76.02% |   54.22% | ON  | ON  | 12.5s ★
     180 |   0.2094 |  76.76% |   54.55% | ON  | ON  | 13.7s ★
     190 |   0.1865 |  76.92% |   54.77% | ON  | ON  | 14.5s ★
     200 |   0.2003 |  77.54% |   55.03% | ON  | ON  | 14.7s ★
     210 |   0.1802 |  77.64% |   55.05% | ON  | ON  | 15.3s ★
     220 |   0.1700 |  77.99% |   55.20% | ON  | ON  | 13.1s ★
     230 |   0.1536 |  78.31% |   55.50% | ON  | ON  | 14.3s ★
     240 |   0.1652 |  78.20% |   55.64% | ON  | ON  | 14.0s
     250 |   0.1424 |  78.44% |   55.68% | ON  | ON  | 12.6s ★
     260 |   0.1608 |  78.76% |   56.00% | ON  | ON  | 13.9s ★
     270 |   0.1680 |  78.81% |   56.11% | ON  | ON  | 14.2s ★
     280 |   0.1420 |  79.27% |   56.32% | ON  | ON  | 12.7s ★
     290 |   0.1539 |  79.03% |   56.23% | ON  | ON  | 13.8s
     300 |   0.1695 |  79.21% |   56.31% | ON  | ON  | 15.3s
     310 |   0.1430 |  79.29% |   56.47% | ON  | ON  | 13.8s ★
     320 |   0.1196 |  79.43% |   56.49% | ON  | ON  | 13.7s ★
     330 |   0.1457 |  79.37% |   56.53% | ON  | ON  | 13.2s
     340 |   0.1398 |  79.48% |   56.55% | ON  | ON  | 12.8s ★
     350 |   0.1316 |  79.77% |   56.86% | ON  | ON  | 13.4s ★
     360 |   0.1136 |  80.25% |   57.36% | ON  | ON  | 13.5s ★
     370 |   0.1246 |  79.85% |   57.11% | ON  | ON  | 13.1s
     380 |   0.1068 |  80.09% |   57.24% | ON  | ON  | 12.4s
     390 |   0.1031 |  80.36% |   57.41% | ON  | ON  | 12.6s ★
  [Emb saved] round= 400  →  emb_steam_round0400.npy
     400 |   0.1398 |  80.62% |   57.60% | ON  | ON  | 14.6s ★

========================================================================
  RESULT  (STEAM)
  Best HR@10:          80.62%  (round 400)
  Best NDCG@10:        57.60%
  Total training time:  91.4 min
  Avg time per round:   13.27s

  LOSS SUMMARY:
  Round 1 loss                       0.58900
  Minimum loss                       0.10311  (round 390)
  Final loss (round 400)             0.13975
  Expected range at convergence      0.01 – 0.08
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        80.62%    80.36%     +0.26%                   +0.3%
  NDCG@10      57.60%    65.55%     -7.95%                  -12.1%

  Verdict: ✓ Matches paper (within 2%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   80.36%     65.55%
  Stage 4 (no LDP)               78.84%     55.28%
  Stage 5 (LDP ε=100)            80.62%     57.60%
========================================================================
  Log → stage5_log_steam.json

  TOP-10 FOR USER 0  (cluster 0):
     1. Ricochet                                                 score=7.4320
     2. Counter-Strike Condition Zero                            score=7.3419
     3. Counter-Strike Condition Zero Deleted Scenes             score=7.0678
     4. Call of Duty Modern Warfare 3                            score=6.9827
     5. Call of Duty Modern Warfare 3 - Multiplayer              score=6.9144
     6. Day of Defeat Source                                     score=6.6902
     7. Insurgency Modern Infantry Combat                        score=6.4126
     8. Call of Duty Black Ops - Multiplayer                     score=6.3378
     9. Serious Sam HD The Second Encounter                      score=6.2199
    10. Call of Duty Black Ops                                   score=6.1519

  Held-out: Ricochet



[Seed] 42
[Device] cuda
[LDP] σ=0.1  λ=0.001  ε=100.0
========================================================================
  Stage 5: FedPCL + LDP — ML100K
========================================================================
  device=cuda  d=64  K_gnn=2
  rounds=400  clients/round=128
  local_epochs=10  lr_item=0.1
  clusters K=5  mu1=0.5  mu2=0.5
  beta1=0.0  lam=0.0  tau=0.2

  LDP ENABLED: σ=0.1  λ=0.001  ε=100.0
  Expected convergence loss: 0.45 – 0.58
  Paper target: HR@10=63.81%  NDCG@10=45.03%
  Embedding snapshots will be saved at: [1, 400]
========================================================================
DataBundle(ml100k)  users=943  items=1682  train=99057  density=6.245%
  train=99057  test=943  neg_per_user=100  kcore=off  split=random
  Building clients + 2-hop neighbourhoods ...
  Clients: 943  |  Avg neighbours: 20.0  |  LDP(σ=0.1,λ=0.001)
  Dataset stats: users=943  items=1682  interactions=100000  density=6.3047%  kcore=off
========================================================================

   Round |     Loss |   HR@10 |  NDCG@10 |    CL |   LDP |   Time
  --------------------------------------------------------------
  [Emb saved] round=   1  →  emb_ml100k_round0001.npy
       1 |   0.6560 |   9.01% |    3.89% | off | ON  | 13.6s ★
      10 |   0.6477 |   9.97% |    4.31% | off | ON  | 16.8s ★
      20 |   0.6403 |  10.60% |    4.79% | off | ON  | 17.2s ★
      30 |   0.6388 |  11.56% |    5.25% | ON  | ON  | 25.5s ★
      40 |   0.6302 |  11.88% |    5.74% | ON  | ON  | 26.2s ★
      50 |   0.6196 |  13.04% |    6.52% | ON  | ON  | 22.7s ★
      60 |   0.6127 |  14.32% |    7.33% | ON  | ON  | 23.4s ★
      70 |   0.5964 |  17.07% |    8.61% | ON  | ON  | 24.3s ★
      80 |   0.5922 |  19.41% |    9.82% | ON  | ON  | 24.2s ★
      90 |   0.5720 |  22.59% |   11.36% | ON  | ON  | 24.4s ★
     100 |   0.5552 |  24.71% |   12.64% | ON  | ON  | 25.2s ★
     110 |   0.5298 |  27.68% |   14.21% | ON  | ON  | 23.3s ★
     120 |   0.5082 |  32.34% |   16.45% | ON  | ON  | 24.8s ★
     130 |   0.4747 |  35.74% |   18.34% | ON  | ON  | 22.9s ★
     140 |   0.4577 |  36.69% |   18.80% | ON  | ON  | 24.0s ★
     150 |   0.4426 |  39.87% |   20.85% | ON  | ON  | 22.4s ★
     160 |   0.4225 |  41.36% |   21.73% | ON  | ON  | 28.5s ★
     170 |   0.3944 |  45.71% |   23.91% | ON  | ON  | 23.7s ★
     180 |   0.3748 |  46.98% |   24.76% | ON  | ON  | 25.7s ★
     190 |   0.3695 |  49.31% |   26.27% | ON  | ON  | 25.3s ★
     200 |   0.3343 |  51.22% |   26.93% | ON  | ON  | 25.8s ★
     210 |   0.3197 |  52.60% |   27.76% | ON  | ON  | 25.4s ★
     220 |   0.3165 |  54.08% |   28.84% | ON  | ON  | 22.7s ★
     230 |   0.3048 |  54.93% |   29.45% | ON  | ON  | 25.4s ★
     240 |   0.2833 |  53.23% |   28.45% | ON  | ON  | 18.5s
     250 |   0.2999 |  55.67% |   29.88% | ON  | ON  | 16.0s ★
     260 |   0.2990 |  57.05% |   30.64% | ON  | ON  | 20.9s ★
     270 |   0.2735 |  58.11% |   31.20% | ON  | ON  | 18.2s ★
     280 |   0.2504 |  57.58% |   30.62% | ON  | ON  | 17.1s
     290 |   0.2627 |  58.85% |   31.29% | ON  | ON  | 16.5s ★
     300 |   0.2577 |  59.49% |   31.50% | ON  | ON  | 19.1s ★
     310 |   0.2453 |  59.17% |   31.88% | ON  | ON  | 17.6s
     320 |   0.2458 |  58.75% |   32.38% | ON  | ON  | 17.3s
     330 |   0.2579 |  55.14% |   29.60% | ON  | ON  | 18.3s
     340 |   0.2706 |  56.10% |   30.03% | ON  | ON  | 17.8s
     350 |   0.2555 |  54.93% |   30.45% | ON  | ON  | 29.0s
     360 |   0.2694 |  55.78% |   30.01% | ON  | ON  | 59.5s
     370 |   0.2588 |  56.63% |   30.67% | ON  | ON  | 52.4s
     380 |   0.2451 |  57.16% |   31.29% | ON  | ON  | 62.1s
     390 |   0.2481 |  59.49% |   32.08% | ON  | ON  | 64.9s
  [Emb saved] round= 400  →  emb_ml100k_round0400.npy
     400 |   0.2371 |  54.83% |   29.57% | ON  | ON  | 57.7s

========================================================================
  RESULT  (ML100K)
  Best HR@10:          59.49%  (round 300)
  Best NDCG@10:        31.50%
  Total training time:  177.5 min
  Avg time per round:   26.49s

  LOSS SUMMARY:
  Round 1 loss                       0.65595
  Minimum loss                       0.22721  (round 399)
  Final loss (round 400)             0.23708
  Expected range at convergence      0.45 – 0.58
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        59.49%    63.81%     -4.32%                   -6.8%
  NDCG@10      31.50%    45.03%    -13.53%                  -30.1%

  Verdict: ~ Close to paper (within 5%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   63.81%     45.03%
  Stage 5 (LDP ε=100)            59.49%     31.50%
========================================================================
  Log → stage5_log_ml100k.json
  [Names] Loaded 1682 item names from u.item
  [Names] Mapped 1682 names to model item IDs

  TOP-10 FOR USER 0  (cluster 4):
     1. Happy Gilmore (1996)                                     score=25.2771
     2. Preacher's Wife, The (1996)                              score=25.1382
     3. People vs. Larry Flynt, The (1996)                       score=25.1328
     4. Ransom (1996)                                            score=24.9603
     5. Unforgettable (1996)                                     score=24.9401
     6. Mission: Impossible (1996)                               score=24.9176
     7. Trainspotting (1996)                                     score=24.8872
     8. Speed 2: Cruise Control (1997)                           score=24.8828
     9. Sense and Sensibility (1995)                             score=24.8075
    10. Two if by Sea (1996)                                     score=24.7834

  Held-out: Terminator 2: Judgment Day (1991)
