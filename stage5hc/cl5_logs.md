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
  beta1=0.1  lam=1.0  tau=0.2

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
       1 |   0.5904 |   9.53% |    4.63% | off | ON  |  5.8s ★
      10 |   0.5741 |  12.11% |    5.91% | off | ON  |  6.9s ★
      20 |   0.5542 |  16.80% |    8.81% | off | ON  |  8.5s ★
      30 |   0.9139 |  21.85% |   12.55% | ON  | ON  | 13.7s ★
      40 |   0.8601 |  27.18% |   16.92% | ON  | ON  | 17.9s ★
      50 |   0.8463 |  32.63% |   21.43% | ON  | ON  | 18.6s ★
      60 |   0.8247 |  39.23% |   26.45% | ON  | ON  | 19.0s ★
      70 |   0.8089 |  47.19% |   32.48% | ON  | ON  | 22.0s ★
      80 |   0.7491 |  53.77% |   38.14% | ON  | ON  | 12.3s ★
      90 |   0.7465 |  58.98% |   42.08% | ON  | ON  | 13.3s ★
     100 |   0.6936 |  63.11% |   45.27% | ON  | ON  | 13.2s ★
     110 |   0.6719 |  67.13% |   48.21% | ON  | ON  | 12.4s ★
     120 |   0.6709 |  69.60% |   49.69% | ON  | ON  | 12.5s ★
     130 |   0.6086 |  72.03% |   51.15% | ON  | ON  | 11.8s ★
     140 |   0.6141 |  73.12% |   52.32% | ON  | ON  | 13.0s ★
     150 |   0.5932 |  74.34% |   52.85% | ON  | ON  | 13.2s ★
     160 |   0.5392 |  75.67% |   53.65% | ON  | ON  | 12.3s ★
     170 |   0.5648 |  76.12% |   53.99% | ON  | ON  | 12.6s ★
     180 |   0.5595 |  76.63% |   54.35% | ON  | ON  | 13.1s ★
     190 |   0.5339 |  76.90% |   54.71% | ON  | ON  | 12.5s ★
     200 |   0.5572 |  77.24% |   54.74% | ON  | ON  | 13.2s ★
     210 |   0.5220 |  77.75% |   55.11% | ON  | ON  | 15.0s ★
     220 |   0.5139 |  77.67% |   55.20% | ON  | ON  | 11.7s
     230 |   0.4875 |  77.75% |   55.34% | ON  | ON  | 15.2s
     240 |   0.5127 |  78.04% |   55.56% | ON  | ON  | 13.2s ★
     250 |   0.4881 |  78.39% |   55.67% | ON  | ON  | 12.1s ★
     260 |   0.5072 |  78.73% |   56.12% | ON  | ON  | 13.0s ★
     270 |   0.5227 |  78.65% |   55.91% | ON  | ON  | 13.2s
     280 |   0.4762 |  78.92% |   56.00% | ON  | ON  | 12.4s ★
     290 |   0.4918 |  78.89% |   56.11% | ON  | ON  | 12.6s
     300 |   0.5182 |  79.21% |   56.12% | ON  | ON  | 13.8s ★
     310 |   0.4755 |  79.13% |   56.10% | ON  | ON  | 14.0s
     320 |   0.4533 |  79.24% |   56.15% | ON  | ON  | 15.8s ★
     330 |   0.4799 |  79.40% |   56.24% | ON  | ON  | 12.4s ★
     340 |   0.4787 |  79.88% |   56.56% | ON  | ON  | 14.2s ★
     350 |   0.4720 |  79.43% |   56.45% | ON  | ON  | 11.9s
     360 |   0.4453 |  79.61% |   56.58% | ON  | ON  | 11.8s
     370 |   0.4561 |  79.96% |   56.64% | ON  | ON  | 13.7s ★
     380 |   0.4447 |  79.82% |   56.79% | ON  | ON  | 13.4s
     390 |   0.4382 |  79.90% |   56.64% | ON  | ON  | 13.1s
  [Emb saved] round= 400  →  emb_steam_round0400.npy
     400 |   0.4959 |  80.28% |   56.77% | ON  | ON  | 14.2s ★

========================================================================
  RESULT  (STEAM)
  Best HR@10:          80.28%  (round 400)
  Best NDCG@10:        56.77%
  Total training time:  90.6 min
  Avg time per round:   13.27s

  LOSS SUMMARY:
  Round 1 loss                       0.59043
  Minimum loss                       0.43109  (round 375)
  Final loss (round 400)             0.49592
  Expected range at convergence      0.01 – 0.08
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        80.28%    80.36%     -0.08%                   -0.1%
  NDCG@10      56.77%    65.55%     -8.78%                  -13.4%

  Verdict: ✓ Matches paper (within 2%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   80.36%     65.55%
  Stage 4 (no LDP)               78.84%     55.28%
  Stage 5 (LDP ε=100)            80.28%     56.77%
========================================================================
  Log → stage5_log_steam.json

  TOP-10 FOR USER 0  (cluster 0):
     1. Call of Duty Modern Warfare 3                            score=7.1231
     2. Counter-Strike Condition Zero                            score=7.1033
     3. Ricochet                                                 score=6.9951
     4. Counter-Strike Condition Zero Deleted Scenes             score=6.8358
     5. Call of Duty Modern Warfare 3 - Multiplayer              score=6.4892
     6. Day of Defeat Source                                     score=6.3959
     7. Call of Duty Black Ops - Multiplayer                     score=6.2434
     8. Call of Duty Black Ops                                   score=6.1550
     9. Infinite Crisis                                          score=6.1147
    10. Call of Duty Modern Warfare 2 - Multiplayer              score=6.0584

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
  beta1=0.1  lam=1.0  tau=0.2

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
       1 |   0.6555 |   9.12% |    3.93% | off | ON  |  8.8s ★
      10 |   0.6475 |  10.18% |    4.33% | off | ON  |  7.7s ★
      20 |   0.6402 |  10.82% |    4.85% | off | ON  |  7.8s ★
      30 |   1.1239 |  11.66% |    5.25% | ON  | ON  | 10.9s ★
      40 |   1.1200 |  11.66% |    5.54% | ON  | ON  | 13.5s
      50 |   1.1036 |  13.26% |    6.39% | ON  | ON  | 11.3s ★
      60 |   1.1008 |  14.32% |    7.10% | ON  | ON  | 13.8s ★
      70 |   1.0742 |  16.54% |    8.02% | ON  | ON  | 11.7s ★
      80 |   1.0807 |  19.09% |    9.30% | ON  | ON  | 12.0s ★
      90 |   1.0614 |  20.47% |   10.39% | ON  | ON  | 11.5s ★
     100 |   1.0511 |  23.12% |   11.63% | ON  | ON  | 10.8s ★
     110 |   1.0287 |  25.98% |   12.88% | ON  | ON  | 11.4s ★
     120 |   1.0153 |  29.06% |   14.59% | ON  | ON  | 12.2s ★
     130 |   0.9698 |  31.50% |   16.27% | ON  | ON  | 10.6s ★
     140 |   0.9751 |  34.57% |   17.49% | ON  | ON  | 13.6s ★
     150 |   0.9477 |  36.27% |   18.62% | ON  | ON  | 11.4s ★
     160 |   0.9479 |  38.81% |   20.13% | ON  | ON  | 11.7s ★
     170 |   0.9048 |  41.36% |   21.59% | ON  | ON  | 11.3s ★
     180 |   0.8865 |  44.11% |   23.01% | ON  | ON  | 13.0s ★
     190 |   0.8964 |  45.49% |   23.50% | ON  | ON  | 13.9s ★
     200 |   0.8540 |  48.67% |   25.24% | ON  | ON  | 13.5s ★
     210 |   0.8395 |  45.81% |   25.10% | ON  | ON  | 11.8s
     220 |   0.8347 |  46.55% |   25.40% | ON  | ON  | 11.8s
     230 |   0.8434 |  50.48% |   27.19% | ON  | ON  | 14.4s ★
     240 |   0.8119 |  51.01% |   26.87% | ON  | ON  | 11.8s ★
     250 |   0.8084 |  52.07% |   27.95% | ON  | ON  | 11.0s ★
     260 |   0.8480 |  51.64% |   27.69% | ON  | ON  | 12.7s
     270 |   0.8212 |  51.96% |   27.95% | ON  | ON  | 13.7s
     280 |   0.7984 |  53.13% |   28.35% | ON  | ON  | 12.0s ★
     290 |   0.7980 |  53.02% |   28.18% | ON  | ON  | 11.9s
     300 |   0.8144 |  53.98% |   28.51% | ON  | ON  | 13.5s ★
     310 |   0.7971 |  54.72% |   29.35% | ON  | ON  | 13.5s ★
     320 |   0.7830 |  55.14% |   29.33% | ON  | ON  | 12.8s ★
     330 |   0.7937 |  54.51% |   29.21% | ON  | ON  | 15.5s
     340 |   0.8002 |  53.13% |   28.14% | ON  | ON  | 12.1s
     350 |   0.8110 |  54.29% |   28.40% | ON  | ON  | 14.2s
     360 |   0.8043 |  55.14% |   29.17% | ON  | ON  | 11.5s
     370 |   0.8035 |  54.61% |   29.28% | ON  | ON  | 11.6s
     380 |   0.7797 |  55.04% |   30.03% | ON  | ON  | 11.6s
     390 |   0.8016 |  54.51% |   28.63% | ON  | ON  | 13.1s
  [Emb saved] round= 400  →  emb_ml100k_round0400.npy
     400 |   0.7907 |  53.45% |   28.92% | ON  | ON  | 13.1s

========================================================================
  RESULT  (ML100K)
  Best HR@10:          55.14%  (round 320)
  Best NDCG@10:        29.33%
  Total training time:  82.2 min
  Avg time per round:   12.27s

  LOSS SUMMARY:
  Round 1 loss                       0.65547
  Minimum loss                       0.64024  (round 20)
  Final loss (round 400)             0.79075
  Expected range at convergence      0.45 – 0.58
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        55.14%    63.81%     -8.67%                  -13.6%
  NDCG@10      29.33%    45.03%    -15.70%                  -34.9%

  Verdict: ✗ Below paper (gap > 5%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   63.81%     45.03%
  Stage 5 (LDP ε=100)            55.14%     29.33%
========================================================================
  Log → stage5_log_ml100k.json
  [Names] Loaded 1682 item names from u.item
  [Names] Mapped 1682 names to model item IDs

  TOP-10 FOR USER 0  (cluster 1):
     1. People vs. Larry Flynt, The (1996)                       score=28.3902
     2. Ransom (1996)                                            score=28.2244
     3. Stand by Me (1986)                                       score=28.0584
     4. Volcano (1997)                                           score=28.0191
     5. Secrets & Lies (1996)                                    score=27.9492
     6. Man Without a Face, The (1993)                           score=27.9484
     7. Congo (1995)                                             score=27.9032
     8. Third Man, The (1949)                                    score=27.9002
     9. Flirting With Disaster (1996)                            score=27.8791
    10. Treasure of the Sierra Madre, The (1948)                 score=27.8695

  Held-out: Terminator 2: Judgment Day (1991)
