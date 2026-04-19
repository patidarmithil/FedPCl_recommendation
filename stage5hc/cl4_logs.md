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
       1 |   0.5904 |   9.53% |    4.63% | off | ON  |  6.6s ★
      10 |   0.5741 |  12.11% |    5.91% | off | ON  |  6.5s ★
      20 |   0.5542 |  16.80% |    8.81% | off | ON  |  6.3s ★
      30 |   0.9139 |  21.85% |   12.55% | ON  | ON  | 14.7s ★
      40 |   0.8601 |  27.18% |   16.92% | ON  | ON  | 14.5s ★
      50 |   0.8463 |  32.63% |   21.43% | ON  | ON  | 16.8s ★
      60 |   0.8247 |  39.23% |   26.45% | ON  | ON  | 14.9s ★
      70 |   0.8089 |  47.19% |   32.48% | ON  | ON  | 16.3s ★
      80 |   0.7491 |  53.77% |   38.14% | ON  | ON  | 15.8s ★
      90 |   0.7465 |  58.98% |   42.08% | ON  | ON  | 14.3s ★
     100 |   0.6936 |  63.11% |   45.27% | ON  | ON  | 13.4s ★
     110 |   0.6719 |  67.13% |   48.21% | ON  | ON  | 14.1s ★
     120 |   0.6709 |  69.60% |   49.69% | ON  | ON  | 17.5s ★
     130 |   0.6086 |  72.03% |   51.15% | ON  | ON  | 13.8s ★
     140 |   0.6141 |  73.12% |   52.32% | ON  | ON  | 17.0s ★
     150 |   0.5932 |  74.34% |   52.85% | ON  | ON  | 14.9s ★
     160 |   0.5392 |  75.67% |   53.65% | ON  | ON  | 13.3s ★
     170 |   0.5648 |  76.12% |   53.99% | ON  | ON  | 14.7s ★
     180 |   0.5595 |  76.63% |   54.35% | ON  | ON  | 14.7s ★
     190 |   0.5339 |  76.90% |   54.71% | ON  | ON  | 15.1s ★
     200 |   0.5572 |  77.24% |   54.74% | ON  | ON  | 17.9s ★
     210 |   0.5220 |  77.75% |   55.11% | ON  | ON  | 15.7s ★
     220 |   0.5139 |  77.67% |   55.20% | ON  | ON  | 14.9s
     230 |   0.4875 |  77.75% |   55.34% | ON  | ON  | 13.8s
     240 |   0.5127 |  78.04% |   55.56% | ON  | ON  | 14.1s ★
     250 |   0.4881 |  78.39% |   55.67% | ON  | ON  | 14.4s ★
     260 |   0.5072 |  78.73% |   56.12% | ON  | ON  | 16.2s ★
     270 |   0.5227 |  78.65% |   55.91% | ON  | ON  | 15.5s
     280 |   0.4762 |  78.92% |   56.00% | ON  | ON  | 16.9s ★
     290 |   0.4918 |  78.89% |   56.11% | ON  | ON  | 18.1s
     300 |   0.5182 |  79.21% |   56.12% | ON  | ON  | 19.0s ★
     310 |   0.4755 |  79.13% |   56.10% | ON  | ON  | 16.4s
     320 |   0.4533 |  79.24% |   56.15% | ON  | ON  | 17.0s ★
     330 |   0.4799 |  79.40% |   56.24% | ON  | ON  | 18.0s ★
     340 |   0.4787 |  79.88% |   56.56% | ON  | ON  | 21.3s ★
     350 |   0.4720 |  79.43% |   56.45% | ON  | ON  | 64.8s
     360 |   0.4453 |  79.61% |   56.58% | ON  | ON  | 48.1s
     370 |   0.4561 |  79.96% |   56.64% | ON  | ON  | 49.2s ★
     380 |   0.4447 |  79.82% |   56.79% | ON  | ON  | 55.5s
     390 |   0.4382 |  79.90% |   56.64% | ON  | ON  | 58.9s
  [Emb saved] round= 400  →  emb_steam_round0400.npy
     400 |   0.4959 |  80.28% |   56.77% | ON  | ON  | 65.8s ★

========================================================================
  RESULT  (STEAM)
  Best HR@10:          80.28%  (round 400)
  Best NDCG@10:        56.77%
  Total training time:  138.4 min
  Avg time per round:   20.20s

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
       1 |   0.6555 |   9.12% |    3.93% | off | ON  | 10.3s ★
      10 |   0.6475 |  10.18% |    4.33% | off | ON  | 12.9s ★
      20 |   0.6402 |  10.82% |    4.85% | off | ON  | 11.9s ★
      30 |   1.1239 |  11.66% |    5.25% | ON  | ON  | 28.8s ★
      40 |   1.1200 |  11.66% |    5.54% | ON  | ON  | 28.6s
      50 |   1.1036 |  13.26% |    6.39% | ON  | ON  | 28.0s ★
      60 |   1.1008 |  14.32% |    7.10% | ON  | ON  | 28.7s ★
      70 |   1.0742 |  16.54% |    8.02% | ON  | ON  | 26.4s ★
      80 |   1.0807 |  19.09% |    9.30% | ON  | ON  | 27.1s ★
      90 |   1.0614 |  20.47% |   10.39% | ON  | ON  | 26.4s ★
     100 |   1.0511 |  23.12% |   11.63% | ON  | ON  | 27.1s ★
     110 |   1.0287 |  25.98% |   12.88% | ON  | ON  | 26.8s ★
     120 |   1.0153 |  29.06% |   14.59% | ON  | ON  | 24.0s ★
     130 |   0.9698 |  31.50% |   16.27% | ON  | ON  | 28.5s ★
     140 |   0.9751 |  34.57% |   17.49% | ON  | ON  | 27.6s ★
     150 |   0.9477 |  36.27% |   18.62% | ON  | ON  | 25.7s ★
     160 |   0.9479 |  38.81% |   20.13% | ON  | ON  | 31.8s ★
     170 |   0.9048 |  41.36% |   21.59% | ON  | ON  | 28.9s ★
     180 |   0.8865 |  44.11% |   23.01% | ON  | ON  | 30.4s ★
     190 |   0.8964 |  45.49% |   23.50% | ON  | ON  | 33.9s ★
     200 |   0.8540 |  48.67% |   25.24% | ON  | ON  | 33.9s ★
     210 |   0.8395 |  45.81% |   25.10% | ON  | ON  | 33.6s
     220 |   0.8347 |  46.55% |   25.40% | ON  | ON  | 120.6s
     230 |   0.8434 |  50.48% |   27.19% | ON  | ON  | 106.4s ★
     240 |   0.8119 |  51.01% |   26.87% | ON  | ON  | 78.7s ★
     250 |   0.8084 |  52.07% |   27.95% | ON  | ON  | 57.3s ★
     260 |   0.8480 |  51.64% |   27.69% | ON  | ON  | 103.4s
     270 |   0.8212 |  51.96% |   27.95% | ON  | ON  | 78.9s
     280 |   0.7984 |  53.13% |   28.35% | ON  | ON  | 48.3s ★
     290 |   0.7980 |  53.02% |   28.18% | ON  | ON  | 75.1s
     300 |   0.8144 |  53.98% |   28.51% | ON  | ON  | 66.4s ★
     310 |   0.7971 |  54.72% |   29.35% | ON  | ON  | 71.3s ★
     320 |   0.7830 |  55.14% |   29.33% | ON  | ON  | 63.9s ★
     330 |   0.7937 |  54.51% |   29.21% | ON  | ON  | 75.8s
     340 |   0.8002 |  53.13% |   28.14% | ON  | ON  | 78.2s
