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
  beta1=0.1  lam=0.0  tau=0.2

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
       1 |   0.5890 |   9.48% |    4.64% | off | ON  | 10.7s ★
      10 |   0.5720 |  11.31% |    5.56% | off | ON  | 10.6s ★
      20 |   0.5512 |  14.91% |    7.68% | off | ON  | 11.0s ★
      30 |   0.5742 |  20.15% |   11.31% | ON  | ON  | 15.9s ★
      40 |   0.5442 |  26.19% |   15.91% | ON  | ON  | 17.4s ★
      50 |   0.5269 |  31.78% |   20.61% | ON  | ON  | 15.5s ★
      60 |   0.4899 |  38.73% |   25.99% | ON  | ON  | 14.0s ★
      70 |   0.4689 |  46.31% |   32.06% | ON  | ON  | 14.0s ★
      80 |   0.4198 |  53.23% |   37.85% | ON  | ON  | 13.6s ★
      90 |   0.4072 |  58.40% |   41.88% | ON  | ON  | 14.8s ★
     100 |   0.3632 |  63.08% |   45.40% | ON  | ON  | 16.1s ★
     110 |   0.3239 |  66.33% |   47.56% | ON  | ON  | 14.6s ★
     120 |   0.3201 |  69.58% |   49.63% | ON  | ON  | 15.9s ★
     130 |   0.2769 |  71.84% |   51.06% | ON  | ON  | 13.6s ★
     140 |   0.2778 |  73.25% |   52.06% | ON  | ON  | 16.0s ★
     150 |   0.2663 |  74.61% |   52.90% | ON  | ON  | 16.3s ★
     160 |   0.2212 |  75.49% |   53.61% | ON  | ON  | 12.8s ★
     170 |   0.2372 |  76.20% |   54.19% | ON  | ON  | 15.5s ★
     180 |   0.2305 |  76.31% |   54.42% | ON  | ON  | 16.2s ★
     190 |   0.2051 |  77.00% |   54.75% | ON  | ON  | 15.6s ★
     200 |   0.2199 |  76.90% |   54.95% | ON  | ON  | 17.9s
     210 |   0.1982 |  77.22% |   54.91% | ON  | ON  | 18.1s ★
     220 |   0.1927 |  77.99% |   55.15% | ON  | ON  | 17.2s ★
     230 |   0.1753 |  78.23% |   55.54% | ON  | ON  | 16.4s ★
     240 |   0.1857 |  78.15% |   55.56% | ON  | ON  | 16.2s
     250 |   0.1633 |  78.09% |   55.73% | ON  | ON  | 16.5s
     260 |   0.1839 |  78.79% |   55.94% | ON  | ON  | 17.4s ★
     270 |   0.1919 |  79.11% |   56.12% | ON  | ON  | 18.0s ★
     280 |   0.1658 |  78.89% |   56.08% | ON  | ON  | 16.5s
     290 |   0.1729 |  78.73% |   55.90% | ON  | ON  | 18.6s
     300 |   0.1893 |  79.11% |   55.99% | ON  | ON  | 18.3s
     310 |   0.1650 |  78.89% |   55.99% | ON  | ON  | 18.9s
     320 |   0.1411 |  79.16% |   56.21% | ON  | ON  | 16.2s ★
     330 |   0.1671 |  79.56% |   56.38% | ON  | ON  | 16.0s ★
     340 |   0.1618 |  79.58% |   56.48% | ON  | ON  | 17.7s ★
     350 |   0.1522 |  79.64% |   56.61% | ON  | ON  | 15.4s ★
     360 |   0.1353 |  79.85% |   56.78% | ON  | ON  | 14.6s ★
     370 |   0.1449 |  79.61% |   56.74% | ON  | ON  | 15.4s
     380 |   0.1293 |  79.80% |   56.89% | ON  | ON  | 13.9s
     390 |   0.1274 |  79.88% |   56.93% | ON  | ON  | 15.3s ★
^[[A^[[B  [Emb saved] round= 400  →  emb_steam_round0400.npy
     400 |   0.1583 |  79.96% |   57.10% | ON  | ON  | 47.8s ★

========================================================================
  RESULT  (STEAM)
  Best HR@10:          79.96%  (round 400)
  Best NDCG@10:        57.10%
  Total training time:  112.9 min
  Avg time per round:   16.36s

  LOSS SUMMARY:
  Round 1 loss                       0.58900
  Minimum loss                       0.12679  (round 375)
  Final loss (round 400)             0.15832
  Expected range at convergence      0.01 – 0.08
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        79.96%    80.36%     -0.40%                   -0.5%
  NDCG@10      57.10%    65.55%     -8.45%                  -12.9%

  Verdict: ✓ Matches paper (within 2%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   80.36%     65.55%
  Stage 4 (no LDP)               78.84%     55.28%
  Stage 5 (LDP ε=100)            79.96%     57.10%
========================================================================
  Log → stage5_log_steam.json

  TOP-10 FOR USER 0  (cluster 0):
     1. Counter-Strike Condition Zero                            score=7.5521
     2. Ricochet                                                 score=7.5058
     3. Counter-Strike Condition Zero Deleted Scenes             score=7.1637
     4. Call of Duty Modern Warfare 3                            score=6.9860
     5. Call of Duty Modern Warfare 3 - Multiplayer              score=6.9673
     6. Day of Defeat Source                                     score=6.7088
     7. Infinite Crisis                                          score=6.3862
     8. Call of Duty Black Ops - Multiplayer                     score=6.3436
     9. Call of Duty Modern Warfare 2 - Multiplayer              score=6.3138
    10. Serious Sam HD The Second Encounter                      score=6.2767

  Held-out: Ricochet



eed] 42
[Device] cuda
[LDP] σ=0.1  λ=0.001  ε=100.0
========================================================================
  Stage 5: FedPCL + LDP — ML100K
========================================================================
  device=cuda  d=64  K_gnn=2
  rounds=400  clients/round=128
  local_epochs=10  lr_item=0.1
  clusters K=5  mu1=0.5  mu2=0.5
  beta1=0.1  lam=0.0  tau=0.2

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
       1 |   0.6560 |   9.01% |    3.89% | off | ON  | 19.7s ★
      10 |   0.6477 |   9.97% |    4.31% | off | ON  | 23.3s ★
      20 |   0.6403 |  10.60% |    4.79% | off | ON  | 21.3s ★
      30 |   0.6605 |  11.45% |    5.22% | ON  | ON  | 30.0s ★
      40 |   0.6518 |  11.88% |    5.72% | ON  | ON  | 31.7s ★
      50 |   0.6412 |  12.83% |    6.43% | ON  | ON  | 28.9s ★
      60 |   0.6341 |  14.42% |    7.22% | ON  | ON  | 31.1s ★
      70 |   0.6192 |  16.44% |    8.18% | ON  | ON  | 28.5s ★
      80 |   0.6157 |  18.13% |    9.16% | ON  | ON  | 32.7s ★
      90 |   0.5953 |  20.68% |   10.44% | ON  | ON  | 29.5s ★
     100 |   0.5816 |  22.91% |   11.56% | ON  | ON  | 30.6s ★
     110 |   0.5637 |  25.45% |   13.20% | ON  | ON  | 30.6s ★
     120 |   0.5380 |  28.10% |   14.46% | ON  | ON  | 31.5s ★
     130 |   0.5163 |  32.03% |   16.33% | ON  | ON  | 27.3s ★
     140 |   0.4958 |  33.83% |   16.90% | ON  | ON  | 32.2s ★
     150 |   0.4806 |  37.01% |   18.85% | ON  | ON  | 30.7s ★
     160 |   0.4617 |  41.25% |   20.86% | ON  | ON  | 32.9s ★
     170 |   0.4279 |  42.21% |   21.42% | ON  | ON  | 30.7s ★
     180 |   0.4184 |  45.17% |   23.10% | ON  | ON  | 31.1s ★
     190 |   0.4012 |  47.08% |   23.91% | ON  | ON  | 31.6s ★
     200 |   0.3691 |  48.14% |   24.91% | ON  | ON  | 31.4s ★
     210 |   0.3609 |  48.78% |   25.45% | ON  | ON  | 30.0s ★
     220 |   0.3605 |  50.16% |   25.94% | ON  | ON  | 66.7s ★
     230 |   0.3398 |  52.28% |   26.85% | ON  | ON  | 72.8s ★
     240 |   0.3196 |  52.49% |   27.31% | ON  | ON  | 29.4s ★
     250 |   0.3116 |  51.54% |   27.17% | ON  | ON  | 74.1s
     260 |   0.3427 |  53.02% |   27.86% | ON  | ON  | 52.2s ★
     270 |   0.3097 |  54.61% |   28.26% | ON  | ON  | 59.0s ★
     280 |   0.2899 |  53.87% |   28.38% | ON  | ON  | 70.1s
     290 |   0.2818 |  51.64% |   27.84% | ON  | ON  | 62.2s
     300 |   0.3109 |  54.83% |   28.72% | ON  | ON  | 70.8s ★
     310 |   0.2789 |  53.66% |   27.40% | ON  | ON  | 23.3s
     320 |   0.2851 |  54.29% |   28.11% | ON  | ON  | 20.2s
     330 |   0.2864 |  54.83% |   28.49% | ON  | ON  | 24.3s
     340 |   0.2768 |  54.08% |   29.09% | ON  | ON  | 22.6s
     350 |   0.2935 |  53.13% |   27.25% | ON  | ON  | 23.5s
     360 |   0.2685 |  51.86% |   27.75% | ON  | ON  | 22.6s
     370 |   0.2881 |  51.64% |   28.12% | ON  | ON  | 24.2s
     380 |   0.2723 |  51.96% |   27.71% | ON  | ON  | 22.8s
     390 |   0.2970 |  53.02% |   28.12% | ON  | ON  | 24.4s
  [Emb saved] round= 400  →  emb_ml100k_round0400.npy
     400 |   0.2860 |  51.01% |   27.47% | ON  | ON  | 21.2s

========================================================================
  RESULT  (ML100K)
  Best HR@10:          54.83%  (round 300)
  Best NDCG@10:        28.72%
  Total training time:  244.6 min
  Avg time per round:   36.53s

  LOSS SUMMARY:
  Round 1 loss                       0.65595
  Minimum loss                       0.26314  (round 339)
  Final loss (round 400)             0.28599
  Expected range at convergence      0.45 – 0.58
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        54.83%    63.81%     -8.98%                  -14.1%
  NDCG@10      28.72%    45.03%    -16.31%                  -36.2%

  Verdict: ✗ Below paper (gap > 5%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   63.81%     45.03%
  Stage 5 (LDP ε=100)            54.83%     28.72%
========================================================================
  Log → stage5_log_ml100k.json
  [Names] Loaded 1682 item names from u.item
  [Names] Mapped 1682 names to model item IDs

  TOP-10 FOR USER 0  (cluster 1):
     1. People vs. Larry Flynt, The (1996)                       score=23.9387
     2. Ransom (1996)                                            score=23.5937
     3. English Patient, The (1996)                              score=23.5762
     4. Trainspotting (1996)                                     score=23.5213
     5. City Slickers II: The Legend of Curly's Gold (1994)      score=23.3815
     6. Anastasia (1997)                                         score=23.3511
     7. Beautician and the Beast, The (1997)                     score=23.3499
     8. Vegas Vacation (1997)                                    score=23.3273
     9. Vertigo (1958)                                           score=23.3147
    10. Piano, The (1993)                                        score=23.3055

  Held-out: Terminator 2: Judgment Day (1991)


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
  beta1=0.05  lam=0.5  tau=0.2

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
       1 |   0.6560 |   9.01% |    3.89% | off | ON  | 14.5s ★
      10 |   0.6477 |   9.97% |    4.31% | off | ON  | 18.5s ★
      20 |   0.6403 |  10.60% |    4.79% | off | ON  | 20.9s ★
      30 |   0.6855 |  11.45% |    5.22% | ON  | ON  | 32.9s ★
      40 |   0.6777 |  11.77% |    5.68% | ON  | ON  | 31.7s ★
      50 |   0.6665 |  13.36% |    6.57% | ON  | ON  | 33.8s ★
      60 |   0.6602 |  13.89% |    7.09% | ON  | ON  | 31.0s ★
      70 |   0.6436 |  16.12% |    8.27% | ON  | ON  | 29.2s ★
      80 |   0.6429 |  18.77% |    9.57% | ON  | ON  | 31.8s ★
      90 |   0.6230 |  20.78% |   10.37% | ON  | ON  | 30.1s ★
     100 |   0.6123 |  23.65% |   11.90% | ON  | ON  | 29.3s ★
     110 |   0.5893 |  26.19% |   13.38% | ON  | ON  | 31.9s ★
     120 |   0.5690 |  28.74% |   14.80% | ON  | ON  | 31.4s ★
     130 |   0.5380 |  31.81% |   16.41% | ON  | ON  | 27.3s ★
     140 |   0.5318 |  35.31% |   18.39% | ON  | ON  | 32.5s ★
     150 |   0.5073 |  38.28% |   19.56% | ON  | ON  | 30.3s ★
     160 |   0.5015 |  39.34% |   20.23% | ON  | ON  | 33.3s ★
     170 |   0.4728 |  41.99% |   21.56% | ON  | ON  | 30.0s ★
     180 |   0.4588 |  43.80% |   22.76% | ON  | ON  | 30.0s ★
     190 |   0.4492 |  46.77% |   24.13% | ON  | ON  | 31.1s ★
     200 |   0.4170 |  47.72% |   24.57% | ON  | ON  | 33.4s ★
     210 |   0.4072 |  49.10% |   25.45% | ON  | ON  | 29.8s ★
     220 |   0.3942 |  48.57% |   25.66% | ON  | ON  | 68.8s
     230 |   0.4016 |  51.54% |   26.81% | ON  | ON  | 72.8s ★
     240 |   0.3776 |  51.54% |   27.78% | ON  | ON  | 29.1s
     250 |   0.3696 |  53.34% |   28.69% | ON  | ON  | 68.7s ★
     260 |   0.3811 |  53.23% |   28.24% | ON  | ON  | 54.2s
     270 |   0.3680 |  53.13% |   27.59% | ON  | ON  | 64.5s
     280 |   0.3556 |  55.14% |   29.43% | ON  | ON  | 64.9s ★
     290 |   0.3471 |  53.87% |   28.68% | ON  | ON  | 61.7s
     300 |   0.3606 |  54.08% |   28.57% | ON  | ON  | 71.9s
     310 |   0.3447 |  54.61% |   28.67% | ON  | ON  | 23.3s
     320 |   0.3337 |  54.93% |   28.88% | ON  | ON  | 23.3s
     330 |   0.3477 |  54.83% |   29.39% | ON  | ON  | 25.0s
     340 |   0.3422 |  54.61% |   29.42% | ON  | ON  | 23.1s
     350 |   0.3430 |  53.87% |   29.21% | ON  | ON  | 23.5s
     360 |   0.3423 |  54.51% |   29.42% | ON  | ON  | 22.8s
     370 |   0.3362 |  53.45% |   28.78% | ON  | ON  | 25.2s
     380 |   0.3399 |  54.08% |   29.16% | ON  | ON  | 24.4s
     390 |   0.3347 |  52.92% |   28.09% | ON  | ON  | 25.0s
  [Emb saved] round= 400  →  emb_ml100k_round0400.npy
     400 |   0.3300 |  52.28% |   27.38% | ON  | ON  | 19.9s

========================================================================
  RESULT  (ML100K)
  Best HR@10:          55.14%  (round 280)
  Best NDCG@10:        29.43%
  Total training time:  245.1 min
  Avg time per round:   36.60s

  LOSS SUMMARY:
  Round 1 loss                       0.65595
  Minimum loss                       0.32085  (round 364)
  Final loss (round 400)             0.32998
  Expected range at convergence      0.45 – 0.58
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        55.14%    63.81%     -8.67%                  -13.6%
  NDCG@10      29.43%    45.03%    -15.60%                  -34.6%

  Verdict: ✗ Below paper (gap > 5%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   63.81%     45.03%
  Stage 5 (LDP ε=100)            55.14%     29.43%
========================================================================
  Log → stage5_log_ml100k.json
  [Names] Loaded 1682 item names from u.item
  [Names] Mapped 1682 names to model item IDs

  TOP-10 FOR USER 0  (cluster 3):
     1. People vs. Larry Flynt, The (1996)                       score=21.3398
     2. Ransom (1996)                                            score=21.0906
     3. English Patient, The (1996)                              score=21.0274
     4. Time to Kill, A (1996)                                   score=20.7812
     5. Volcano (1997)                                           score=20.7262
     6. Trainspotting (1996)                                     score=20.6859
     7. Sense and Sensibility (1995)                             score=20.6282
     8. Birds, The (1963)                                        score=20.6225
     9. Titanic (1997)                                           score=20.6170
    10. Vertigo (1958)                                           score=20.6047

  Held-out: Terminator 2: Judgment Day (1991)

