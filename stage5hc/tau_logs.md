for TAU in 0.15 0.20 0.25 0.30 0.35; do
  python3 train_stage5.py --dataset ml100k --data_path u.data --tau $TAU
  python3 train_stage5.py --dataset steam  --data_path steam_processed.json --tau $TAU
done
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
  beta1=0.1  lam=1.0  tau=0.15

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
       1 |   0.6560 |   9.01% |    3.89% | off | ON  | 21.9s ★
      10 |   0.6477 |   9.97% |    4.31% | off | ON  | 17.5s ★
      20 |   0.6403 |  10.60% |    4.79% | off | ON  | 18.7s ★
      30 |   0.7277 |  11.66% |    5.27% | ON  | ON  | 25.2s ★
      40 |   0.7213 |  11.66% |    5.56% | ON  | ON  | 25.1s
      50 |   0.7090 |  12.83% |    6.36% | ON  | ON  | 26.9s ★
      60 |   0.7048 |  13.47% |    6.90% | ON  | ON  | 28.4s ★
      70 |   0.6856 |  15.48% |    7.92% | ON  | ON  | 27.6s ★
      80 |   0.6895 |  17.82% |    9.05% | ON  | ON  | 35.0s ★
      90 |   0.6700 |  20.57% |   10.60% | ON  | ON  | 31.5s ★
     100 |   0.6617 |  22.48% |   11.36% | ON  | ON  | 33.0s ★
     110 |   0.6477 |  24.18% |   12.28% | ON  | ON  | 31.8s ★
     120 |   0.6345 |  27.57% |   14.00% | ON  | ON  | 31.1s ★
     130 |   0.6010 |  29.69% |   15.00% | ON  | ON  | 28.9s ★
     140 |   0.6043 |  32.13% |   16.64% | ON  | ON  | 25.0s ★
     150 |   0.5841 |  35.31% |   18.25% | ON  | ON  | 22.6s ★
     160 |   0.5813 |  36.69% |   18.93% | ON  | ON  | 25.0s ★
     170 |   0.5497 |  38.60% |   19.84% | ON  | ON  | 23.5s ★
     180 |   0.5443 |  42.42% |   21.77% | ON  | ON  | 23.3s ★
     190 |   0.5366 |  44.33% |   22.66% | ON  | ON  | 25.5s ★
     200 |   0.5142 |  46.87% |   23.96% | ON  | ON  | 23.1s ★
     210 |   0.4984 |  45.49% |   23.56% | ON  | ON  | 22.3s
     220 |   0.4978 |  48.78% |   25.07% | ON  | ON  | 22.2s ★
     230 |   0.4937 |  48.25% |   25.51% | ON  | ON  | 23.5s
     240 |   0.4845 |  49.20% |   25.99% | ON  | ON  | 21.7s ★
     250 |   0.4806 |  50.16% |   26.80% | ON  | ON  | 22.9s ★
     260 |   0.5014 |  51.75% |   26.95% | ON  | ON  | 25.3s ★
     270 |   0.4851 |  52.60% |   27.71% | ON  | ON  | 23.7s ★
     280 |   0.4648 |  51.96% |   28.07% | ON  | ON  | 23.3s
     290 |   0.4689 |  53.87% |   28.22% | ON  | ON  | 22.8s ★
     300 |   0.4845 |  52.81% |   27.83% | ON  | ON  | 22.9s
     310 |   0.4690 |  52.70% |   27.90% | ON  | ON  | 25.3s
     320 |   0.4521 |  54.61% |   28.46% | ON  | ON  | 21.2s ★
     330 |   0.4753 |  54.40% |   28.44% | ON  | ON  | 24.9s
     340 |   0.4747 |  53.02% |   27.97% | ON  | ON  | 23.4s
     350 |   0.4848 |  52.60% |   27.89% | ON  | ON  | 24.5s
     360 |   0.4802 |  51.96% |   27.70% | ON  | ON  | 24.3s
     370 |   0.4872 |  51.54% |   27.84% | ON  | ON  | 26.8s
     380 |   0.4806 |  53.45% |   28.50% | ON  | ON  | 26.7s
     390 |   0.4891 |  52.39% |   28.15% | ON  | ON  | 24.0s
  [Emb saved] round= 400  →  emb_ml100k_round0400.npy
     400 |   0.4794 |  52.49% |   27.86% | ON  | ON  | 22.8s

========================================================================
  RESULT  (ML100K)
  Best HR@10:          54.61%  (round 320)
  Best NDCG@10:        28.46%
  Total training time:  164.1 min
  Avg time per round:   24.51s

  LOSS SUMMARY:
  Round 1 loss                       0.65595
  Minimum loss                       0.44176  (round 302)
  Final loss (round 400)             0.47938
  Expected range at convergence      0.45 – 0.58
  Convergence status                 within expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        54.61%    63.81%     -9.20%                  -14.4%
  NDCG@10      28.46%    45.03%    -16.57%                  -36.8%

  Verdict: ✗ Below paper (gap > 5%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   63.81%     45.03%
  Stage 5 (LDP ε=100)            54.61%     28.46%
========================================================================
  Log → stage5_log_ml100k.json
  [Names] Loaded 1682 item names from u.item
  [Names] Mapped 1682 names to model item IDs

  TOP-10 FOR USER 0  (cluster 2):
     1. Ransom (1996)                                            score=18.0983
     2. People vs. Larry Flynt, The (1996)                       score=17.9216
     3. English Patient, The (1996)                              score=17.8224
     4. Vertigo (1958)                                           score=17.7390
     5. My Life as a Dog (Mitt liv som hund) (1985)              score=17.6536
     6. Annie Hall (1977)                                        score=17.6069
     7. Schindler's List (1993)                                  score=17.5966
     8. Preacher's Wife, The (1996)                              score=17.5832
     9. Dr. Strangelove or: How I Learned to Stop Worrying and   score=17.5646
    10. Birds, The (1963)                                        score=17.5448

  Held-out: Terminator 2: Judgment Day (1991)
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
  beta1=0.1  lam=1.0  tau=0.15

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
      10 |   0.5720 |  11.31% |    5.56% | off | ON  |  7.1s ★
      20 |   0.5512 |  14.91% |    7.68% | off | ON  |  8.0s ★
      30 |   0.5914 |  20.31% |   11.07% | ON  | ON  | 13.2s ★
      40 |   0.5565 |  25.05% |   15.01% | ON  | ON  | 13.9s ★
      50 |   0.5452 |  30.29% |   19.31% | ON  | ON  | 12.4s ★
      60 |   0.5147 |  36.73% |   24.48% | ON  | ON  | 12.3s ★
      70 |   0.5004 |  43.71% |   29.86% | ON  | ON  | 13.8s ★
      80 |   0.4552 |  50.44% |   35.14% | ON  | ON  | 12.6s ★
      90 |   0.4516 |  55.10% |   39.07% | ON  | ON  | 15.4s ★
     100 |   0.4071 |  60.02% |   42.61% | ON  | ON  | 13.5s ★
     110 |   0.3755 |  63.45% |   45.32% | ON  | ON  | 12.7s ★
     120 |   0.3821 |  66.68% |   47.47% | ON  | ON  | 14.2s ★
     130 |   0.3293 |  69.55% |   49.55% | ON  | ON  | 13.8s ★
     140 |   0.3344 |  71.44% |   50.61% | ON  | ON  | 14.7s ★
     150 |   0.3238 |  72.66% |   51.41% | ON  | ON  | 13.8s ★
     160 |   0.2763 |  73.86% |   52.26% | ON  | ON  | 12.0s ★
     170 |   0.2954 |  74.74% |   52.85% | ON  | ON  | 13.1s ★
     180 |   0.2909 |  75.43% |   53.30% | ON  | ON  | 14.6s ★
     190 |   0.2669 |  75.91% |   53.76% | ON  | ON  | 14.9s ★
     200 |   0.2882 |  76.28% |   53.72% | ON  | ON  | 15.4s ★
     210 |   0.2603 |  76.28% |   54.17% | ON  | ON  | 15.1s
     220 |   0.2527 |  76.82% |   54.37% | ON  | ON  | 13.5s ★
     230 |   0.2301 |  77.11% |   54.53% | ON  | ON  | 13.8s ★
     240 |   0.2460 |  77.35% |   54.53% | ON  | ON  | 15.1s ★
     250 |   0.2263 |  77.32% |   54.91% | ON  | ON  | 13.2s
     260 |   0.2460 |  77.38% |   55.10% | ON  | ON  | 14.2s ★
     270 |   0.2560 |  78.09% |   55.39% | ON  | ON  | 15.2s ★
     280 |   0.2273 |  78.09% |   55.35% | ON  | ON  | 12.9s
     290 |   0.2354 |  78.28% |   55.41% | ON  | ON  | 15.3s ★
     300 |   0.2590 |  78.17% |   55.33% | ON  | ON  | 16.1s
     310 |   0.2239 |  78.44% |   55.45% | ON  | ON  | 12.6s ★
     320 |   0.2035 |  78.44% |   55.30% | ON  | ON  | 12.3s
     330 |   0.2270 |  78.71% |   55.46% | ON  | ON  | 11.5s ★
     340 |   0.2260 |  78.79% |   55.55% | ON  | ON  | 11.1s ★
     350 |   0.2138 |  78.89% |   55.80% | ON  | ON  | 11.0s ★
     360 |   0.1959 |  78.87% |   55.64% | ON  | ON  | 11.1s
     370 |   0.2065 |  78.95% |   55.53% | ON  | ON  | 11.8s ★
     380 |   0.1897 |  79.16% |   55.74% | ON  | ON  | 10.2s ★
     390 |   0.1906 |  79.27% |   55.67% | ON  | ON  | 12.8s ★
  [Emb saved] round= 400  →  emb_steam_round0400.npy
     400 |   0.2226 |  79.24% |   55.57% | ON  | ON  | 12.0s

========================================================================
  RESULT  (STEAM)
  Best HR@10:          79.27%  (round 390)
  Best NDCG@10:        55.67%
  Total training time:  89.2 min
  Avg time per round:   13.00s

  LOSS SUMMARY:
  Round 1 loss                       0.58900
  Minimum loss                       0.18397  (round 375)
  Final loss (round 400)             0.22260
  Expected range at convergence      0.01 – 0.08
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        79.27%    80.36%     -1.09%                   -1.4%
  NDCG@10      55.67%    65.55%     -9.88%                  -15.1%

  Verdict: ✓ Matches paper (within 2%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   80.36%     65.55%
  Stage 4 (no LDP)               78.84%     55.28%
  Stage 5 (LDP ε=100)            79.27%     55.67%
========================================================================
  Log → stage5_log_steam.json

  TOP-10 FOR USER 0  (cluster 1):
     1. Counter-Strike Condition Zero                            score=5.7095
     2. Call of Duty Modern Warfare 3                            score=5.7032
     3. Call of Duty Modern Warfare 3 - Multiplayer              score=5.5216
     4. Ricochet                                                 score=5.4753
     5. Counter-Strike Condition Zero Deleted Scenes             score=5.3360
     6. Call of Duty Black Ops - Multiplayer                     score=5.3066
     7. Serious Sam HD The Second Encounter                      score=5.2909
     8. Call of Duty Modern Warfare 2 - Multiplayer              score=5.1821
     9. Day of Defeat Source                                     score=5.0537
    10. Insurgency Modern Infantry Combat                        score=5.0376

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
       1 |   0.6560 |   9.01% |    3.89% | off | ON  | 15.3s ★
      10 |   0.6477 |   9.97% |    4.31% | off | ON  | 12.4s ★
      20 |   0.6403 |  10.60% |    4.79% | off | ON  | 11.0s ★
      30 |   0.8028 |  11.35% |    5.18% | ON  | ON  | 18.3s ★
      40 |   0.7968 |  11.35% |    5.43% | ON  | ON  | 18.4s
      50 |   0.7830 |  12.41% |    6.16% | ON  | ON  | 18.9s ★
      60 |   0.7789 |  13.47% |    6.79% | ON  | ON  | 17.2s ★
      70 |   0.7574 |  15.69% |    7.70% | ON  | ON  | 16.5s ★
      80 |   0.7646 |  16.65% |    8.49% | ON  | ON  | 17.3s ★
      90 |   0.7472 |  19.19% |    9.87% | ON  | ON  | 17.3s ★
     100 |   0.7410 |  21.31% |   10.62% | ON  | ON  | 17.3s ★
     110 |   0.7241 |  23.65% |   12.05% | ON  | ON  | 17.4s ★
     120 |   0.7091 |  25.66% |   13.13% | ON  | ON  | 19.0s ★
     130 |   0.6773 |  27.78% |   14.00% | ON  | ON  | 18.0s ★
     140 |   0.6834 |  31.92% |   16.59% | ON  | ON  | 19.3s ★
     150 |   0.6589 |  33.40% |   16.74% | ON  | ON  | 18.2s ★
     160 |   0.6653 |  39.13% |   20.08% | ON  | ON  | 18.6s ★
     170 |   0.6212 |  39.13% |   19.68% | ON  | ON  | 16.7s
     180 |   0.6172 |  42.84% |   21.81% | ON  | ON  | 17.2s ★
     190 |   0.6127 |  44.96% |   22.62% | ON  | ON  | 19.5s ★
     200 |   0.5819 |  45.92% |   23.41% | ON  | ON  | 18.5s ★
     210 |   0.5663 |  47.51% |   24.53% | ON  | ON  | 16.7s ★
     220 |   0.5680 |  45.71% |   23.78% | ON  | ON  | 17.1s
     230 |   0.5794 |  49.52% |   25.47% | ON  | ON  | 18.2s ★
     240 |   0.5462 |  48.67% |   25.67% | ON  | ON  | 18.2s
     250 |   0.5525 |  50.05% |   26.45% | ON  | ON  | 16.2s ★
     260 |   0.5744 |  51.75% |   26.44% | ON  | ON  | 18.4s ★
     270 |   0.5448 |  51.75% |   27.12% | ON  | ON  | 18.2s
     280 |   0.5376 |  53.02% |   27.78% | ON  | ON  | 18.6s ★
     290 |   0.5399 |  52.39% |   27.43% | ON  | ON  | 16.3s
     300 |   0.5530 |  54.08% |   28.33% | ON  | ON  | 19.0s ★
     310 |   0.5435 |  51.96% |   27.97% | ON  | ON  | 17.2s
     320 |   0.5290 |  53.98% |   28.52% | ON  | ON  | 18.2s
     330 |   0.5354 |  54.19% |   28.91% | ON  | ON  | 17.3s ★
     340 |   0.5388 |  51.64% |   27.47% | ON  | ON  | 17.5s
     350 |   0.5484 |  52.92% |   27.67% | ON  | ON  | 19.0s
     360 |   0.5334 |  52.70% |   27.88% | ON  | ON  | 18.4s
     370 |   0.5470 |  51.54% |   27.71% | ON  | ON  | 18.5s
     380 |   0.5357 |  51.54% |   26.40% | ON  | ON  | 17.8s
     390 |   0.5620 |  52.70% |   28.23% | ON  | ON  | 17.2s
  [Emb saved] round= 400  →  emb_ml100k_round0400.npy
     400 |   0.5407 |  51.33% |   27.46% | ON  | ON  | 18.2s

========================================================================
  RESULT  (ML100K)
  Best HR@10:          54.19%  (round 330)
  Best NDCG@10:        28.91%
  Total training time:  116.2 min
  Avg time per round:   17.36s

  LOSS SUMMARY:
  Round 1 loss                       0.65595
  Minimum loss                       0.51461  (round 349)
  Final loss (round 400)             0.54068
  Expected range at convergence      0.45 – 0.58
  Convergence status                 within expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        54.19%    63.81%     -9.62%                  -15.1%
  NDCG@10      28.91%    45.03%    -16.12%                  -35.8%

  Verdict: ✗ Below paper (gap > 5%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   63.81%     45.03%
  Stage 5 (LDP ε=100)            54.19%     28.91%
========================================================================
  Log → stage5_log_ml100k.json
  [Names] Loaded 1682 item names from u.item
  [Names] Mapped 1682 names to model item IDs

  TOP-10 FOR USER 0  (cluster 3):
     1. People vs. Larry Flynt, The (1996)                       score=17.3412
     2. Ransom (1996)                                            score=16.8378
     3. Titanic (1997)                                           score=16.5856
     4. Trainspotting (1996)                                     score=16.5043
     5. Vertigo (1958)                                           score=16.4936
     6. Mission: Impossible (1996)                               score=16.4907
     7. Sense and Sensibility (1995)                             score=16.2977
     8. Sabrina (1995)                                           score=16.2783
     9. Schindler's List (1993)                                  score=16.2530
    10. One Flew Over the Cuckoo's Nest (1975)                   score=16.2004

  Held-out: Terminator 2: Judgment Day (1991)
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
       1 |   0.5890 |   9.48% |    4.64% | off | ON  |  5.9s ★
      10 |   0.5720 |  11.31% |    5.56% | off | ON  |  5.4s ★
      20 |   0.5512 |  14.91% |    7.68% | off | ON  |  6.0s ★
      30 |   0.6363 |  19.83% |   10.93% | ON  | ON  | 11.0s ★
      40 |   0.5990 |  24.83% |   14.76% | ON  | ON  | 11.2s ★
      50 |   0.5883 |  29.86% |   19.11% | ON  | ON  | 10.6s ★
      60 |   0.5573 |  35.99% |   23.69% | ON  | ON  |  9.8s ★
      70 |   0.5497 |  42.61% |   29.20% | ON  | ON  | 12.1s ★
      80 |   0.5025 |  48.92% |   34.32% | ON  | ON  | 12.2s ★
      90 |   0.5003 |  54.75% |   38.89% | ON  | ON  | 11.5s ★
     100 |   0.4533 |  58.88% |   42.06% | ON  | ON  | 10.4s ★
     110 |   0.4257 |  62.71% |   44.95% | ON  | ON  | 10.9s ★
     120 |   0.4330 |  65.80% |   46.99% | ON  | ON  | 10.8s ★
     130 |   0.3762 |  69.10% |   48.95% | ON  | ON  | 11.2s ★
     140 |   0.3871 |  71.31% |   50.43% | ON  | ON  | 12.0s ★
     150 |   0.3727 |  72.58% |   51.43% | ON  | ON  | 11.3s ★
     160 |   0.3226 |  73.65% |   52.23% | ON  | ON  | 10.3s ★
     170 |   0.3447 |  74.71% |   52.94% | ON  | ON  | 11.4s ★
     180 |   0.3404 |  75.11% |   53.31% | ON  | ON  | 10.9s ★
     190 |   0.3136 |  75.33% |   53.60% | ON  | ON  | 10.5s ★
     200 |   0.3389 |  75.89% |   53.94% | ON  | ON  | 13.5s ★
     210 |   0.3093 |  76.34% |   54.05% | ON  | ON  | 11.6s ★
     220 |   0.3028 |  76.76% |   54.31% | ON  | ON  | 11.1s ★
     230 |   0.2790 |  77.19% |   54.62% | ON  | ON  | 10.8s ★
     240 |   0.2946 |  77.48% |   54.97% | ON  | ON  | 11.5s ★
     250 |   0.2739 |  77.54% |   55.05% | ON  | ON  | 12.0s ★
     260 |   0.2979 |  77.56% |   55.15% | ON  | ON  | 12.5s ★
     270 |   0.3091 |  77.64% |   55.37% | ON  | ON  | 11.8s ★
     280 |   0.2742 |  77.99% |   55.42% | ON  | ON  | 11.0s ★
     290 |   0.2833 |  78.28% |   55.56% | ON  | ON  | 11.8s ★
     300 |   0.3069 |  78.36% |   55.37% | ON  | ON  | 12.0s ★
     310 |   0.2691 |  78.31% |   55.64% | ON  | ON  | 12.7s
     320 |   0.2496 |  78.57% |   55.67% | ON  | ON  | 10.9s ★
     330 |   0.2709 |  79.00% |   55.71% | ON  | ON  | 12.4s ★
     340 |   0.2757 |  78.76% |   55.84% | ON  | ON  | 11.1s
     350 |   0.2641 |  79.16% |   55.88% | ON  | ON  | 10.8s ★
     360 |   0.2443 |  79.19% |   55.92% | ON  | ON  | 10.7s ★
     370 |   0.2510 |  79.16% |   55.84% | ON  | ON  | 10.6s
     380 |   0.2367 |  79.13% |   55.86% | ON  | ON  | 10.7s
     390 |   0.2340 |  79.61% |   56.11% | ON  | ON  | 10.8s ★
  [Emb saved] round= 400  →  emb_steam_round0400.npy
     400 |   0.2723 |  79.24% |   55.89% | ON  | ON  | 12.1s

========================================================================
  RESULT  (STEAM)
  Best HR@10:          79.61%  (round 390)
  Best NDCG@10:        56.11%
  Total training time:  74.5 min
  Avg time per round:   10.90s

  LOSS SUMMARY:
  Round 1 loss                       0.58900
  Minimum loss                       0.22721  (round 375)
  Final loss (round 400)             0.27230
  Expected range at convergence      0.01 – 0.08
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        79.61%    80.36%     -0.75%                   -0.9%
  NDCG@10      56.11%    65.55%     -9.44%                  -14.4%

  Verdict: ✓ Matches paper (within 2%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   80.36%     65.55%
  Stage 4 (no LDP)               78.84%     55.28%
  Stage 5 (LDP ε=100)            79.61%     56.11%
========================================================================
  Log → stage5_log_steam.json

  TOP-10 FOR USER 0  (cluster 2):
     1. Call of Duty Modern Warfare 3 - Multiplayer              score=5.9486
     2. Call of Duty Modern Warfare 3                            score=5.7793
     3. Counter-Strike Condition Zero                            score=5.7242
     4. Ricochet                                                 score=5.5068
     5. Counter-Strike Condition Zero Deleted Scenes             score=5.3738
     6. Call of Duty Black Ops - Multiplayer                     score=5.0176
     7. Infinite Crisis                                          score=4.9307
     8. Day of Defeat Source                                     score=4.9181
     9. Serious Sam HD The Second Encounter                      score=4.9087
    10. Call of Duty Modern Warfare 2 - Multiplayer              score=4.9076

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
  beta1=0.1  lam=1.0  tau=0.25

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
       1 |   0.6560 |   9.01% |    3.89% | off | ON  | 14.2s ★
      10 |   0.6477 |   9.97% |    4.31% | off | ON  | 12.2s ★
      20 |   0.6403 |  10.60% |    4.79% | off | ON  | 12.5s ★
      30 |   0.8721 |  11.24% |    5.16% | ON  | ON  | 18.0s ★
      40 |   0.8657 |  11.66% |    5.50% | ON  | ON  | 19.3s ★
      50 |   0.8503 |  12.09% |    5.98% | ON  | ON  | 17.7s ★
      60 |   0.8476 |  13.68% |    6.77% | ON  | ON  | 19.4s ★
      70 |   0.8247 |  15.59% |    7.69% | ON  | ON  | 16.8s ★
      80 |   0.8316 |  17.29% |    8.66% | ON  | ON  | 18.6s ★
      90 |   0.8135 |  18.45% |    9.32% | ON  | ON  | 17.0s ★
     100 |   0.8088 |  21.31% |   10.76% | ON  | ON  | 18.8s ★
     110 |   0.7875 |  23.12% |   11.67% | ON  | ON  | 17.6s ★
     120 |   0.7796 |  25.24% |   12.81% | ON  | ON  | 17.2s ★
     130 |   0.7439 |  27.47% |   14.00% | ON  | ON  | 15.6s ★
     140 |   0.7491 |  30.75% |   15.49% | ON  | ON  | 20.6s ★
     150 |   0.7275 |  32.66% |   16.65% | ON  | ON  | 17.5s ★
     160 |   0.7303 |  36.48% |   18.54% | ON  | ON  | 20.5s ★
     170 |   0.6899 |  37.86% |   19.35% | ON  | ON  | 18.4s ★
     180 |   0.6836 |  40.93% |   20.80% | ON  | ON  | 17.5s ★
     190 |   0.6823 |  42.31% |   21.43% | ON  | ON  | 18.5s ★
     200 |   0.6541 |  45.17% |   23.01% | ON  | ON  | 19.8s ★
     210 |   0.6406 |  44.86% |   23.12% | ON  | ON  | 17.1s
     220 |   0.6340 |  47.40% |   24.22% | ON  | ON  | 18.3s ★
     230 |   0.6330 |  47.40% |   24.38% | ON  | ON  | 17.7s
     240 |   0.6172 |  49.20% |   25.63% | ON  | ON  | 16.5s ★
     250 |   0.6044 |  49.42% |   25.77% | ON  | ON  | 17.9s ★
     260 |   0.6276 |  51.22% |   26.52% | ON  | ON  | 20.1s ★
     270 |   0.6144 |  51.86% |   26.96% | ON  | ON  | 18.9s ★
     280 |   0.5887 |  52.17% |   27.16% | ON  | ON  | 18.8s ★
     290 |   0.5943 |  52.49% |   27.31% | ON  | ON  | 18.2s ★
     300 |   0.6057 |  52.49% |   27.48% | ON  | ON  | 19.1s
     310 |   0.5910 |  52.70% |   27.29% | ON  | ON  | 16.9s ★
     320 |   0.5704 |  53.98% |   28.02% | ON  | ON  | 16.5s ★
     330 |   0.5890 |  49.95% |   26.23% | ON  | ON  | 16.6s
     340 |   0.6125 |  50.05% |   26.34% | ON  | ON  | 18.2s
     350 |   0.6099 |  51.01% |   26.77% | ON  | ON  | 18.1s
     360 |   0.6041 |  51.22% |   27.35% | ON  | ON  | 19.6s
     370 |   0.6021 |  52.92% |   27.73% | ON  | ON  | 17.9s
     380 |   0.5955 |  51.11% |   26.84% | ON  | ON  | 17.9s
     390 |   0.6013 |  51.33% |   25.85% | ON  | ON  | 18.0s
  [Emb saved] round= 400  →  emb_ml100k_round0400.npy
     400 |   0.5974 |  50.16% |   25.88% | ON  | ON  | 16.8s

========================================================================
  RESULT  (ML100K)
  Best HR@10:          53.98%  (round 320)
  Best NDCG@10:        28.02%
  Total training time:  117.2 min
  Avg time per round:   17.51s

  LOSS SUMMARY:
  Round 1 loss                       0.65595
  Minimum loss                       0.56976  (round 302)
  Final loss (round 400)             0.59743
  Expected range at convergence      0.45 – 0.58
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        53.98%    63.81%     -9.83%                  -15.4%
  NDCG@10      28.02%    45.03%    -17.01%                  -37.8%

  Verdict: ✗ Below paper (gap > 5%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   63.81%     45.03%
  Stage 5 (LDP ε=100)            53.98%     28.02%
========================================================================
  Log → stage5_log_ml100k.json
  [Names] Loaded 1682 item names from u.item
  [Names] Mapped 1682 names to model item IDs

  TOP-10 FOR USER 0  (cluster 0):
     1. Ransom (1996)                                            score=18.4390
     2. People vs. Larry Flynt, The (1996)                       score=18.2959
     3. Mission: Impossible (1996)                               score=18.1205
     4. Trainspotting (1996)                                     score=18.0909
     5. Mother (1996)                                            score=18.0882
     6. Annie Hall (1977)                                        score=18.0618
     7. Time to Kill, A (1996)                                   score=18.0426
     8. Last Supper, The (1995)                                  score=17.9948
     9. Hercules (1997)                                          score=17.9863
    10. English Patient, The (1996)                              score=17.9851

  Held-out: Terminator 2: Judgment Day (1991)


saumya@SaumyaSuman:~/Recommendation_System/fedpcl/stage5h_a$ python3 train_stage5.py --dataset steam  --data_p
ath steam_processed.json --tau 0.25
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
  beta1=0.1  lam=1.0  tau=0.25

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
      10 |   0.5720 |  11.31% |    5.56% | off | ON  |  7.6s ★
      20 |   0.5512 |  14.91% |    7.68% | off | ON  |  7.6s ★
      30 |   0.6861 |  19.62% |   10.83% | ON  | ON  | 14.3s ★
      40 |   0.6465 |  25.02% |   14.79% | ON  | ON  | 13.1s ★
      50 |   0.6357 |  30.32% |   19.30% | ON  | ON  | 12.2s ★
      60 |   0.6041 |  36.12% |   24.04% | ON  | ON  | 11.9s ★
      70 |   0.5951 |  43.01% |   29.12% | ON  | ON  | 13.6s ★
      80 |   0.5488 |  48.90% |   34.27% | ON  | ON  | 12.0s ★
      90 |   0.5475 |  54.54% |   38.67% | ON  | ON  | 16.2s ★
     100 |   0.5043 |  59.22% |   42.39% | ON  | ON  | 12.9s ★
     110 |   0.4749 |  63.48% |   45.28% | ON  | ON  | 14.4s ★
     120 |   0.4836 |  66.17% |   47.45% | ON  | ON  | 15.3s ★
     130 |   0.4228 |  68.86% |   49.11% | ON  | ON  | 13.8s ★
     140 |   0.4338 |  71.04% |   50.51% | ON  | ON  | 15.1s ★
     150 |   0.4197 |  72.45% |   51.32% | ON  | ON  | 14.4s ★
     160 |   0.3694 |  73.68% |   52.16% | ON  | ON  | 12.6s ★
     170 |   0.3889 |  74.85% |   52.77% | ON  | ON  | 13.7s ★
     180 |   0.3892 |  75.19% |   53.27% | ON  | ON  | 13.9s ★
     190 |   0.3589 |  75.86% |   53.71% | ON  | ON  | 15.6s ★
     200 |   0.3865 |  76.50% |   53.97% | ON  | ON  | 15.7s ★
     210 |   0.3553 |  76.58% |   54.20% | ON  | ON  | 15.1s ★
     220 |   0.3459 |  77.24% |   54.66% | ON  | ON  | 14.1s ★
     230 |   0.3224 |  77.30% |   54.63% | ON  | ON  | 11.9s ★
     240 |   0.3432 |  77.40% |   55.00% | ON  | ON  | 13.5s ★
     250 |   0.3197 |  77.59% |   55.05% | ON  | ON  | 14.5s ★
     260 |   0.3429 |  77.88% |   55.29% | ON  | ON  | 12.5s ★
     270 |   0.3550 |  78.15% |   55.50% | ON  | ON  | 15.7s ★
     280 |   0.3200 |  78.07% |   55.58% | ON  | ON  | 12.9s
     290 |   0.3296 |  78.15% |   55.57% | ON  | ON  | 13.7s
     300 |   0.3573 |  78.44% |   55.62% | ON  | ON  | 14.6s ★
     310 |   0.3177 |  78.73% |   55.80% | ON  | ON  | 14.8s ★
     320 |   0.2956 |  78.71% |   55.80% | ON  | ON  | 12.3s
     330 |   0.3165 |  79.03% |   55.77% | ON  | ON  | 12.0s ★
     340 |   0.3214 |  78.87% |   55.92% | ON  | ON  | 12.8s
     350 |   0.3108 |  78.95% |   56.00% | ON  | ON  | 12.3s
     360 |   0.2895 |  79.24% |   56.06% | ON  | ON  | 12.6s ★
     370 |   0.2970 |  79.11% |   56.04% | ON  | ON  | 13.5s
     380 |   0.2817 |  79.27% |   56.12% | ON  | ON  | 12.8s ★
     390 |   0.2808 |  79.35% |   56.12% | ON  | ON  | 13.3s ★
  [Emb saved] round= 400  →  emb_steam_round0400.npy
     400 |   0.3206 |  79.56% |   56.13% | ON  | ON  | 12.6s ★

========================================================================
  RESULT  (STEAM)
  Best HR@10:          79.56%  (round 400)
  Best NDCG@10:        56.13%
  Total training time:  91.3 min
  Avg time per round:   13.28s

  LOSS SUMMARY:
  Round 1 loss                       0.58900
  Minimum loss                       0.27464  (round 375)
  Final loss (round 400)             0.32058
  Expected range at convergence      0.01 – 0.08
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        79.56%    80.36%     -0.80%                   -1.0%
  NDCG@10      56.13%    65.55%     -9.42%                  -14.4%

  Verdict: ✓ Matches paper (within 2%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   80.36%     65.55%
  Stage 4 (no LDP)               78.84%     55.28%
  Stage 5 (LDP ε=100)            79.56%     56.13%
========================================================================
  Log → stage5_log_steam.json

  TOP-10 FOR USER 0  (cluster 1):
     1. Counter-Strike Condition Zero                            score=6.0389
     2. Ricochet                                                 score=5.8689
     3. Call of Duty Modern Warfare 3 - Multiplayer              score=5.7227
     4. Call of Duty Modern Warfare 3                            score=5.7071
     5. Counter-Strike Condition Zero Deleted Scenes             score=5.5523
     6. Serious Sam HD The Second Encounter                      score=5.4445
     7. Call of Duty Modern Warfare 2 - Multiplayer              score=5.2104
     8. Call of Duty Black Ops - Multiplayer                     score=5.1425
     9. Infinite Crisis                                          score=5.0827
    10. Day of Defeat Source                                     score=5.0603

  Held-out: Ricochet


tau = 0.3

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
  beta1=0.1  lam=1.0  tau=0.3

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
       1 |   0.5890 |   9.48% |    4.64% | off | ON  |  6.4s ★
      10 |   0.5720 |  11.31% |    5.56% | off | ON  |  7.7s ★
      20 |   0.5512 |  14.91% |    7.68% | off | ON  |  7.9s ★
      30 |   0.7331 |  19.94% |   10.90% | ON  | ON  | 12.5s ★
      40 |   0.6919 |  24.86% |   14.73% | ON  | ON  | 11.0s ★
      50 |   0.6811 |  30.10% |   19.07% | ON  | ON  | 10.0s ★
      60 |   0.6503 |  36.17% |   23.97% | ON  | ON  | 13.0s ★
      70 |   0.6426 |  43.01% |   29.33% | ON  | ON  | 11.6s ★
      80 |   0.5957 |  49.24% |   34.45% | ON  | ON  | 13.9s ★
      90 |   0.5944 |  54.86% |   38.92% | ON  | ON  | 13.0s ★
     100 |   0.5460 |  59.30% |   42.53% | ON  | ON  | 11.1s ★
     110 |   0.5185 |  63.08% |   45.56% | ON  | ON  | 12.1s ★
     120 |   0.5290 |  66.44% |   47.71% | ON  | ON  | 13.5s ★
     130 |   0.4646 |  68.83% |   49.06% | ON  | ON  | 11.2s ★
     140 |   0.4798 |  71.23% |   50.71% | ON  | ON  | 13.9s ★
     150 |   0.4628 |  72.88% |   51.57% | ON  | ON  | 14.3s ★
     160 |   0.4091 |  73.89% |   52.41% | ON  | ON  | 12.0s ★
     170 |   0.4303 |  74.79% |   53.08% | ON  | ON  | 12.7s ★
     180 |   0.4336 |  75.57% |   53.64% | ON  | ON  | 12.2s ★
     190 |   0.4003 |  75.91% |   53.87% | ON  | ON  | 13.4s ★
     200 |   0.4280 |  76.26% |   54.24% | ON  | ON  | 14.9s ★
     210 |   0.3946 |  76.79% |   54.59% | ON  | ON  | 14.1s ★
     220 |   0.3885 |  77.00% |   54.76% | ON  | ON  | 12.2s ★
     230 |   0.3627 |  77.32% |   54.59% | ON  | ON  | 11.8s ★
     240 |   0.3867 |  77.51% |   54.88% | ON  | ON  | 12.0s ★
     250 |   0.3658 |  77.69% |   55.27% | ON  | ON  | 12.1s ★
     260 |   0.3850 |  77.96% |   55.29% | ON  | ON  | 11.4s ★
     270 |   0.3977 |  78.07% |   55.31% | ON  | ON  | 13.8s ★
     280 |   0.3653 |  78.09% |   55.63% | ON  | ON  | 11.1s ★
     290 |   0.3718 |  78.23% |   55.34% | ON  | ON  | 14.1s ★
     300 |   0.3992 |  78.65% |   55.62% | ON  | ON  | 12.8s ★
     310 |   0.3571 |  78.65% |   55.83% | ON  | ON  | 12.0s
     320 |   0.3370 |  78.52% |   55.70% | ON  | ON  | 13.3s
     330 |   0.3623 |  78.63% |   55.42% | ON  | ON  | 11.3s
     340 |   0.3668 |  78.97% |   55.85% | ON  | ON  | 12.8s ★
     350 |   0.3539 |  78.97% |   55.71% | ON  | ON  | 12.0s
     360 |   0.3323 |  79.11% |   55.99% | ON  | ON  | 11.3s ★
     370 |   0.3377 |  79.29% |   56.11% | ON  | ON  | 12.5s ★
     380 |   0.3231 |  79.72% |   56.32% | ON  | ON  | 12.4s ★
     390 |   0.3237 |  79.72% |   56.21% | ON  | ON  | 11.8s
  [Emb saved] round= 400  →  emb_steam_round0400.npy
     400 |   0.3643 |  79.64% |   55.97% | ON  | ON  | 12.3s

========================================================================
  RESULT  (STEAM)
  Best HR@10:          79.72%  (round 380)
  Best NDCG@10:        56.32%
  Total training time:  81.8 min
  Avg time per round:   11.95s

  LOSS SUMMARY:
  Round 1 loss                       0.58900
  Minimum loss                       0.31535  (round 375)
  Final loss (round 400)             0.36431
  Expected range at convergence      0.01 – 0.08
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        79.72%    80.36%     -0.64%                   -0.8%
  NDCG@10      56.32%    65.55%     -9.23%                  -14.1%

  Verdict: ✓ Matches paper (within 2%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   80.36%     65.55%
  Stage 4 (no LDP)               78.84%     55.28%
  Stage 5 (LDP ε=100)            79.72%     56.32%
========================================================================
  Log → stage5_log_steam.json

  TOP-10 FOR USER 0  (cluster 4):
     1. Call of Duty Modern Warfare 3                            score=6.1598
     2. Counter-Strike Condition Zero                            score=6.1535
     3. Ricochet                                                 score=6.1426
     4. Call of Duty Modern Warfare 3 - Multiplayer              score=6.1339
     5. Counter-Strike Condition Zero Deleted Scenes             score=5.7129
     6. Call of Duty Black Ops - Multiplayer                     score=5.5164
     7. Serious Sam HD The Second Encounter                      score=5.4805
     8. Call of Duty Modern Warfare 2 - Multiplayer              score=5.3758
     9. Infinite Crisis                                          score=5.3488
    10. Call of Duty Black Ops                                   score=5.3135

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
  beta1=0.1  lam=1.0  tau=0.3

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
       1 |   0.6560 |   9.01% |    3.89% | off | ON  | 16.3s ★
      10 |   0.6477 |   9.97% |    4.31% | off | ON  | 12.3s ★
      20 |   0.6403 |  10.60% |    4.79% | off | ON  | 13.9s ★
      30 |   0.9316 |  11.35% |    5.16% | ON  | ON  | 18.0s ★
      40 |   0.9249 |  11.45% |    5.48% | ON  | ON  | 21.6s ★
      50 |   0.9086 |  12.30% |    6.04% | ON  | ON  | 19.7s ★
      60 |   0.9058 |  13.15% |    6.70% | ON  | ON  | 25.2s ★
      70 |   0.8813 |  14.95% |    7.43% | ON  | ON  | 22.8s ★
      80 |   0.8889 |  17.07% |    8.57% | ON  | ON  | 24.1s ★
      90 |   0.8714 |  19.19% |    9.56% | ON  | ON  | 25.4s ★
     100 |   0.8658 |  21.10% |   10.73% | ON  | ON  | 23.6s ★
     110 |   0.8443 |  23.54% |   11.91% | ON  | ON  | 21.9s ★
     120 |   0.8360 |  25.98% |   13.34% | ON  | ON  | 27.2s ★
     130 |   0.7972 |  26.94% |   13.66% | ON  | ON  | 23.5s ★
     140 |   0.8072 |  30.75% |   15.82% | ON  | ON  | 22.7s ★
     150 |   0.7817 |  33.72% |   17.39% | ON  | ON  | 27.7s ★
     160 |   0.7799 |  36.16% |   18.71% | ON  | ON  | 28.2s ★
     170 |   0.7387 |  37.22% |   18.64% | ON  | ON  | 24.9s ★
     180 |   0.7382 |  40.08% |   20.42% | ON  | ON  | 26.8s ★
     190 |   0.7347 |  43.69% |   21.98% | ON  | ON  | 25.4s ★
     200 |   0.6994 |  45.81% |   23.00% | ON  | ON  | 22.5s ★
     210 |   0.6854 |  45.39% |   23.07% | ON  | ON  | 12.8s
     220 |   0.6799 |  46.87% |   24.22% | ON  | ON  | 17.7s ★
     230 |   0.6771 |  48.25% |   24.91% | ON  | ON  | 17.2s ★
     240 |   0.6699 |  49.20% |   25.73% | ON  | ON  | 19.6s ★
     250 |   0.6578 |  50.27% |   26.21% | ON  | ON  | 16.8s ★
     260 |   0.6790 |  50.27% |   26.38% | ON  | ON  | 19.2s
     270 |   0.6553 |  52.92% |   27.39% | ON  | ON  | 19.7s ★
     280 |   0.6378 |  52.28% |   27.55% | ON  | ON  | 18.0s
     290 |   0.6394 |  51.11% |   26.30% | ON  | ON  | 17.8s
     300 |   0.6691 |  51.11% |   26.59% | ON  | ON  | 19.0s
     310 |   0.6515 |  51.54% |   26.79% | ON  | ON  | 19.3s
     320 |   0.6267 |  52.81% |   27.91% | ON  | ON  | 16.4s
     330 |   0.6392 |  52.17% |   26.83% | ON  | ON  | 17.4s
     340 |   0.6500 |  52.07% |   26.78% | ON  | ON  | 20.8s
     350 |   0.6502 |  51.96% |   27.82% | ON  | ON  | 18.3s
     360 |   0.6449 |  53.02% |   27.87% | ON  | ON  | 17.5s ★
     370 |   0.6468 |  51.54% |   27.26% | ON  | ON  | 21.2s
     380 |   0.6382 |  52.39% |   27.49% | ON  | ON  | 20.1s
     390 |   0.6465 |  50.16% |   26.91% | ON  | ON  | 20.8s
  [Emb saved] round= 400  →  emb_ml100k_round0400.npy
     400 |   0.6452 |  51.75% |   27.49% | ON  | ON  | 17.5s

========================================================================
  RESULT  (ML100K)
  Best HR@10:          53.02%  (round 360)
  Best NDCG@10:        27.87%
  Total training time:  440.6 min
  Avg time per round:   66.00s

  LOSS SUMMARY:
  Round 1 loss                       0.65595
  Minimum loss                       0.61516  (round 289)
  Final loss (round 400)             0.64523
  Expected range at convergence      0.45 – 0.58
  Convergence status                 OUTSIDE expected range

  SCORE DEVIATION FROM PAPER:
  Metric         Ours     Paper    Abs gap    Rel gap (% of paper)
  -----------------------------------------------------------------
  HR@10        53.02%    63.81%    -10.79%                  -16.9%
  NDCG@10      27.87%    45.03%    -17.16%                  -38.1%

  Verdict: ✗ Below paper (gap > 5%)

  PRIVACY-UTILITY TABLE:
  Method                          HR@10    NDCG@10
  ------------------------------------------------
  Paper FedPCL                   63.81%     45.03%
  Stage 5 (LDP ε=100)            53.02%     27.87%
========================================================================
  Log → stage5_log_ml100k.json
  [Names] Loaded 1682 item names from u.item
  [Names] Mapped 1682 names to model item IDs

  TOP-10 FOR USER 0  (cluster 3):
     1. Ransom (1996)                                            score=18.7054
     2. People vs. Larry Flynt, The (1996)                       score=18.6796
     3. Titanic (1997)                                           score=18.4042
     4. Annie Hall (1977)                                        score=18.3953
     5. Trainspotting (1996)                                     score=18.2865
     6. Mission: Impossible (1996)                               score=18.2314
     7. Piano, The (1993)                                        score=18.1961
     8. Great Escape, The (1963)                                 score=18.1554
     9. Mother (1996)                                            score=18.1523
    10. Liar Liar (1997)                                         score=18.1458

  Held-out: Terminator 2: Judgment Day (1991)



tau = 0.35


=================================================================
  Stage 5 — FedPCL + LDP  --  STEAM
  File: stage5_log_steam.json
=================================================================

  RESULTS:
  Best HR@10                     79.558%   (round 390)
  Best NDCG@10                   56.376%
  Privacy budget ε = σ/λ         100.0

  COMPARISON TO PAPER:
  Metric         Ours     Paper    Abs gap     Rel gap
  ------------------------------------------------------
  HR@10        79.56%    80.36%     -0.80%       -1.0%
  NDCG@10      56.38%    65.55%     -9.17%      -14.0%

  Status: ✓ Matches paper (within 2%)

  DATASET STATISTICS:
  Field                                 Yours         Paper  Match
  ------------------------------------------------------------
  Users                                  3757          3753  DIFF
  Items                                  5113          5134  DIFF
  Total interactions                   115139        114713  DIFF
  Train interactions                   111382  (excl. test)
  Test interactions                      3757    1 per user
  Density (%)                          0.5994
  K-core filtering                        off
  Split method                         random

  WARNING — Data mismatch detected:
    Users: yours=3757, paper=3753
    Items: yours=5113, paper=5134
    Interactions: yours=115139, paper=114713

  HYPERPARAMETERS USED:

  +-- Architecture
  |   Embedding dimension d               64
  |   GNN layers K                        2
  +--

  +-- Federated Training
  |   Communication rounds T              400
  |   Clients per round                   128
  |   Local epochs E                      10
  +--

  +-- Learning Rates
  |   Item SGD learning rate η            0.1
  |   User Adam learning rate             0.001
  |   L2 regularisation β₂                1e-06
  +--

  +-- Personalisation (Stage 3+)
  |   K-means clusters K                  5
  |   Cluster model weight μ₁             0.5
  |   Global model weight μ₂              0.5
  |   Re-cluster every N rounds           10
  +--

  +-- Contrastive Learning (Stage 4+)
  |   CL loss weight β₁                   0.1
  |   Item CL weight λ                    1
  |   Temperature τ                       0.35  <- paper: 0.3
  |   Item augmentation dropout           0.3
  |   Warmup rounds (no CL)               20
  |   Max 2-hop neighbours                20
  |   Max items per neighbour             10
  +--

  +-- Local Differential Privacy (Stage 5+)
  |   LDP enabled                         Yes
  |   Clipping bound σ                    0.1
  |   Laplacian noise scale λ             0.001
  +--

  +-- Evaluation
  |   Top-K for HR and NDCG               10
  |   Evaluate every N rounds             10
  +--

  NOTE: 1 parameter(s) differ from paper defaults:
    tau = 0.35  (paper: 0.3)

  TRAINING CURVE  (sampled every ~10% of rounds):
  Expected convergence loss range: 0.01 – 0.08
  Final loss: 0.40371  (OUTSIDE range)

   Round    HR@10   NDCG@10       Loss
  --------------------------------------
       1    9.48%     4.64%    0.58900 ↑
      40   24.94%    14.86%    0.73099 ↑
      80   49.40%    34.42%    0.63483 ↑
     120   66.70%    47.76%    0.56947 ↑
     160   74.34%    52.51%    0.45024 ↑
     200   76.31%    54.13%    0.46671 ↑
     240   77.69%    55.17%    0.42323 ↑
     280   78.20%    55.78%    0.40227 ↑
     320   78.73%    55.74%    0.37435 ↑
     360   79.24%    56.12%    0.36776 ↑
     400   79.37%    56.30%    0.40371 ↑
  (↑ = loss above expected range at this stage)

=================================================================


=================================================================
  Stage 5 — FedPCL + LDP  --  ML100K
  File: stage5_log_ml100k.json
=================================================================

  RESULTS:
  Best HR@10                     53.446%   (round 380)
  Best NDCG@10                   28.119%
  Privacy budget ε = σ/λ         100.0

  COMPARISON TO PAPER:
  Metric         Ours     Paper    Abs gap     Rel gap
  ------------------------------------------------------
  HR@10        53.45%    63.81%    -10.36%      -16.2%
  NDCG@10      28.12%    45.03%    -16.91%      -37.6%

  Status: ✗ Below paper (gap > 5%)

  DATASET STATISTICS:
  Field                                 Yours         Paper  Match
  ------------------------------------------------------------
  Users                                   943           943  OK
  Items                                  1682          1682  OK
  Total interactions                   100000        100000  OK
  Train interactions                    99057  (excl. test)
  Test interactions                       943    1 per user
  Density (%)                          6.3047
  K-core filtering                        off
  Split method                         random

  Dataset matches paper exactly.

  HYPERPARAMETERS USED:

  +-- Architecture
  |   Embedding dimension d               64
  |   GNN layers K                        2
  +--

  +-- Federated Training
  |   Communication rounds T              400
  |   Clients per round                   128
  |   Local epochs E                      10
  +--

  +-- Learning Rates
  |   Item SGD learning rate η            0.1
  |   User Adam learning rate             0.001
  |   L2 regularisation β₂                1e-06
  +--

  +-- Personalisation (Stage 3+)
  |   K-means clusters K                  5
  |   Cluster model weight μ₁             0.5
  |   Global model weight μ₂              0.5
  |   Re-cluster every N rounds           10
  +--

  +-- Contrastive Learning (Stage 4+)
  |   CL loss weight β₁                   0.1
  |   Item CL weight λ                    1
  |   Temperature τ                       0.35  <- paper: 0.3
  |   Item augmentation dropout           0.3
  |   Warmup rounds (no CL)               20
  |   Max 2-hop neighbours                20
  |   Max items per neighbour             10
  +--

  +-- Local Differential Privacy (Stage 5+)
  |   LDP enabled                         Yes
  |   Clipping bound σ                    0.1
  |   Laplacian noise scale λ             0.001
  +--

  +-- Evaluation
  |   Top-K for HR and NDCG               10
  |   Evaluate every N rounds             10
  +--

  NOTE: 1 parameter(s) differ from paper defaults:
    tau = 0.35  (paper: 0.3)

  TRAINING CURVE  (sampled every ~10% of rounds):
  Expected convergence loss range: 0.45 – 0.58
  Final loss: 0.68591  (OUTSIDE range)

   Round    HR@10   NDCG@10       Loss
  --------------------------------------
       1    9.01%     3.89%    0.65595
      40   11.45%     5.46%    0.97445 ↑
      80   16.65%     8.48%    0.93735 ↑
     120   26.09%    13.11%    0.88368 ↑
     160   36.80%    18.63%    0.82807
     200   44.43%    22.72%    0.74816
     240   47.83%    24.91%    0.70937
     280   50.69%    26.42%    0.68295
     320   52.28%    27.03%    0.65605
     360   49.42%    26.45%    0.67107
     400   51.43%    27.55%    0.68591
  (↑ = loss above expected range at this stage)

=================================================================
