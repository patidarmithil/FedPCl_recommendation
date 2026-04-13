ML100K

python3 train_stage5.py \
  --dataset ml100k \
  --data_path u.data \
  --n_rounds 400 \
  --local_epochs 10 \
  --tau 0.2 \
  --beta1 0.1 \
  --drop_rate 0.3 \
  --max_neigh 20 \
  --lam 1.0 \
  --warmup_rounds 20

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

=================================================================
  Stage 5 — FedPCL + LDP  --  ML100K
  File: stage5_log_ml100k.json
=================================================================

  RESULTS:
  Best HR@10                     54.189%   (round 330)
  Best NDCG@10                   28.911%
  Privacy budget ε = σ/λ         100.0

  COMPARISON TO PAPER:
  Metric         Ours     Paper    Abs gap     Rel gap
  ------------------------------------------------------
  HR@10        54.19%    63.81%     -9.62%      -15.1%
  NDCG@10      28.91%    45.03%    -16.12%      -35.8%

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
  |   Temperature τ                       0.2  <- paper: 0.3
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
    tau = 0.2  (paper: 0.3)

  TRAINING CURVE  (sampled every ~10% of rounds):
  Expected convergence loss range: 0.45 – 0.58
  Final loss: 0.54068  (within range)

   Round    HR@10   NDCG@10       Loss
  --------------------------------------
       1    9.01%     3.89%    0.65595
      40   11.35%     5.43%    0.79681
      80   16.65%     8.49%    0.76465
     120   25.66%    13.13%    0.70912
     160   39.13%    20.08%    0.66535
     200   45.92%    23.41%    0.58193
     240   48.67%    25.66%    0.54623
     280   53.02%    27.78%    0.53759
     320   53.98%    28.52%    0.52897
     360   52.70%    27.88%    0.53342
     400   51.33%    27.46%    0.54068
  (↑ = loss above expected range at this stage)

=================================================================

AMAZON

python3 train_stage5.py \
  --dataset amazon \
  --data_path amazon_processed.json \
  --n_rounds 400 \
  --local_epochs 10 \
  --tau 0.2 \
  --beta1 0.1 \
  --drop_rate 0.1 \
  --max_neigh 10 \
  --lam 1.0 \
  --warmup_rounds 20

  TOP-10 FOR USER 0  (cluster 4):
     1. item_1300                                                score=20.0873
     2. item_1871                                                score=19.4500
     3. item_1494                                                score=19.2420
     4. item_1138                                                score=19.1505
     5. item_1117                                                score=19.1082
     6. item_1617                                                score=19.0882
     7. item_364                                                 score=19.0725
     8. item_953                                                 score=19.0546
     9. item_1268                                                score=18.9810
    10. item_1272                                                score=18.9615

  Held-out: item_1145

=================================================================
  Stage 5 — FedPCL + LDP  --  AMAZON
  File: stage5_log_amazon.json
=================================================================

  RESULTS:
  Best HR@10                     31.782%   (round 280)
  Best NDCG@10                   16.623%
  Privacy budget ε = σ/λ         100.0

  COMPARISON TO PAPER:
  Metric         Ours     Paper    Abs gap     Rel gap
  ------------------------------------------------------
  HR@10        31.78%    34.04%     -2.26%       -6.6%
  NDCG@10      16.62%    22.93%     -6.31%      -27.5%

  Status: ~ Close to paper (within 5%)

  DATASET STATISTICS:
  Field                                 Yours         Paper  Match
  ------------------------------------------------------------
  Users                                  1504          1435  DIFF
  Items                                  1954          1522  DIFF
  Total interactions                    39115         35931  DIFF
  Train interactions                    37611  (excl. test)
  Test interactions                      1504    1 per user
  Density (%)                           1.331
  K-core filtering                        off
  Split method                         random

  WARNING — Data mismatch detected:
    Users: yours=1504, paper=1435
    Items: yours=1954, paper=1522
    Interactions: yours=39115, paper=35931

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
  |   Temperature τ                       0.2  <- paper: 0.3
  |   Item augmentation dropout           0.1
  |   Warmup rounds (no CL)               20
  |   Max 2-hop neighbours                10
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
    tau = 0.2  (paper: 0.3)

  TRAINING CURVE  (sampled every ~10% of rounds):
  Expected convergence loss range: 0.25 – 0.38
  Final loss: 0.30645  (within range)

   Round    HR@10   NDCG@10       Loss
  --------------------------------------
       1   10.71%     4.97%    0.62434 ↑
      40   11.84%     5.56%    0.62579 ↑
      80   15.76%     7.69%    0.58171 ↑
     120   20.41%    10.43%    0.51800
     160   25.53%    13.75%    0.45422
     200   29.12%    15.54%    0.39083
     240   30.59%    16.30%    0.36582
     280   31.78%    16.62%    0.33374 *
     320   30.65%    16.12%    0.30500
     360   29.45%    15.76%    0.30715
     400   30.05%    16.20%    0.30645
  (↑ = loss above expected range at this stage)

=================================================================


STEAM

python3 train_stage5.py \
  --dataset steam \
  --data_path steam_processed.json \
  --n_rounds 400 \
  --local_epochs 10 \
  --tau 0.2 \
  --beta1 0.1 \
  --drop_rate 0.1 \
  --max_neigh 10 \
  --lam 1.0 \
  --warmup_rounds 20

  TOP-10 FOR USER 0  (cluster 0):
     1. Counter-Strike Condition Zero                            score=5.5541
     2. Ricochet                                                 score=5.4109
     3. Call of Duty Modern Warfare 3 - Multiplayer              score=5.2896
     4. Call of Duty Modern Warfare 3                            score=5.2709
     5. Counter-Strike Condition Zero Deleted Scenes             score=5.2509
     6. Call of Duty Black Ops - Multiplayer                     score=5.1401
     7. Call of Duty Modern Warfare 2 - Multiplayer              score=4.9230
     8. Serious Sam HD The Second Encounter                      score=4.8972
     9. Infinite Crisis                                          score=4.8514
    10. Call of Duty Black Ops                                   score=4.7478

  Held-out: Ricochet

=================================================================
  Stage 5 — FedPCL + LDP  --  STEAM
  File: stage5_log_steam.json
=================================================================

  RESULTS:
  Best HR@10                     79.319%   (round 390)
  Best NDCG@10                   55.814%
  Privacy budget ε = σ/λ         100.0

  COMPARISON TO PAPER:
  Metric         Ours     Paper    Abs gap     Rel gap
  ------------------------------------------------------
  HR@10        79.32%    80.36%     -1.04%       -1.3%
  NDCG@10      55.81%    65.55%     -9.74%      -14.9%

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
  |   Temperature τ                       0.2  <- paper: 0.3
  |   Item augmentation dropout           0.1
  |   Warmup rounds (no CL)               20
  |   Max 2-hop neighbours                10
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
    tau = 0.2  (paper: 0.3)

  TRAINING CURVE  (sampled every ~10% of rounds):
  Expected convergence loss range: 0.01 – 0.08
  Final loss: 0.22797  (OUTSIDE range)

   Round    HR@10   NDCG@10       Loss
  --------------------------------------
       1    9.48%     4.64%    0.58900 ↑
      40   25.26%    15.04%    0.56761 ↑
      80   50.76%    35.25%    0.46097 ↑
     120   66.54%    47.45%    0.38519 ↑
     160   73.84%    52.51%    0.28282 ↑
     200   76.02%    53.90%    0.29395 ↑
     240   76.98%    54.59%    0.25165 ↑
     280   77.99%    55.03%    0.23093 ↑
     320   78.28%    55.27%    0.20545 ↑
     360   79.16%    55.62%    0.20172 ↑
     400   79.13%    55.61%    0.22797 ↑
  (↑ = loss above expected range at this stage)

=================================================================


ML1M

python3 train_stage5.py \
  --dataset ml1m \
  --data_path ratings.dat \
  --n_rounds 400 \
  --local_epochs 10 \
  --tau 0.2 \
  --beta1 0.1 \
  --drop_rate 0.3 \
  --max_neigh 20 \
  --lam 1.0 \
  --warmup_rounds 20

 TOP-10 FOR USER 0  (cluster 4):
     1. Frequency (2000)                                         score=3.1997
     2. Mask, The (1994)                                         score=3.1629
     3. Mission: Impossible (1996)                               score=3.1458
     4. What Lies Beneath (2000)                                 score=3.0765
     5. Space Cowboys (2000)                                     score=3.0335
     6. Battlefield Earth (2000)                                 score=3.0245
     7. Boat, The (Das Boot) (1981)                              score=2.9657
     8. African Queen, The (1951)                                score=2.8850
     9. L.A. Confidential (1997)                                 score=2.8812
    10. Double Indemnity (1944)                                  score=2.8803

  Held-out: Toy Story (1995)

=================================================================
  Stage 5 — FedPCL + LDP  --  ML1M
  File: stage5_log_ml1m.json
=================================================================

  RESULTS:
  Best HR@10                     38.957%   (round 400)
  Best NDCG@10                   19.558%
  Privacy budget ε = σ/λ         100.0

  COMPARISON TO PAPER:
  Metric         Ours     Paper    Abs gap     Rel gap
  ------------------------------------------------------
  HR@10        38.96%    62.86%    -23.90%      -38.0%
  NDCG@10      19.56%    44.12%    -24.56%      -55.7%

  Status: ✗ Below paper (gap > 5%)

  DATASET STATISTICS:
  Field                                 Yours         Paper  Match
  ------------------------------------------------------------
  Users                                  6040          6040  OK
  Items                                  3706          3706  OK
  Total interactions                  1000209       1000209  OK
  Train interactions                   994169  (excl. test)
  Test interactions                      6040    1 per user
  Density (%)                          4.4684
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
  |   Temperature τ                       0.2  <- paper: 0.3
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
    tau = 0.2  (paper: 0.3)

  TRAINING CURVE  (sampled every ~10% of rounds):
  Expected convergence loss range: 0.40 – 0.55
  Final loss: 0.65564  (OUTSIDE range)

   Round    HR@10   NDCG@10       Loss
  --------------------------------------
       1   10.10%     4.55%    0.66264
      40   10.55%     4.88%    0.82006
      80   11.41%     5.43%    0.81751
     120   13.36%     6.39%    0.80990
     160   16.72%     7.85%    0.79748
     200   20.32%     9.73%    0.79291
     240   24.57%    11.92%    0.74823
     280   29.17%    14.33%    0.72525
     320   33.10%    16.43%    0.67878
     360   36.36%    18.11%    0.66599
     400   38.96%    19.56%    0.65564 *
  (↑ = loss above expected range at this stage)

=================================================================

