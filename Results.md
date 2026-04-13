# Result Comparision

## Fed-PCL-

| Dataset | Our HR@10 | Paper HR@10 | Gap | Our NDCG@10 | Paper NDCG@10 | Gap | Status |
|---|---|---|---|---|---|---|---|
| Steam | 79.32% | 80.36% | −1.04% | 55.81% | 65.55% | −9.74% | ✅ Matches |
| ML-100K | 54.19% | 63.81% | −9.62% | 28.91% | 45.03% | −16.12% | ⚠️ Below |
| ML-1M | 38.96% | 62.86% | −23.90% | 19.56%* | 44.12% | −24.56% | ❌ Not converged |
| Amazon | 31.78% | 34.04% | −2.26% | 16.62% | 22.93% | −6.31% | ✅ Acceptable |





## Fedavg-

| Dataset | Our HR@10 | Paper HR@10 | Gap | Our NDCG@10 | Paper NDCG@10 | Gap | Status |
|---|---|---|---|---|---|---|---|
| Steam | 74.47% | 71.21% | +3.26% | 47.87% | 50.22% | −2.35% | ✅ Acceptable |
| ML-100K | 34.36% | 42.70% | −8.34% | 17.32% | 23.87% | −6.55% | ✅ Acceptable |
| ML-1M | 17.42% | 44.70% | −27.28% | 8.39% | 24.90% | −16.51% | ❌ Not converged |
| Amazon | 31.98% | 26.53% | +5.45% | 15.93% | 14.53% | +1.40% | ✅ Better |



## PerFedRec-

| Dataset | Our HR@10 | Paper HR@10 | Gap | Our NDCG@10 | Paper NDCG@10 | Gap | Status |
|---|---|---|---|---|---|---|---|
| Steam |  | 76.61% |  |  | 62.23% |  | |
| ML-100K |  | 61.87% |  |  | 43.51% |  | |
| ML-1M |  | 61.31% |  |  | 42.83% |  |  |
| Amazon |  | 32.64% |  |  | 21.39% |  |  |
