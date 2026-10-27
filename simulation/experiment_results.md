# TA-BC-BFL Experimental Results

## E1. Ablation Study (final accuracy)

| Configuration | Mean | Std | 95% CI | Runs |
|---|---|---|---|---|
| FedAvg | 92.14% | 0.0167 | [90.06%, 94.21%] | 5 |
| Per-FedAvg | 93.78% | 0.0146 | [91.96%, 95.59%] | 5 |
| Blockchain-FL | 92.53% | 0.0168 | [90.44%, 94.62%] | 5 |
| Trust-FL | 92.20% | 0.0182 | [89.95%, 94.46%] | 5 |
| Bio-Aware FL | 92.11% | 0.0174 | [89.96%, 94.27%] | 5 |
| Trust-Aware FL | 87.03% | 0.0401 | [82.06%, 92.00%] | 5 |
| Full TA-BC-BFL | 88.49% | 0.0323 | [84.48%, 92.50%] | 5 |

## E2. Adversarial Robustness (method x attack)

| Method | Attack | Mean Acc | Std | 95% CI | Drop vs Clean | Adv. Wt (final) |
|---|---|---|---|---|---|---|
| FedAvg | No attack | 92.14% | 0.0167 | [90.06%, 94.21%] | 0.00% | 0.00% |
| FedAvg | Model poisoning | 55.12% | 0.0649 | [47.06%, 63.18%] | 37.02% | 21.08% |
| FedAvg | Label flipping | 86.81% | 0.0158 | [84.86%, 88.77%] | 5.32% | 21.08% |
| FedAvg | Sybil | 56.00% | 0.0648 | [47.96%, 64.04%] | 36.14% | 21.08% |
| Per-FedAvg | No attack | 93.78% | 0.0146 | [91.96%, 95.59%] | 0.00% | 0.00% |
| Per-FedAvg | Model poisoning | 89.41% | 0.0165 | [87.36%, 91.45%] | 4.37% | 21.08% |
| Per-FedAvg | Label flipping | 88.22% | 0.0164 | [86.18%, 90.25%] | 5.56% | 21.08% |
| Per-FedAvg | Sybil | 89.60% | 0.0165 | [87.55%, 91.64%] | 4.18% | 21.08% |
| Blockchain-FL | No attack | 92.53% | 0.0168 | [90.44%, 94.62%] | 0.00% | 0.00% |
| Blockchain-FL | Model poisoning | 87.08% | 0.0437 | [81.65%, 92.51%] | 5.45% | 0.00% |
| Blockchain-FL | Label flipping | 87.11% | 0.0152 | [85.22%, 89.00%] | 5.42% | 19.73% |
| Blockchain-FL | Sybil | 87.08% | 0.0437 | [81.65%, 92.51%] | 5.45% | 0.00% |
| Trust-FL | No attack | 92.20% | 0.0182 | [89.95%, 94.46%] | 0.00% | 0.00% |
| Trust-FL | Model poisoning | 55.92% | 0.0797 | [46.03%, 65.81%] | 36.28% | 22.10% |
| Trust-FL | Label flipping | 86.67% | 0.0168 | [84.59%, 88.75%] | 5.54% | 14.49% |
| Trust-FL | Sybil | 56.73% | 0.0808 | [46.70%, 66.77%] | 35.47% | 22.44% |
| Bio-Aware FL | No attack | 92.11% | 0.0174 | [89.96%, 94.27%] | 0.00% | 0.00% |
| Bio-Aware FL | Model poisoning | 56.01% | 0.0734 | [46.90%, 65.12%] | 36.10% | 19.86% |
| Bio-Aware FL | Label flipping | 86.92% | 0.0163 | [84.89%, 88.94%] | 5.20% | 16.91% |
| Bio-Aware FL | Sybil | 56.92% | 0.0735 | [47.79%, 66.06%] | 35.19% | 19.89% |
| Trust-Aware FL | No attack | 87.03% | 0.0401 | [82.06%, 92.00%] | 0.00% | 0.00% |
| Trust-Aware FL | Model poisoning | 68.73% | 0.0487 | [62.68%, 74.77%] | 18.30% | 1.24% |
| Trust-Aware FL | Label flipping | 85.09% | 0.0347 | [80.77%, 89.40%] | 1.94% | 17.39% |
| Trust-Aware FL | Sybil | 72.70% | 0.0432 | [67.33%, 78.07%] | 14.33% | 1.25% |
| Full TA-BC-BFL | No attack | 86.96% | 0.0307 | [83.15%, 90.77%] | 0.00% | 0.00% |
| Full TA-BC-BFL | Model poisoning | 70.13% | 0.0611 | [62.55%, 77.71%] | 16.83% | 1.14% |
| Full TA-BC-BFL | Label flipping | 84.81% | 0.0358 | [80.37%, 89.26%] | 2.15% | 16.22% |
| Full TA-BC-BFL | Sybil | 76.84% | 0.0383 | [72.08%, 81.60%] | 10.12% | 1.11% |

## E3. Privacy–Accuracy Trade-off

| Epsilon | Mean Accuracy | Std | 95% CI | Runs |
|---|---|---|---|---|
| epsilon=0.1 | 81.07% | 0.0345 | [76.78%, 85.36%] | 5 |
| epsilon=0.5 | 84.82% | 0.0453 | [79.19%, 90.45%] | 5 |
| epsilon=1.0 | 86.96% | 0.0307 | [83.15%, 90.77%] | 5 |
| epsilon=2.0 | 87.91% | 0.0285 | [84.37%, 91.46%] | 5 |
| epsilon=5.0 | 87.88% | 0.0289 | [84.29%, 91.48%] | 5 |
| epsilon=10.0 | 87.95% | 0.0295 | [84.28%, 91.61%] | 5 |

### Gradient-Inversion Leakage (lower = better)

| Epsilon | Reconstruction Quality |
|---|---|
| no DP (baseline) | 0.9974 |
| 0.1 | 0.0000 |
| 0.5 | 0.0000 |
| 1.0 | 0.0000 |
| 2.0 | 0.0000 |
| 5.0 | 0.0000 |
| 10.0 | 0.0000 |

## E4. Scalability

| Clients | Mean Time/Round (s) | Rounds |
|---|---|---|
| 5 | 0.0436 | 100 |
| 10 | 0.0841 | 100 |
| 20 | 0.1909 | 100 |
| 40 | 0.4268 | 100 |

## E5. Statistical Validation

- **clean: Full TA-BC-BFL vs FedAvg**: t=-2.567, p=0.0622 (NOT significant at alpha=0.05), mean diff=-0.0365, n=5
- **clean: Full TA-BC-BFL vs Bio-Aware FL**: t=-2.605, p=0.0597 (NOT significant at alpha=0.05), mean diff=-0.0363, n=5
- **clean: Full TA-BC-BFL vs Trust-Aware FL**: t=1.540, p=0.1983 (NOT significant at alpha=0.05), mean diff=+0.0146, n=5
- **attack(Model poisoning): Full TA-BC-BFL vs FedAvg**: t=11.113, p=0.0004 (SIGNIFICANT at alpha=0.05), mean diff=+0.1501, n=5
- **attack(Label flipping): Full TA-BC-BFL vs FedAvg**: t=-1.853, p=0.1375 (NOT significant at alpha=0.05), mean diff=-0.0200, n=5
- **attack(Sybil): Full TA-BC-BFL vs FedAvg**: t=10.874, p=0.0004 (SIGNIFICANT at alpha=0.05), mean diff=+0.2084, n=5
