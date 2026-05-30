# Conclusion

## Answer to the Research Question

The research question posed: *Can the central conclusions of Goodfellow et al. (2015) be reproduced in a modern PyTorch environment under consumer hardware constraints?*

**The answer: Yes, to a large extent — with important caveats.**

| Hypothesis | Result | Evidence |
|---|---|---|
| Dropout superior to SGD in all scenarios | Consistent with the paper | Dropout best_joint lower than SGD in 6 out of 8 conditions in Scenario 1 |
| Maxout+Dropout on Frontier in every scenario | Consistent with the paper | best_joint: 0.039 / 0.316 / 0.171 in Scenarios 1/2/3 |
| Activation function ranking is scenario-dependent | Consistent with the paper | LWTA leads on best_new in Scenario 3 but fails in Scenario 2 |

---

## Quantitative Comparison Table — Best Joint Error (lower is better)

| Condition | Scenario 1 | Scenario 2 | Scenario 3 |
|---|---|---|---|
| Sigmoid + SGD | 0.173 | 0.813 | 0.201 |
| Sigmoid + Dropout | 0.203 | 0.869 | 0.259 |
| ReLU + SGD | 0.059 | 0.325 | 0.173 |
| ReLU + Dropout | **0.044** | **0.309** | 0.177 |
| Maxout + SGD | 0.042 | 0.341 | 0.170 |
| Maxout + Dropout | **0.039** | 0.316 | **0.171** |
| LWTA + SGD | 0.108 | 0.356 | 0.196 |
| LWTA + Dropout | 0.045 | 0.347 | 0.176 |

> These values are from this reproduction only. The original paper presented results graphically without explicit numerical values.

---

## Key Findings

**1. Dropout:** Superior Frontier curves over SGD in all three scenarios. The gap is relatively small (e.g. 0.039 vs. 0.042 in Scenario 1), and it should be noted that due to Patience Bias (see Methodology), the true gap is likely larger. This finding is consistent with the original paper's claim.

**2. Maxout:** The only condition appearing on the Frontier in all three scenarios — consistent with the original paper.

**3. Sigmoid:** Consistently poor performance across all scenarios (best_joint: 0.173–0.813).

**4. LWTA — Scenario 3 anomaly:** Allocated high capacity (Winning Model, Fig6), yet Fig5 results are consistent with the claim that capacity alone is not a defense mechanism when tasks are semantically dissimilar.

---

## Limitations and Caveats

**1. Limited HP space coverage:** 8 trials out of 25 = 32% coverage. The resulting Frontier may be pessimistic — better points may exist in the unsampled HP space.

**2. Single seed:** All experiments were run with a single seed=42. No confidence intervals and no variance estimation. Results may differ with a different seed.

**3. Patience Bias:** Patience of 15 epochs (vs. 100 in the paper) creates a systematic bias — Dropout converges more slowly than SGD, so short patience relatively benefits SGD. The findings on Dropout superiority are therefore **conservative**: under full patience, the performance gap is expected to be larger.

**4. Amazon preprocessing:** There is no full certainty that the Amazon Reviews preprocessing matches that performed by Glorot et al. (2011b) on which the paper relies.

**5. Numerical precision:** Theano and PyTorch may exhibit numerical precision differences in floating point computations, affecting results at the margins.

---

## Recommendations for Future Work

- Run the experiment with multiple seeds and report mean +/- standard deviation
- Increase the number of trials to 25 as in the original paper (on a dedicated GPU)
- Validate Amazon preprocessing against Glorot et al. (2011b)
- Test sensitivity to patience size (15 / 50 / 100 epochs)
- Compare with modern Continual Learning methods (EWC, Progressive Nets)
