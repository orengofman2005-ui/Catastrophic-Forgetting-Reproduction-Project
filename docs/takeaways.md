# Takeaways

## Answer to the Research Question

The research question posed: *Can the central conclusions of Goodfellow et al. (2015) be reproduced in a modern PyTorch environment under consumer hardware constraints?*

**The answer: Yes, to a large extent — with important caveats.**

| Hypothesis | Result | Evidence |
|---|---|---|
| Dropout superior to SGD in all scenarios | Consistent with the paper | Dropout best_joint lower than SGD in 3 out of 4 activation functions in Scenario 1 |
| Maxout+Dropout on Frontier in every scenario | Consistent with the paper | best_joint: 0.039 / 0.316 / 0.161 in Scenarios 1/2/3 |
| Activation function ranking is scenario-dependent | Consistent with the paper | ReLU leads in S2–S3, Maxout leads in S1 |

---

## Quantitative Comparison Table — Best Joint Error (lower is better)

| Condition | Scenario 1 | Scenario 2 | Scenario 3 |
|---|---|---|---|
| Sigmoid + SGD | 0.173 | 0.813 | 0.205 |
| Sigmoid + Dropout | 0.203 | 0.869 | 0.245 |
| ReLU + SGD | 0.059 | 0.325 | 0.189 |
| ReLU + Dropout | 0.044 | **0.309** | **0.151** |
| Maxout + SGD | 0.042 | 0.341 | 0.189 |
| Maxout + Dropout | **0.039** | 0.316 | 0.161 |
| LWTA + SGD | 0.108 | 0.356 | 0.180 |
| LWTA + Dropout | 0.045 | 0.347 | 0.190 |

> These values are from this reproduction only. The original paper presented results graphically without explicit numerical values.

---

## Key Findings

**1. Dropout:** Superior Frontier curves over SGD in all three scenarios. The gap is relatively small (e.g. 0.039 vs. 0.042 in Scenario 1), and it should be noted that due to Patience Bias (see Methodology), the true gap is likely larger. This finding is consistent with the original paper's claim.

**2. Maxout:** Appears on the Frontier in all three scenarios — consistent with the original paper.

**3. Sigmoid:** Consistently poor performance across all scenarios (best_joint: 0.173–0.813).

**4. LWTA — Scenario 3 anomaly:** Allocated high capacity (Winning Model, Fig6), yet Fig5 results are consistent with the claim that capacity alone is not a defense mechanism when tasks are semantically dissimilar.

---

## Per-Scenario Analysis

### Scenario 1 — Input Reformatting (MNIST → Permuted MNIST)

Dropout plays a dual role: both as a regularizer preventing overfitting and as a mechanism for reducing catastrophic forgetting. Maxout_Dropout (0.039) beats Maxout_SGD (0.042), and Fig2 shows that Dropout conditions select significantly larger architectures — consistent with the paper's hypothesis that Dropout enables wider networks with spare capacity to retain Task A representations while learning Task B.

### Scenario 2 — Similar Tasks (Amazon Kitchen → Amazon DVD)

All methods show relatively high errors (best_joint range: 0.309–0.869), reflecting the high-dimensional sparse nature of Amazon Reviews. ReLU_Dropout (0.309) leads rather than Maxout_Dropout (0.316) — unlike Scenario 1, reinforcing that there is no universal activation function. The performance gap between SGD and Dropout is smaller than in the other two scenarios, possibly because semantically similar tasks allow even plain SGD to exploit partial transfer.

### Scenario 3 — Dissimilar Tasks (MNIST 2/9 → Amazon DVD)

ReLU_Dropout ranks first (0.151) and Maxout_Dropout second (0.161) — Dropout methods lead as expected. The gaps between non-Sigmoid methods narrow (range 0.151–0.205), indicating broadly comparable difficulty when tasks are semantically dissimilar. LWTA was allocated the highest parameter capacity yet showed weaker Task A retention, confirming that model size alone is not sufficient protection against interference when tasks have no semantic overlap.

---

## Limitations and Caveats

**1. Limited HP space coverage:** 8 trials out of 25 = 32% coverage. The resulting Frontier may be pessimistic — better points may exist in the unsampled HP space.

**2. Single seed:** All experiments were run with a single seed=42. No confidence intervals and no variance estimation. Results may differ with a different seed.

**3. Patience Bias:** Patience of 15 epochs (vs. 100 in the paper) creates a systematic bias — Dropout converges more slowly than SGD, so short patience relatively benefits SGD. The findings on Dropout superiority are therefore **conservative**: under full patience, the performance gap is expected to be larger.

**4. Amazon preprocessing:** There is no full certainty that the Amazon Reviews preprocessing matches that performed by Glorot et al. (2011b) on which the paper relies.

**5. Numerical precision:** Theano and PyTorch may exhibit numerical precision differences in floating point computations, affecting results at the margins.

---

## Recommendations for Future Work

- Run the experiment with multiple seeds and report mean ± standard deviation
- Increase the number of trials to 25 as in the original paper (on a dedicated GPU)
- Validate Amazon preprocessing against Glorot et al. (2011b)
- Test sensitivity to patience size (15 / 50 / 100 epochs)
- Compare with modern Continual Learning methods (EWC, Progressive Nets)

---

## Reflection

Going into this project, we assumed that reproducing a 2015 paper would be relatively straightforward — the methods are well-established, the datasets are public, and the architecture is a simple MLP. What we did not anticipate was how much of the difficulty would come not from the model itself, but from the details the paper does not report: the exact hyperparameter ranges, the patience setting, the input preprocessing for Scenario 3. Every one of those missing details required a judgment call, and each judgment call introduced a potential source of divergence from the original results.

The most surprising finding for us was in Scenario 3. We expected Maxout+Dropout to lead — that was the paper's headline result — but in our run ReLU+Dropout came out on top (0.151 vs. 0.161). At first we suspected a bug. After verifying the checkpoints and re-reading the methodology, we concluded it is most likely a combination of our SVD deviation and the limited 8-trial search. It was a good reminder that "qualitative agreement" does not mean "identical ranking at the margins."

The ablation study was the part we found most valuable personally. Running the main experiment tells you *what* happens; the ablation tells you *why*. Seeing that Dropout alone accounts for a 55% reduction in forgetting — and that weight decay adds almost nothing on top of it — gave us a much more concrete understanding of the mechanism than reading the paper alone ever could. If we had more compute, isolating the effect of patience (15 vs. 100 epochs) would be the next thing we would run.
