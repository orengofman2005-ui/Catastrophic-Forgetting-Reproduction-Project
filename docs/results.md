# Reproduction Results

## Possibilities Frontier Figures

Each figure shows the lower convex hull curve of all 8 methods.
X axis = error on the old task, Y axis = error on the new task (both on a logarithmic scale).
**Points close to the origin = good performance on both tasks simultaneously.**
The gray dashed vertical line = median old-task error at the start of new-task training (pre-forgetting reference point).

---

### Scenario 1 — Input Reformatting (MNIST -> Permuted MNIST)

![Frontier](../paper_figures/Fig1_frontier_input_reformatting.png)
![Model Sizes](../paper_figures/Fig2_model_sizes_input_reformatting.png)

Both figures correspond to Figure 1 and Figure 2 in the paper. The two tasks are structurally identical but with a different pixel permutation — requiring the network to relearn the pixel mapping while retaining the abstract representations.

> **Connection between figures:** Fig2 demonstrates that under Dropout, the Winning Models have significantly larger parameter capacity — particularly Maxout and LWTA. This finding is consistent with the paper's hypothesis that Dropout enables training of wider networks with spare capacity to retain Task A representations while learning Task B, as reflected in the Frontier curves closer to the origin in Fig1.

---

### Scenario 2 — Similar Tasks (Amazon Kitchen -> Amazon DVD)

![Frontier](../paper_figures/Fig3_frontier_similar_tasks.png)
![Model Sizes](../paper_figures/Fig4_model_sizes_similar_tasks.png)

Both figures correspond to Figure 3 and Figure 4 in the paper. Both tasks are sentiment analysis on different Amazon product categories — semantically similar but with different language and features per category.

In Scenario 2, all methods show relatively high errors (best_joint range: 0.309–0.869), reflecting the nature of Amazon Reviews as high-dimensional sparse representations. ReLU_Dropout leads (best_joint=0.309) rather than Maxout_Dropout (0.316) — unlike Scenario 1, which reinforces the claim that there is no universal activation function. Additionally, the performance gap between Dropout and SGD is smaller here than in the other scenarios, which may stem from natural transfer learning when tasks are semantically similar — even SGD manages to exploit the learned representations.

---

### Scenario 3 — Dissimilar Tasks (MNIST -> Amazon DVD)

![Frontier](../paper_figures/Fig5_frontier_dissimilar_tasks.png)
![Model Sizes](../paper_figures/Fig6_model_sizes_dissimilar_tasks.png)

Both figures correspond to Figure 5 and Figure 6 in the paper. This is the most challenging pair — computer vision (MNIST) versus natural language processing (Amazon), with no semantic overlap.

> **Anomaly in Scenario 3:** Fig6 shows that the LWTA model is allocated the highest parameter capacity. However, the results of Fig5 are consistent with the claim that increasing capacity alone is not a sufficient defense against interference when tasks are completely different semantically — best_old of LWTA_SGD (0.0078) is higher than that of ReLU_Dropout (0.0054), meaning LWTA retains Task A performance less well despite its larger size.

---

## Key Findings

| Finding | Supporting Quantitative Value |
|---|---|
| Dropout superior to SGD | best_joint of Maxout_Dropout (0.039) vs. Maxout_SGD (0.042) — Scenario 1 |
| Maxout+Dropout on Frontier in every scenario | best_joint: 0.039 / 0.316 / 0.171 in Scenarios 1/2/3 |
| LWTA inconsistent | leads on best_new in Scenario 3 (0.1775) but poor in Scenario 2 (best_joint=0.347) |
| Sigmoid consistently poor | best_joint: 0.173 / 0.813 / 0.201 — bottom in every scenario |
