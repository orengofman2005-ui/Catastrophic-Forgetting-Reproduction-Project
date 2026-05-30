# Project Takeaways — Catastrophic Forgetting

## Summary of Findings

This project reproduced the central experiments of Goodfellow et al. (2015) on catastrophic forgetting in neural networks. The reproduction covered three sequential learning scenarios, comparing 8 training methods that combine 4 activation functions with two algorithms — SGD and Dropout.

## Takeaway 1: The Dual Role of Dropout

The findings are consistent with the claim that Dropout plays a dual role: both as a regularizer preventing overfitting and as a mechanism for reducing the tendency toward catastrophic forgetting. In all three scenarios, Dropout-based methods produced Frontier curves closer to the origin compared to SGD. For example, in Scenario 1, the best_joint of Maxout_Dropout (0.039) is lower than that of Maxout_SGD (0.042), consistent with the paper's hypothesis that Dropout enables training of higher-capacity networks that are more resistant to forgetting.

## Takeaway 2: Scenario-Dependent Activation Function Ranking

The ranking of activation functions is not universal and varies across scenarios — a finding consistent with the original paper. In Scenario 1 (Input Reformatting), Maxout shows a clear advantage (best_joint=0.039) over Sigmoid (0.173). In Scenario 3 (Dissimilar Tasks), the gaps between methods narrow significantly (range 0.170–0.259), indicating uniform difficulty for all methods when dealing with semantically different task pairs. This finding reinforces the recommendation to cross-validate the choice of activation function for each task pair.

## Takeaway 3: Scenario 2 Analysis — Similar Tasks (Amazon Kitchen -> DVD)

Scenario 2 presents the hardest challenge on the frontier, with error values significantly higher than Scenario 1 (best best_joint: 0.309 for ReLU_Dropout, vs. 0.039 in Scenario 1). This can be attributed to the nature of the data: Amazon Reviews are high-dimensional sparse representations that affect the generalization ability of small MLPs.

Notably, in Scenario 2, ReLU_Dropout (best_joint=0.309) outperformed Maxout_Dropout (0.316), unlike Scenario 1. This finding is consistent with the claim that there is no "universal winner" among activation functions — the choice depends on the data and task characteristics.

Additionally, in Scenario 2, the performance gap between SGD and Dropout is smaller than in the other two scenarios (a difference of ~0.016 for ReLU), which may indicate that when tasks are semantically similar, even plain SGD can exploit partial transfer learning.

## Takeaway 4: Model Capacity Alone Is Not Enough

In Scenario 3, the LWTA model was allocated especially high parameter capacity (see Fig6), yet the results of Fig5 are consistent with the claim that increasing capacity alone is not a sufficient defense against interference when tasks are completely different semantically. This finding supports the hypothesis that semantic dissimilarity between tasks is a stronger predictor of forgetting severity than model size.

## Takeaway 5: Sensitivity of Reproduction to Implementation Details

The reproduction process revealed that machine learning experiment results are highly sensitive to specific implementation decisions — weight initialization, learning rate schedule, and batch size. Achieving qualitative trends consistent with the original paper required revisiting hyperparameters and multiple rounds of debugging.

---

## Reproduced Figures

### Scenario 1 — Input Reformatting (MNIST -> Permuted MNIST)

![Frontier S1](paper_figures/Fig1_frontier_input_reformatting.png)
![Sizes S1](paper_figures/Fig2_model_sizes_input_reformatting.png)

Fig2 demonstrates that under Dropout conditions, the architectures selected as best-performing have significantly larger parameter capacity — particularly Maxout and LWTA. This finding is consistent with the paper's hypothesis that Dropout enables training of wider networks that have spare capacity to retain Task A representations while learning Task B, as reflected in the Frontier curves closer to the origin in Fig1.

---

### Scenario 2 — Similar Tasks (Amazon Kitchen -> Amazon DVD)

![Frontier S2](paper_figures/Fig3_frontier_similar_tasks.png)
![Sizes S2](paper_figures/Fig4_model_sizes_similar_tasks.png)

In Scenario 2, all methods show relatively high errors (best_joint range: 0.309–0.869), reflecting the difficulty of the Amazon Reviews dataset. Nevertheless, the qualitative trend is preserved: ReLU_Dropout (0.309) and Maxout_Dropout (0.316) lead, while Sigmoid — in both variants — sits at the bottom (0.813–0.869). These findings are consistent with Fig3 in the original paper.

---

### Scenario 3 — Dissimilar Tasks (MNIST -> Amazon DVD)

![Frontier S3](paper_figures/Fig5_frontier_dissimilar_tasks.png)
![Sizes S3](paper_figures/Fig6_model_sizes_dissimilar_tasks.png)

Fig6 shows that the LWTA model is allocated the highest parameter capacity. However, the results of Fig5 are consistent with the claim that despite this capacity, the model experiences severe forgetting when transitioning to the dissimilar task — best_old of LWTA_SGD (0.0078) is higher than that of ReLU_Dropout (0.0054), indicating weaker retention of Task A performance.

---

## Summary: Were the Trends Reproduced?

Comparing the produced findings against the figures in the original paper, the key qualitative trends are consistent with the paper in all three scenarios: Frontier curves of Dropout-based methods are closer to the origin, and the ranking of activation functions varies between scenarios. It should be noted that these findings were achieved with only 8 trials (32% of the paper's HP space coverage) and a single seed, and should therefore be treated as qualitative support only — not as full quantitative validation.
