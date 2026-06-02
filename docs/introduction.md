# Introduction and Research Question

## Why We Chose This Paper

We were looking for a paper that met three conditions: published in a recognized venue, uses publicly available datasets, and has a clear experimental structure that can actually be reproduced without institutional resources.

We came across Goodfellow et al. (2015) while searching for papers on continual learning. What caught our attention was that the paper directly pits several well-known techniques against each other — Sigmoid, ReLU, Maxout, LWTA, SGD, Dropout — in a controlled comparison. Most papers we found either proposed a new method without a fair baseline comparison, or used proprietary datasets we had no access to.

The second reason is practical: the paper uses MNIST (freely available, small, fast to train on) and Amazon product reviews (also public). We estimated that with a consumer GPU we could actually run the experiments in a reasonable time frame, which turned out to be true — though with some adjustments to the number of trials (8 instead of 25) due to hardware limits.

---

## Background

Catastrophic Forgetting is a phenomenon in which a neural network trained on a new task loses its ability to perform a previous task sharply and severely. The phenomenon was first described by McCloskey & Cohen (1989) and Ratcliff (1990), and has been identified as a fundamental challenge in building multi-task and continual learning systems.

The work of Goodfellow et al. (2015) is one of the first systematic attempts to quantify and compare the level of catastrophic forgetting across modern training methods and activation functions. The paper examined 8 combinations of 4 activation functions with 2 training algorithms, across 3 sequential learning scenarios.

---

## Related Work

**Srivastava et al. (2013) — "Compete to Compute"** was the central work that Goodfellow et al. sought to challenge. Srivastava et al. argued that the LWTA (Local Winner Take All) activation function is superior to Sigmoid and ReLU in resisting catastrophic forgetting when training with plain SGD. Their study was conducted on a single task pair, with fixed hyperparameters, and without comparison to the Dropout algorithm.

Goodfellow et al. (2015) extended the experiment along four dimensions: (1) three different task pairs, (2) random hyperparameter search, (3) inclusion of Dropout as a training algorithm, and (4) presentation of Frontier curves instead of single points. Their central conclusion — that Dropout outperforms all activation functions and SGD across all scenarios — contradicted Srivastava et al.'s main claim about LWTA superiority.

This reproduction project validates whether that conclusion holds in a modern PyTorch environment and under computational constraints.

---

## Research Question

**Can the central conclusions of Goodfellow et al. (2015) — in particular the superiority of Dropout over SGD in preventing catastrophic forgetting, and the scenario-dependent ranking of activation functions — be reproduced in a modern PyTorch environment under consumer hardware constraints?**

Sub-question: Is it possible to arrive at the same qualitative trends with 8 trials per condition (instead of 25 in the original paper), without changing the general conclusions?

---

## Working Hypotheses

Three hypotheses are tested:

1. Dropout methods will achieve Possibilities Frontier curves closer to the origin than SGD — in all three scenarios.
2. Maxout+Dropout will appear on the Frontier in all three scenarios.
3. The ranking of activation functions will vary between scenarios — with no universal winner.

---

## Key Definitions

**Possibilities Frontier:** The lower convex hull boundary curve of the point cloud (Task A error, Task B error). Points close to the origin represent optimal retention of both tasks simultaneously. Logarithmic scale presentation allows clear distinction between leading methods.

**Winning Model:** The model with the lowest joint validation error (Task A error + Task B error) within each condition. Used for the model-size figures.

**Sequential Learning:** Training on Task A until convergence (via early stopping on a validation set), then training on Task B while measuring performance on both throughout the training process.
