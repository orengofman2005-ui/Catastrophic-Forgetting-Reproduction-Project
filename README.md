# Replication Study: Empirical Investigation of Catastrophic Forgetting (Sami Shamoon College)

This repository contains a student replication of the 2013 paper by Ian Goodfellow et al. investigating catastrophic forgetting in neural networks, completed at Sami Shamoon College (SCE).

## Scenarios
The project replicates the experimental findings across three primary scenarios, testing whether a model forgets its original training when forced to learn a new dataset:
1. **Scenario 1: Reformatting Task** (MNIST vs. Permuted MNIST): The identical problem but with a randomly scrambled input vector.
2. **Scenario 2: Similar Task** (Amazon Reviews Books vs. Electronics): A domain adaptation problem using text sentiment classification.
3. **Scenario 3: Dissimilar Task** (MNIST Subset vs. Amazon Reviews): Transitioning between two completely unrelated data domains.

## The Role of Dropout
Our replication confirms the original findings that **Dropout** is critical in preventing catastrophic forgetting. By randomly zeroing out units during training, Dropout prevents complex co-adaptations between neurons. This forces the network to develop broader, more robust features rather than hyper-specializing, which allows the model to preserve useful representations when moving from the old task to the new task.

## Results & Figures
The `results/` folder contains generated visual evidence corresponding to the hyperparameter search runs. 
* **Figures 1, 3, and 5** display the 'Frontiers' (Old Task Error vs. New Task Error) for each scenario. An "Ideal" marker at `[0.01, 0.01]` represents the target goal of maintaining minimal error on both tasks.
* **Figures 2, 4, and 6** detail the 'Model Sizes' (Parameter Count) of the winning models found during the hyperparameter search.