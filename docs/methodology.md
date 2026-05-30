# Experimental Methodology

## Experiment Structure

The experiment follows the original structure of Goodfellow et al. (2015):

1. **Training on the old task** — until convergence (early stopping on a validation set)
2. **Training on the new task** — while simultaneously measuring two metrics:
   - New task error (Y axis)
   - Old task error (X axis)
3. **Drawing the Possibilities Frontier curve** — the lower convex hull of the point cloud, on a logarithmic scale

## 8 Conditions

| Activation Function | Algorithm |
|---|---|
| Sigmoid | SGD |
| Sigmoid | Dropout |
| ReLU | SGD |
| ReLU | Dropout |
| Maxout | SGD |
| Maxout | Dropout |
| LWTA | SGD |
| LWTA | Dropout |

## Model Architecture

- **Layers:** 2 hidden layers + softmax classification layer
- **Maxout:** pool size k=2 (each output unit is the maximum over 2 inputs)
- **LWTA:** group size k=2 (in each pair of units, only the larger one receives a gradient)
- **Max-norm constraint:** dynamic constraint per layer in the range 1.0–5.0, sampled separately for fc1, fc2, fc_out in each trial
- **Dropout:** dropout_hidden=0.5, dropout_input=0.2 (fixed, not part of the search)

## Hyperparameter Search

Random search — 8 trials per condition (paper: 25):

| Parameter | Search Range |
|---|---|
| Learning rate | 10^U[-2.0, -0.5] |
| Hidden layer size | U[250, 5000] |
| Max-norm (per layer) | U[1.0, 5.0] |
| Weight initialization range | 10^U[-2.3, -1.0] |
| Sparse init k (Sigmoid/ReLU) | U[10, 30] |
| Momentum | linearly increasing from 0.5 |

> Maxout and LWTA: bias initialized to 0 (random initialization causes one unit in the group to dominate).
> Sigmoid: bias initialized from a negative range to encourage sparsity.
> ReLU: slight positive bias to prevent "dead" units.

## Deviations from the Original Paper

| Parameter | Paper | This Reproduction | Impact on Validity |
|---|---|---|---|
| Trials per condition | 25 | 8 | High — partial HP space coverage |
| Early-stopping patience | 100 epochs | 15 epochs | High — see note on Patience Bias |
| Framework | Theano / Pylearn2 | PyTorch | Medium — numerical precision differences |
| Batch size | 128 | 256 | Medium — different gradient noise |
| Seed | not specified | 42 only | High — no variance estimation |

## Note on Patience Bias

A critical deviation to highlight: reducing patience from 100 to 15 epochs is not neutral — it creates a **systematic bias against Dropout**. Dropout methods generally converge more slowly than SGD (due to the noise introduced by the dropout mechanism into the gradients). As a result, short patience gives a relative advantage to SGD and may suppress the apparent superiority of Dropout. This means the findings on Dropout superiority in this reproduction are **conservative** — under full patience (100 epochs), the performance gap is expected to be larger, not smaller.

## Checkpoint and Resume

The script saves a checkpoint after each condition. If the run is interrupted — it automatically resumes from the last saved point.
