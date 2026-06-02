# AI Collaboration Documentation

This file documents the process of working with an AI assistant (Claude) throughout the project. As students without a deep background in machine learning or Python, we relied heavily on AI guidance at every stage. The conversation log below reflects the real back-and-forth — including the many times things broke, we didn't understand the answer, or had to ask the same thing in a different way.

---

## Stage 1: Choosing a Paper

**Us:**
We need to do a project where we reproduce results from a scientific paper about catastrophic forgetting. We don't really know what that is. Can you explain it simply and help us find a good paper to use?

**Claude:**
Catastrophic forgetting is when a neural network learns a new task and in doing so completely forgets what it learned on the previous task. Imagine you teach someone to recognize cats, they get good at it, then you teach them to recognize dogs — and suddenly they can't recognize cats anymore. That's what happens to neural networks.

A good paper to start with is Goodfellow et al. (2015): "An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks." It's well-cited, uses public datasets (MNIST and Amazon reviews), and has a clear experimental structure. The paper compares 8 combinations of training methods and activation functions across 3 scenarios.

**Us:**
What's an activation function? And what's MNIST?

**Claude:**
An activation function is a mathematical function applied inside a neural network that decides whether a neuron "fires" or not. Common ones are ReLU (which outputs 0 for negative inputs and the value itself for positive ones), Sigmoid (which squishes values between 0 and 1), Maxout, and LWTA.

MNIST is a dataset of 70,000 handwritten digit images (0–9), each 28×28 pixels. It's one of the most common datasets used in machine learning research because it's small, clean, and free to download.

**Us:**
Ok we found the paper. It's on arXiv. Is arXiv a real journal?

**Claude:**
arXiv is a preprint server — papers there haven't necessarily gone through peer review. However, this specific paper was also published as part of a workshop at ICLR 2014, which is a top machine learning conference. So yes, it counts as a recognized scientific publication.

---

## Stage 2: Setting Up the Environment

**Us:**
We installed Python but when we try to run anything we get "ModuleNotFoundError: No module named 'torch'". What do we do?

**Claude:**
You need to install PyTorch. Open your terminal (Command Prompt or Anaconda Prompt) and run:

```
pip install torch torchvision torchaudio
```

If you have a GPU and want to use it, go to pytorch.org and use their install selector to get the right command for your CUDA version.

**Us:**
We ran that but now we get a different error: "ModuleNotFoundError: No module named 'sklearn'"

**Claude:**
You need to install scikit-learn as well:

```
pip install scikit-learn
```

It's also a good idea to install all dependencies at once. I'll write you a requirements.txt file you can use with `pip install -r requirements.txt`.

**Us:**
There's also no tqdm and no matplotlib

**Claude:**
Let me just give you the full requirements.txt with everything you'll need for this project:

```
torch
torchvision
torchaudio
scikit-learn
matplotlib
numpy
tqdm
```

Run `pip install -r requirements.txt` and it will install everything in one go.

**Us:**
ok it worked but now when we run the main script it says "RuntimeError: CUDA error: no kernel image is available for execution on the device". Our GPU is GTX 1660 Super.

**Claude:**
This means there's a mismatch between the PyTorch version and your CUDA version. Go to pytorch.org/get-started/locally, select your OS, CUDA version (check it by running `nvcc --version` in your terminal), and use the exact install command they give you. The generic `pip install torch` sometimes installs a CPU-only version.

**Us:**
nvcc --version says CUDA 11.8

**Claude:**
Then run this exact command:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

That will give you the PyTorch build compiled for CUDA 11.8.

**Us:**
ok that worked! the script runs now. but it's been running for like 30 minutes and nothing happened, is it frozen?

**Claude:**
It's probably downloading the MNIST dataset for the first time — that can take a few minutes depending on your internet speed. After that it will start training. You should see a progress bar from tqdm once it actually starts. If after another 10 minutes you still see nothing, let me know and we'll add some print statements to check where it's stuck.

---

## Stage 3: Understanding the Paper's Experiment

**Us:**
We read the paper and we don't understand what a "Possibilities Frontier" is. The paper keeps talking about it but we don't get the graphs.

**Claude:**
A Possibilities Frontier in this context is a curve that shows the best achievable trade-off between two competing goals — in this case, minimizing error on the old task vs. minimizing error on the new task.

Think of it like this: after training on the new task, you can look at each checkpoint (snapshot of the model) and record two numbers — how bad it is on the old task, and how bad it is on the new task. If you plot all those points, you get a cloud. The Frontier is the lower-left boundary of that cloud — the set of points where you can't do better on one task without getting worse on the other.

A method is "better" if its Frontier is closer to the origin (0,0), meaning it manages to keep both tasks' errors low simultaneously.

**Us:**
why is it on a log scale? the numbers on the axes look weird

**Claude:**
The log scale is used because the differences between methods are most visible in proportional terms, not absolute ones. If one method gets 0.01 error and another gets 0.02, that's 2x worse — but on a regular scale it looks like almost nothing. On a log scale, that 2x difference looks the same as the difference between 0.1 and 0.2.

It also helps when you have methods spanning a wide range — like Sigmoid at 0.8 error and Maxout at 0.04. On a regular scale, Sigmoid would be way off to the right and everything else would be squished together near zero.

**Us:**
can you write the code that draws the frontier curve? we understand the concept now but we don't know how to actually compute the lower-left boundary from a list of points

**Claude:**
Sure. The function takes a list of (old_error, new_error) points and returns the Pareto-optimal subset — the points where no other point is strictly better on both axes simultaneously:

```python
def pareto_lower_left(points):
    """Return the lower-left Pareto frontier from a list of (x, y) points."""
    pts = sorted(points, key=lambda p: p[0])  # sort by x (old task error)
    frontier = []
    min_y = float("inf")
    for x, y in pts:
        if y < min_y:
            frontier.append((x, y))
            min_y = y
    return frontier
```

To draw it, sort the frontier points by x, then plot on a log-log scale. Add markers at each point so you can see individual trials.

**Us:**
ok but when we look at our output graphs they look different from the paper. the curves don't match

**Claude:**
Some differences are expected — you're running 8 trials per condition instead of 25, and with a different random seed. The qualitative pattern should still match: Dropout curves should be closer to the origin than SGD curves, and Maxout should be among the best.

If the qualitative pattern is completely different (e.g., Sigmoid is doing better than Maxout), that would suggest a bug. Can you share what you're seeing?

**Us:**
sigmoid is actually really bad and maxout is the best so i think it's correct. but the shape is different, our curve is more jagged

**Claude:**
Jagged curves are expected with only 8 trials — you have fewer points, so the lower convex hull looks rougher. With 25 trials the paper had more points to form a smooth curve. This is actually a good thing to mention in your methodology section as one of the known deviations.

---

## Stage 4: Amazon Data Problems

**Us:**
we're trying to run scenario 2 and 3 but it says "FileNotFoundError: data/amazon/kitchen.npz". where do we get this file?

**Claude:**
You need to download the Amazon product review dataset manually. Go to:
https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

Download the processed version (it's a tar.gz file). Extract it and you'll get folders named `books`, `dvd`, `electronics`, `kitchen` — each containing `positive.review` and `negative.review` files. Put these folders inside `data/amazon/` in your project directory.

Then run `python prepare_amazon_npz.py` — that script will process the raw text files and create the `.npz` files the main experiment needs.

**Us:**
we downloaded it but when we run prepare_amazon_npz.py we get "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe2 in position 47"

**Claude:**
The Amazon review files have some encoding issues — some characters aren't valid UTF-8. Change the file open line in `load_category_rows` to use `errors='ignore'`:

```python
with open(file_path, encoding="utf-8", errors="ignore") as f:
```

This will skip any malformed characters rather than crashing.

**Us:**
ok that fixed it. but the Amazon feature vectors are 5000-dimensional and MNIST is 784. can we reduce the Amazon features to 784 so the model architecture is the same across all scenarios?

**Claude:**
Yes — use TruncatedSVD from scikit-learn. The key rule is that you must fit the SVD on training data only, then apply the same transform to validation and test. Otherwise information from the test set leaks into the feature space:

```python
from sklearn.decomposition import TruncatedSVD

svd  = TruncatedSVD(n_components=784, random_state=42)
X_tr = svd.fit_transform(X_train)   # fit on train only — no leakage
X_va = svd.transform(X_val)
X_te = svd.transform(X_test)
```

This reduces the 5000-dimensional bag-of-words vectors to 784 components (the dominant directions of variance in the training set), matching MNIST's input size. Note this is a deviation from the paper — the paper feeds the full vocabulary. Scenario 3 results should therefore be treated as approximate.

**Us:**
ok that fixed it. but now prepare_amazon_npz.py ran and created .npz files but when we run the main experiment it crashes on scenario 3 with "AssertionError: assert amazon_cls == 2"

**Claude:**
This assertion checks that the Amazon DVD dataset has exactly 2 classes (positive/negative). The crash means the dataset is being loaded with more or fewer classes — probably because some label values aren't 0 and 1.

Check the `y_train` values in your `.npz` file:

```python
import numpy as np
data = np.load("data/amazon/dvd.npz", allow_pickle=True)
print(np.unique(data["y_train"]))
```

If you see values other than 0 and 1, it means the label remapping in `vectorize_category` didn't work correctly. Share what you see and I'll help fix it.

**Us:**
it prints [0 1] so that's fine? but it still crashes

**Claude:**
Good — labels are correct. The issue is then that `np.unique(y_tr_full).numel()` is counting unique values after some processing and getting a different number. Let me look at the `get_amazon_reduced` function more carefully.

The problem is likely that `np.unique` returns a numpy array and `.numel()` is a PyTorch method — it doesn't exist on numpy arrays. Change it to:

```python
n_cls = int(np.unique(y_tr_full).size)
```

**Us:**
yes that fixed it! scenario 3 is running now

---

## Stage 5: The Run Crashed Halfway

**Us:**
we ran the experiment overnight and when we woke up it had crashed. the error was "RuntimeError: CUDA out of memory". we lost all our progress

**Claude:**
CUDA out of memory means the GPU ran out of VRAM. With a GTX 1660 Super (6GB), this can happen with large hidden layer sizes.

Two fixes:
1. Add a try/except around the training loop to catch OOM errors and skip that trial rather than crashing.
2. More importantly — implement checkpointing so you save progress after each condition and can resume from where you stopped instead of starting over.

Let me write you a checkpoint system. After each condition completes, we save all results to a `.pt` file. At the start of each condition, we check if that file already exists and skip if it does.

**Us:**
can you write the code for that? we don't know how to do it

**Claude:**
Yes. Here's the pattern — for each condition, before running trials, check:

```python
ckpt_path = f"results_repro/ckpt_{scenario_name}_{label}.pt"
if os.path.exists(ckpt_path):
    print(f"Skipping {label} — loaded from checkpoint")
    saved = torch.load(ckpt_path)
    # restore results from saved
    continue

# ... run trials ...

torch.save({"results": ..., "trial_summaries": ..., "winning_model": ...}, ckpt_path)
```

I'll write the full version integrated into your existing code.

**Us:**
ok we added it. but now when we restart it loads the checkpoints fine but the progress bar shows wrong numbers

**Claude:**
The main progress bar is initialized with `total=total_models` but when you skip conditions, you need to manually advance it for the skipped trials:

```python
main_bar.update(trials_per_condition)
```

Add that line right after the `continue` in the checkpoint-found branch.

**Us:**
perfect, that fixed it!

---

## Stage 6: Results and Graphs

**Us:**
the experiment finished! but when we run plot_results.py it crashes with "KeyError: 'trial_summaries'"

**Claude:**
The checkpoint format saved by the old version of the code doesn't match what the new plotting code expects. The old code saved the checkpoint with different key names.

Check what keys are actually in your `.pt` file:

```python
import torch
data = torch.load("results_repro/scenario_1_repro.pt", weights_only=False)
print(data.keys())
```

**Us:**
it prints dict_keys(['results', 'trial_summaries', 'winning_models']) — so trial_summaries IS there?

**Claude:**
Ah, then the issue is that `plot_results.py` is loading from the scenario checkpoint file but one of the scenario files might be missing or corrupted. Check which files exist:

```python
import os
print(os.listdir("results_repro"))
```

**Us:**
scenario_1 and scenario_3 are there but not scenario_3_repro... wait there are two scenario 3 files? scenario_3_repro.pt and scenario_5_repro.pt

**Claude:**
That's correct — the naming follows the paper's figure numbers. Scenario 1 corresponds to Figures 1–2, Scenario 2 (Amazon similar tasks) to Figures 3–4, and Scenario 3 (dissimilar tasks) to Figures 5–6. So the files are `scenario_1_repro.pt`, `scenario_3_repro.pt`, and `scenario_5_repro.pt`. This is confusing naming but it matches the paper's figure numbering. If scenario_3_repro.pt is missing, Scenario 2 (similar tasks) didn't finish — you'll need to re-run it.

**Us:**
oh ok we re-ran scenario 2 and now all three files are there. plot_results.py ran successfully and we have 6 graphs

**Claude:**
Great! Now compare your graphs to the paper's figures:

- Your Fig1 should show Maxout_Dropout and ReLU_Dropout with curves closer to the origin than the SGD variants
- Sigmoid should be clearly worse than the others
- Fig3 and Fig5 should show similar qualitative patterns but with higher error values

If those patterns hold, you've successfully reproduced the paper's main findings.

---

## Stage 7: Ablation Study

**Us:**
the main experiment is done and the results match the paper qualitatively. we want to do something extra to understand *why* dropout helps. the assignment says bonus is for improving on the paper. can we do some kind of deeper analysis?

**Claude:**
Yes — a good extension would be an ablation study. The idea is to isolate the contribution of individual components by varying one thing at a time. For catastrophic forgetting, two obvious candidates are:

1. **Dropout rate** — does a higher dropout rate lead to less forgetting, and by how much?
2. **Weight decay** — does L2 regularization help on top of dropout?

You already have all the infrastructure (data loaders, training loop, evaluation). You just need a simpler MLP class that lets you configure dropout rate and weight decay as parameters.

**Us:**
can you write that MLP class and the training loop for the ablation? we want to run 4 trials per configuration and measure "forgetting rate" specifically, not just best_joint

**Claude:**
Sure. Here's the core class and the forgetting metric:

```python
class AblationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, p_input=0.2, p_hidden=0.5):
        super().__init__()
        self.drop_in  = nn.Dropout(p_input)  if p_input  > 0 else nn.Identity()
        self.drop_hid = nn.Dropout(p_hidden) if p_hidden > 0 else nn.Identity()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.drop_in(x)
        x = self.drop_hid(self.act(self.fc1(x)))
        x = self.drop_hid(self.act(self.fc2(x)))
        return self.fc3(x)
```

For the forgetting metric:
```python
# After training on Task 1:
baseline_old = evaluate_error(model, task1_test)

# After training on Task 2:
final_old = evaluate_error(model, task1_test)

forgetting = final_old - baseline_old  # positive = forgot Task A
```

I'll write the full `run_sequential` function and the plotting code as well.

**Us:**
we ran it and the dropout ablation shows -58% forgetting at p=0.5 compared to no dropout. is that a meaningful result?

**Claude:**
Yes, that's a strong and meaningful result. A 58% reduction in forgetting rate from p=0 to p=0.5 directly quantifies what the paper claims qualitatively. The paper shows dropout curves are closer to the origin but never gives a forgetting rate number — your ablation adds quantitative depth that wasn't in the original paper.

The weight decay result (−7% at 1e-4) is smaller but also consistent with what regularization theory would predict: small L2 prevents extreme weight magnitudes but doesn't change the fundamental learning dynamics the way dropout does.

---

## Stage 8: Final Documentation

**Us:**
we need to write the docs but we're not sure what to include. the assignment says to include a report, graphs, comparison to original data, and the AI documentation (which is this file). what else should we cover?

**Claude:**
For a complete project you should have:

1. **Introduction** — what the paper is about, why you chose it, research question, hypotheses
2. **Methodology** — how the experiment is structured, your deviations from the paper (8 trials instead of 25, patience=15, etc.), and why you made those choices
3. **Results** — your 6 graphs with explanation of what each shows, and comparison to the paper's figures
4. **Conclusion** — did you reproduce the findings? What worked, what didn't, what would you do differently with more time/compute?
5. **Bonus section** — document the improvements you attempted even if results were modest
6. **This file** — the AI collaboration log

The most important thing is to be honest about limitations. Don't claim perfect reproduction — explain where your results differ from the paper and why (fewer trials, different patience, single seed).

**Us:**
one more thing — is it ok that our numbers are slightly different from the paper? like our Maxout_Dropout best_joint is 0.039 but the paper doesn't give exact numbers

**Claude:**
Yes, that's completely fine — and actually it's expected. The paper only shows graphs without printing exact numerical values. You can't directly compare your numbers to the paper. What matters is the qualitative pattern: Dropout better than SGD, Maxout among the best, Sigmoid among the worst, and rankings shift between scenarios. If those patterns hold in your results, you've reproduced the paper's conclusions.

---

*Total AI sessions: approximately 20–25 exchanges across 3 weeks. Main areas where AI help was essential: environment setup, understanding the paper's experimental design, writing the Pareto frontier and feature reduction code (DictVectorizer + TruncatedSVD), debugging data loading errors, implementing checkpointing, designing the ablation study, and writing documentation.*
