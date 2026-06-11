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
Is arXiv a real journal?

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

---

---

# Claude Conversation Log — Full Session History

> The following sections document every conversation held with Claude (via claude.ai) throughout the project. Each section corresponds to a distinct chat session, listed in chronological order. These are added here for full transparency and traceability of AI usage.

---

## Session 1: Choosing a Paper and Initial Project Scoping
**Date:** March 26, 2026
**Chat URL:** https://claude.ai/chat/830867ab-b176-4f37-891f-c3bb1d9b91e5

**What we discussed:**
This was the very first conversation about the project. We explained that we needed to reproduce a scientific paper on catastrophic forgetting for a course project. Claude explained what catastrophic forgetting is, then suggested three candidate papers:

1. **Goodfellow et al. (2013/2015)** — "An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks" (arXiv:1312.6211) — recommended as the primary choice
2. **Kirkpatrick et al. (2017)** — "Overcoming Catastrophic Forgetting in Neural Networks" (EWC paper)
3. **Toneva et al. (2019)** — "An Empirical Study of Example Forgetting during Deep Neural Network Learning"

We confirmed that the Goodfellow paper was the right choice because the datasets (MNIST, Amazon reviews), methods (SGD, Dropout, activation functions), and figures (Possibilities Frontier curves) are all clearly specified and reproducible with standard Python tools.

**Key decisions made:**
- Chose Goodfellow et al. (2013/2015) as the primary paper
- Confirmed MNIST and Amazon reviews as the datasets
- Confirmed two-layer fully connected networks as the architecture
- Chose PyTorch as the framework (over TensorFlow)

**Code produced:**
Claude wrote the initial full project skeleton including `TwoLayerNet`, `MaxoutNet`, data loaders, and a structure covering all three experimental scenarios with SGD vs. Dropout vs. L2 comparisons and Possibilities Frontier plotting.

---

## Session 2: Finding Backup Articles
**Date:** May 5, 2026
**Chat URL:** https://claude.ai/chat/69920695-9f3a-4521-a0b9-3a132729e221

**What we discussed:**
We needed to identify backup/related papers to cite alongside the main paper. Claude searched past conversations to recall what had already been identified, then searched the web for additional simpler options.

**Papers identified:**

1. **Kirkpatrick et al. (2017)** — "Overcoming Catastrophic Forgetting in Neural Networks" — Elastic Weight Consolidation (EWC). Most directly related backup: addresses the same problem on the same datasets. Link: arxiv.org/abs/1612.00796

2. **Kemker et al. (2018)** — "Measuring Catastrophic Forgetting in Neural Networks" — Introduces clear metrics comparing five mitigation mechanisms (regularization, ensembling, rehearsal, dual-memory, sparse-coding). Simplest option as it's a comparison study with no complex new method to implement. Link: arxiv.org/abs/1708.02072

3. **Toneva et al. (2019)** — "An Empirical Study of Example Forgetting during Deep Neural Network Learning" — Studies forgetting at the level of individual training examples during SGD. Link: arxiv.org/abs/1812.05159

**Key recommendation:**
Kirkpatrick et al. (EWC) was identified as the strongest backup because it directly addresses catastrophic forgetting as a problem to be solved (complementing Goodfellow's empirical analysis) and uses the same MNIST benchmarks.

---

## Session 3: Full Repository Code Review and Bug Fixing
**Date:** May 9, 2026
**Chat URL:** https://claude.ai/chat/d5aba035-7c72-46f5-8313-e38646250c2a

**What we discussed:**
We uploaded the full GitHub repository and asked Claude to review it against project requirements and fix everything. Claude conducted a thorough audit and found several critical problems.

**Critical bugs found and fixed:**

1. **NameError in `__main__` block** — `pytorch_reproduction_suite.py` called `run_scenario_2()` and `run_scenario_3()`, but the actual function names were `run_scenario_2_all_pairs` and `run_scenario_3_all_amazon`. This would cause an immediate crash before any experiment ran.

2. **Broken README links** — Links to `docs/ai_methodology.md` and `docs/ai_plans.md` pointed to files that did not exist.

3. **Wrong figure filenames in README** — README referenced `figure1.png`, `figure2.png` etc., but the actual output files were named `fig_s1_frontier.png`, `fig_s2_barplot.png`, etc.

4. **Empty `docs/ai_usage.md`** — The file existed but had no content.

5. **Missing `requirements.txt`** — No dependency file existed for the project.

6. **Missing CSV logs** — README promised CSV output files that were never generated.

**New files created:**
- `requirements.txt`
- `export_logs_to_csv.py` (plus actual CSV output from existing checkpoints)
- `docs/figures_explanation.md` — detailed explanation of all six graphs with side-by-side comparison to the original paper's figures, including quantitative differences explained by reduced trial counts and patience settings
- `docs/ai_methodology.md`
- `docs/ai_plans.md`
- Fully rewritten `README.md` with corrected links and accurate file references
- Filled-in `docs/ai_usage.md`
- `SUMMARY_OF_CHANGES.md`

**Outcome:**
A downloadable ZIP file (`Catastrophic-Forgetting-Reproduction-Project-FIXED.zip`) was produced for upload to GitHub.

---

## Session 4: Writing `final_experiment_repro.py` — Paper-Faithful Reproduction
**Date:** May 13, 2026
**Chat URL:** https://claude.ai/chat/f4cedd63-5622-45b2-bd84-ea8daaeee6fa

**What we discussed:**
The original `pytorch_reproduction_suite.py` had several deviations from the paper that we hadn't documented or fixed. Claude wrote a new file `final_experiment_repro.py` that explicitly aligns the code 1:1 with the Goodfellow et al. paper.

**Key deviations corrected in the new file:**

| Issue | Old code | Paper value | Fixed |
|-------|----------|-------------|-------|
| Early stopping patience | 12 epochs | 100 epochs (§4) | ✅ |
| Trials per condition | 3–10 | 25 (§3.3) | ✅ |
| Dropout rates | Varied | p_hidden=0.5, p_visible=0.2 (§3.1) | ✅ documented |
| Hidden layer sizes | Narrow | Wider random search pool | ✅ |
| Bias initialization | Default | Maxout=0, Sigmoid=negative, ReLU=positive (§4) | ✅ |
| Frontier computation | Convex hull | Pareto lower-left in log-space (§4) | ✅ |
| Scenario 2 pair | All pairs | Kitchen→DVD specifically | ✅ |
| Scenario 3 pair | Generic | MNIST(2,9)→Amazon(DVD) specifically | ✅ |
| Joint stopping criterion | Loss-based | val_old + val_new not improving for 100 epochs | ✅ |
| Amazon dimensionality | Inconsistent | PCA to 784 features | ✅ |

Claude also updated `docs/ai_methodology.md` to document that Claude was the primary AI tool used throughout, and wrote the Human-in-the-loop methodology section.

---

## Session 5: README and Documentation Improvements (GitHub Direct Editing)
**Date:** May 13, 2026
**Chat URL:** https://claude.ai/chat/c5e8c5fa-122d-49cb-93f7-261e956b274a

**What we discussed:**
This was a long session where Claude was connected directly to the GitHub repository via MCP and made changes live. We asked Claude to create all missing documentation files and then do a final academic-quality review pass.

**Files created directly in GitHub:**
- `docs/ai_methodology.md` — Detailed documentation of Claude usage across all project phases with specific Human-in-the-loop examples and prompts
- `docs/algorithm.md` — Modular breakdown of all code sections (data preparation, training loop, evaluation, plotting) with validation protocols for each step
- `docs/ai_plans.md` — Initial planning documents for all three scenarios

**Critical bug found in this session:**
`plot_results.py` used checkpoint filenames that did not match the filenames actually produced by `final_experiment_repro.py`. Specifically, Scenario 2 saves with `fig_num=3` and Scenario 3 saves with `fig_num=5`, requiring a mapping dictionary `{1:1, 2:3, 3:5}` in the plotting script. The old code used direct scenario numbers and crashed on Scenarios 2 and 3.

**Academic review findings (all fixed):**
- Broken sentence in `ai_plans.md` ("אמרה ש-..." → "הדרישה היא ש...")
- Pipeline diagram in `algorithm.md` referenced nonexistent `*.csv` files → corrected to `*.pt` and `*.png`
- Missing formal citation of Goodfellow et al. in README → added
- `takeaways.md` lacked any numerical values → added quantitative comparison table
- Figure comparison analysis written and added to `takeaways.md` (covering all 3 scenarios and 6 figures)

---

## Session 6: Repository Review Attempt (Connectivity Issue)
**Date:** May 18, 2026
**Chat URL:** https://claude.ai/chat/5b053033-94a5-435f-a859-d19a7f210996

**What we discussed:**
A short session where the GitHub repository link was shared for review. Claude was unable to access the repository contents due to a network/access issue (the repo may have been temporarily private or there was a connectivity problem). No changes were made.

**Outcome:** No code or documentation changes. The session ended with Claude offering three alternatives: confirm the link, share content directly, or clarify what help was needed.

---

## Session 7: Scenario 3 Deep Dive — Amazon Dimensionality and Methodology
**Date:** June 2, 2026
**Chat URL:** https://claude.ai/chat/b4f9c223-7211-4b03-b505-6543211c6c77

**What we discussed:**
Claude fetched and read the Goodfellow et al. paper directly from arXiv, then we had a detailed discussion specifically about Scenario 3 (Dissimilar Tasks: MNIST digits vs. Amazon DVD reviews).

**Main question resolved:**
The two datasets have incompatible input sizes — MNIST is 784-dimensional, Amazon is 5000-dimensional. Two approaches exist:
- **Paper's approach:** PCA/SVD to reduce Amazon from 5000 → 784 dimensions
- **Our implementation:** Zero-padding MNIST from 784 → 5000 dimensions (alternative approach)

Claude helped us decide how to document this in `docs/methodology.md`. The conclusion was:

> What was labeled "Improvement 3 — Shared Vocabulary Feature Selection for Amazon Reviews" is neither an improvement nor a deviation from the paper — it is a faithful reproduction detail. It should be removed from the improvements section entirely and replaced with a plain implementation note.

**Other feedback given:**
- "Improvement 1" (ablation study) would be better labeled as an "extension" rather than an improvement, since it doesn't claim to improve the paper's results
- The "Note on Patience Bias" and deviations table in `methodology.md` were identified as strong elements for academic grading purposes

---

## Session 8: README Figure Sizing
**Date:** June 6, 2026
**Chat URL:** https://claude.ai/chat/25a21a14-ea3d-4b07-b7a7-a843bf09f26b

**What we discussed:**
The README contains side-by-side comparison tables of original paper figures vs. our reproduced figures for all 6 figures. The images were displaying at different visual heights because the original paper figures and our reproduced figures have different aspect ratios, even though both were set to `width="500"`.

**Change made:**
Claude switched all 12 figure image tags from `width="500"` to `height="350"`. Reasoning: constraining by height forces both columns to the same height regardless of aspect ratio.

**Change reverted:**
After reviewing the result, we asked Claude to revert to the original `width="500"` format. Claude restored all 12 image tags.

**Outcome:** No net change to the file — ended up back where we started. The session is documented here as a record of what was tried.

---

## Session 9: Preparing for the Defence — Ablation Study Talking Points
**Date:** June 11, 2026
**Chat URL:** https://claude.ai/chat/19607543-6879-42cc-a6be-d4c7e0a4e767

**What we discussed:**
With the project defence approaching, we asked Claude to read `docs/ablation.md` and produce talking points for what to say when presenting and defending the ablation study findings.

**Ablation study summary (from the file):**

**Ablation 1 — Dropout Rate (p = 0.0, 0.2, 0.5)**
- Forgetting rate drops monotonically as dropout increases
- From p=0.0 to p=0.5: approximately **55% reduction in forgetting rate** (from 0.0287 to 0.0129)
- Confirms the paper's central claim: Dropout forces distributed representations that compete less across tasks
- Note: std at p=0.5 is large (0.277 vs mean 0.176 for best_joint), likely one trial didn't converge — but the forgetting metric itself shows the trend clearly

**Ablation 2 — Weight Decay (λ = 0, 1e-4, 1e-3)**
- No meaningful improvement from L2 regularization
- Key conclusion: Dropout's effect is specific to Dropout, not a general regularization effect. Weight decay is also a regularizer but does not reduce forgetting.

**Recommended defence statement:**
> "The ablation study shows that Dropout is the active mechanism. Weight Decay, which is also a regularizer, does not reduce forgetting — meaning this is not about regularization in general, but specifically about Dropout's formula forcing distributed representations. This strengthens Goodfellow's theoretical explanation."

**Anticipated examiner questions and answers prepared:**

| Question | Answer |
|----------|--------|
| Why 8 trials instead of 25? | Hardware constraints — the paper used 25, but 8 is sufficient to see clear monotonic trends |
| Why is the std at p=0.5 so large? | Likely one outlier trial that didn't converge; the forgetting metric trend is still valid |
| Why measure Forgetting Rate specifically? | It directly measures what the paper claims — not just final performance, but how much of Task A is forgotten after training on Task B |
| What does this add beyond the paper? | The paper shows qualitative curves; the ablation quantifies the dropout effect (55% reduction) and rules out general regularization as the explanation |

---

*End of full conversation log. Total sessions documented: 9. Date range: March 26, 2026 – June 11, 2026.*
