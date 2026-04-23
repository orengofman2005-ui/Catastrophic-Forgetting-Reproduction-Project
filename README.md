# Catastrophic Forgetting – Reproduction Project

This repository contains code and documentation for reproducing key results from the paper on catastrophic forgetting.

## Documentation

### Final Takeaways
Contains personal conclusions and interpretation of the reproduced results.

File:
docs/takeaway.md

### AI Usage & Reproducibility
Documents:
- how AI assisted planning, debugging and verification
- prompt examples used during development
- human-in-the-loop validation process
- reproducibility settings

File:
docs/ai_usage.md

---

## Project Structure

Data preparation  
prepare_amazon_npz.py

Training / mitigation engine  
pytorch_reproduction_suite.py

Visualization  
plot_results.py

---

## Reproducibility Settings

Fixed random seed:
42

Controlled parameters:
- learning rate: [fill yours]
- epochs: [fill yours]
- patience: [fill yours]

---

## Run

Step 1:
python prepare_amazon_npz.py

Step 2:
python pytorch_reproduction_suite.py

Step 3:
python plot_results.py

---

## Expected Outputs

Generated files:

results/*.csv

figures:
- figure1.png
- figure2.png
- figure3.png

These outputs should reproduce the main qualitative trends reported in the original paper.
