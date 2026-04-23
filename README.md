# Catastrophic Forgetting - Reproduction Project

This repository contains the source code and documentation for reproducing the results of the paper regarding Catastrophic Forgetting.

## 📖 Essential Documentation
* **[Final Takeaways (Hebrew)](./takeaways.md)** - My personal reflection and project conclusions (1-2 pages).
* **[AI Usage & Process](./docs/ai_usage.md)** - Documentation of how AI was used to plan and verify the project.

## 🛠️ Project Structure & Logic (Algorithmic Thinking)
1. **Data Prep:** `prepare_amazon_npz.py` - Processes the dataset for training.
2. **Main Engine:** `pytorch_reproduction_suite.py` - The core logic for training and forgetting mitigation.
3. **Visualization:** `plot_results.py` - Generates graphs to compare with the original paper's results.

## 🚀 How to Run
```bash
python pytorch_reproduction_suite.py
python plot_results.py
