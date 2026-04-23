🧠 Catastrophic Forgetting – Reproduction Project
This repository contains the source code and documentation for an academic reproduction of results regarding Catastrophic Forgetting in neural networks. The study specifically analyzes the balance between learning new tasks and retaining prior knowledge across sequential training stages.

📂 Project Navigation
In accordance with academic documentation requirements, the following files detail the project's methodology and findings:

Final Takeaways: A reflective analysis (1–2 pages) detailing personal conclusions, interpretation of reproduced results, and insights gained regarding the learning process.

AI Usage & Methodology: Documentation of the integration of AI tools for code scaffolding and initial planning, including the human-in-the-loop verification process.

Algorithmic Thinking: A modular breakdown of the project stages, implementation logic, and specific verification protocols used for each phase.

AI Work Plans: Raw planning documents generated during the initial design phase, used to structure the project's development.

🛠 Project Structure
The repository is organized to facilitate peer review and modular testing:

Data Preprocessing: prepare_amazon_npz.py – Prepares and formats the Amazon Reviews dataset for continual learning scenarios.

Reproduction Engine: pytorch_reproduction_suite.py – Implements the core training loops and forgetting mitigation strategies (e.g., EWC, Dropout).

Visualization: plot_results.py – Generates comparative visualizations based on experimental logs.

⚙️ Controlled Experimental Setup
While exact numerical identity across different hardware environments is rarely feasible, this project employs a controlled setup to ensure that the main qualitative trends reported in the original paper can be recovered.

Experimental Pipeline: Hypothesis → Implementation → Testing → Refinement → Comparison → Reproduction.

Fixed Random Seed: 42 (applied to PyTorch and NumPy) to minimize stochastic variance between local runs.

Hyperparameter Configuration:

Search Ranges: Parameters such as Learning Rate and Weight Decay were constrained to ranges specified in the paper's appendix.

Early Stopping: A patience configuration of [insert value] was implemented to prevent overfitting and ensure fair comparison across tasks.

Variability Reduction: The results presented are based on repeated runs to ensure that the reported trends are robust and not artifacts of a single initialization.

🚀 Execution Guide
To reproduce the experimental results, execute the following scripts in sequence:

Bash
# Step 1: Preprocess the raw dataset
python prepare_amazon_npz.py

# Step 2: Execute the training and mitigation suite
python pytorch_reproduction_suite.py

# Step 3: Generate comparative figures
python plot_results.py
📊 Expected Qualitative Outputs
The execution generates data logs and figures intended to align with the primary trends observed in the original study:

Data Logs: Saved in results/*.csv.

Visualizations:

figure1.png: MNIST Permutation results.

figure2.png: Similar Tasks (Amazon Reviews).

figure3.png: Dissimilar Tasks.

These outputs are expected to demonstrate the performance trade-off between task plasticity and memory stability, consistent with the paper's conclusions.
