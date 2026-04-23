import torch

data = torch.load("results_fixed/scenario_1_checkpoint.pt")

results = data["results"]

for key, points in results.items():
    print(f"{key}: {len(points)} points")