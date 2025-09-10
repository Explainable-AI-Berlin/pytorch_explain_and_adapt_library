import matplotlib.pyplot as plt

# Data organized in a dict for easy extension
data = {
    "Square": {
        "Original": [92.8, 90.4, 90.4, 83.6, 55.3, 50.3],
        "DFR": [92.8, 91.3, 90.6, 85.1, 59.9, 50.3],
        "RR-ClarC": [92.8, 96.7, 95.2, 94.0, 82.0, 50.3],
        "CFKD": [95.7, 95.2, 96.7, 96.3, 96.0, 95.8],
    },
    "CelebA Smiling vs Copyrighttag": {
        "Original": [70.7, 73.0, 70.7, 67.1, 56.9, 51.6],
        "DFR": [70.7, 73.0, 71.0, 67.9, 57.7, 51.6],
        "RR-ClarC": [70.7, 40.9, 40.9, 40.9, 40.9, 51.6],
        "CFKD": [86.4, 85.9, 84.5, 83.8, 84.3, 80.2],
    }
}

# Poisoning levels (x-axis)
poisoning_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Colors for methods
colors = {
    "Original": "tab:blue",
    "DFR": "tab:orange",
    "RR-ClarC": "tab:purple",
    "CFKD": "tab:green",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, (dataset, methods) in zip(axes, data.items()):
    for method, values in methods.items():
        ax.plot(poisoning_levels, values, marker="o", label=method, color=colors.get(method, None))
    ax.set_title(dataset)
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

fig.suptitle("Performance under Different Poisoning Levels")
plt.tight_layout()
plt.savefig("poisoning_effect_comparison.png", dpi=300)
