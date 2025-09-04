import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# Data organized in a dict for easy extension
data = {
    "Square": {
        "Original": [51.1, 58.0, 74.1, 84.4],
        "DFR": [52.1, 63.5, 84.2, 91.5],
        "CFKD": [96.5, 96.6, 97.3, 97.9],
    },
    "CelebA Smiling vs Copyrighttag": {
        "Original": [51.4, 52.4, 52.0, 52.2],
        "DFR": [53.0, 51.8, 51.9, 61.1],
        "CFKD": [86.7, 88.6, 89.7, 89.8],
    }
}

# Sample sizes (x-axis)
sample_sizes = [1000, 2000, 4000, 8000]

# Colors for methods
colors = {
    "Original": "tab:blue",
    "DFR": "tab:orange",
    "CFKD": "tab:green",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, (dataset, methods) in zip(axes, data.items()):
    for method, values in methods.items():
        ax.plot(sample_sizes, values, marker="o", label=method, color=colors.get(method, None))
    ax.set_title(dataset)
    ax.set_xlabel("Sample Size (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log")
    # Force only custom ticks
    ax.xaxis.set_major_locator(FixedLocator(sample_sizes))
    ax.set_xticklabels(sample_sizes)
    ax.set_xticks(sample_sizes)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

fig.suptitle("Performance under Different Sample Sizes")
plt.tight_layout()
plt.show()


plt.savefig("sample_size_effect.png", dpi=300)
