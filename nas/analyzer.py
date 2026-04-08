# nas/analyzer.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Resolve the project root relative to this file, regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = Path("E:/B-Tech/Projects/DL-snn/results/phase3_results_CartPole-v1.json")


def load_phase3_results():
    with open(str(RESULTS_PATH), "r") as f:
        data = json.load(f)
    return data


def plot_trial_results():
    data = load_phase3_results()
    df = pd.DataFrame(data["trials"])

    # Map some numeric columns for plotting
    df["hidden_size"] = df["params"].apply(lambda d: d["hidden_sizes"])
    df["timesteps"] = df["params"].apply(lambda d: d["timesteps"])
    df["neuron_type"] = df["params"].apply(lambda d: d["neuron_type"])

    # Plot 1: reward vs hidden size
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    df.boxplot("value", by="hidden_size", ax=plt.gca())
    plt.title("Reward vs hidden size")
    plt.ylabel("episode return")

    # Plot 2: reward vs timesteps
    plt.subplot(1, 3, 2)
    df.boxplot("value", by="timesteps", ax=plt.gca())
    plt.title("Reward vs timesteps")
    plt.ylabel("episode return")

    # Plot 3: reward vs neuron type
    plt.subplot(1, 3, 3)
    df.boxplot("value", by="neuron_type", ax=plt.gca())
    plt.title("Reward vs neuron type")
    plt.ylabel("episode return")

    plt.tight_layout()
    plt.show()

    # Best configuration
    best = data["best_params"]
    best_value = data["best_value"]
    print("Best reward:", best_value)
    print("Best config:", best)
    return best, best_value
