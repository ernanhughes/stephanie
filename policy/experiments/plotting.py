import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rolling_stats(values, window=50):
    values = np.asarray(values)
    means = []
    variances = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i+1]
        means.append(np.mean(window_vals))
        variances.append(np.var(window_vals))

    return np.array(means), np.array(variances)


def plot_trajectories(baseline, bounded, out_dir):
    ensure_dir(out_dir)

    plt.figure(figsize=(12, 6))
    plt.plot(baseline, alpha=0.6, label="Baseline")
    plt.plot(bounded, alpha=0.8, label="Policy Bounded")
    plt.title("Energy Trajectory Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory.png"))
    plt.close()


def plot_rolling_variance(baseline, bounded, out_dir, window=50):
    ensure_dir(out_dir)

    _, var_base = rolling_stats(baseline, window)
    _, var_bound = rolling_stats(bounded, window)

    plt.figure(figsize=(12, 6))
    plt.plot(var_base, label="Baseline Variance")
    plt.plot(var_bound, label="Bounded Variance")
    plt.title(f"Rolling Variance (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_variance.png"))
    plt.close()


def plot_rolling_mean(baseline, bounded, out_dir, window=50):
    ensure_dir(out_dir)

    mean_base, _ = rolling_stats(baseline, window)
    mean_bound, _ = rolling_stats(bounded, window)

    plt.figure(figsize=(12, 6))
    plt.plot(mean_base, label="Baseline Mean")
    plt.plot(mean_bound, label="Bounded Mean")
    plt.title(f"Rolling Mean (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_mean.png"))
    plt.close()


def plot_histogram(baseline, bounded, out_dir):
    ensure_dir(out_dir)

    plt.figure(figsize=(10, 6))
    plt.hist(baseline, bins=40, alpha=0.5, label="Baseline")
    plt.hist(bounded, bins=40, alpha=0.5, label="Bounded")
    plt.title("Energy Distribution")
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distribution.png"))
    plt.close()
