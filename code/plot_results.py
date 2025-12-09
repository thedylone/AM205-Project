"""Plotting functions for compression results."""

import matplotlib.pyplot as plt
import numpy as np


colours = {"DWT": "tab:blue", "DFT": "tab:orange", "PCA": "tab:green"}
labels = ["PSNR (dB)", "SSIM", "Time(s)"]


def calculate_mean_ci(x, y, bins):
    """Calculate mean and 95% confidence intervals for y in bins of x.

    Args:
        x (ndarray): x values.
        y (ndarray): y values.
        bins (ndarray): Bin edges.

    Returns:
        tuple: bin centers, means, confidence intervals."""
    bin_centers = (bins[:-1] + bins[1:]) / 2
    means = []
    cis = []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        y_bin = y[mask]
        if len(y_bin) > 0:
            mean = np.mean(y_bin)
            se = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
            ci = 1.96 * se
        else:
            mean = np.nan
            ci = np.nan
        means.append(mean)
        cis.append(ci)
    return bin_centers, np.array(means), np.array(cis)


def plot_results_ci(psnr_results, ssim_results, time_results, filename):
    """Plot results with 95% confidence intervals.

    Args:
        psnr_results (dict): PSNR results for each method.
        ssim_results (dict): SSIM results for each method.
        time_results (dict): Time results for each method."""
    _, axs = plt.subplots(1, 3, figsize=(9, 3))
    for method, color in colours.items():
        crs = np.hstack(psnr_results[method][::2])
        results_list = [
            np.hstack(psnr_results[method][1::2]),  # PSNR
            np.hstack(ssim_results[method][1::2]),  # SSIM
            np.hstack(time_results[method][1::2]),  # Time
        ]
        n = 10 if method == "PCA" else 20
        bins = np.logspace(np.log10(np.min(crs)), np.log10(np.max(crs)), n)

        for j, y in enumerate(results_list):
            bin_centers, means, cis = calculate_mean_ci(crs, y, bins)
            axs[j].plot(bin_centers, means, color=color, label=method)
            axs[j].fill_between(
                bin_centers, means - cis, means + cis, color=color, alpha=0.2
            )

    for j, ax in enumerate(axs):
        ax.set_xlabel("Compression Ratio")
        ax.set_xscale("log")
        ax.set_xlim(1, 5e3)
        ax.grid(True)
        ax.set_ylabel(labels[j])
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
