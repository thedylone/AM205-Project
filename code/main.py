"""Main script to run image compression experiments."""

import glob
import numpy as np
import scipy
import matplotlib.pyplot as plt

from compression import (
    dwt_compression,
    dft_compression,
    pca_compression,
    eval_compressions,
    methods,
)
from sample_run import compress_images, ratios
from plot_results import plot_results_ci


def single_image_run(img):
    """Run compression experiment on a single image and print results."""
    method_funcs = [
        ("DWT", dwt_compression),
        ("DFT", dft_compression),
        ("PCA", pca_compression),
    ]
    for name, func in method_funcs:
        ratio, psnr, ssim, t = func(img, np.array([50]), plot=True)
        print(
            f"{name} →  Compression Ratio: {ratio[0]:.2f} | "
            f"PSNR: {psnr[0]:.2f} dB | "
            f"SSIM: {ssim[0]:.3f} | "
            f"Time: {t[0]:.3f}s"
        )


def single_image_runs(img):
    """Run compression experiment on a single image and print results."""
    face_psnr_results, face_ssim_results, face_time_results = (
        eval_compressions(img, ratios)
    )
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    for method in methods:
        cr = face_psnr_results[method][0]
        psnr = face_psnr_results[method][1]
        axs[0].scatter(cr, psnr, marker="o", label=method, s=16)

        ssim = face_ssim_results[method][1]
        axs[1].scatter(cr, ssim, marker="o", label=method, s=16)

        times = face_time_results[method][1]
        axs[2].scatter(cr, times, marker="o", label=method, s=16)

    for ax in axs:
        ax.set_xlabel("Compression Ratio")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1, None)
        ax.grid(True)
    axs[0].set_ylabel("PSNR (dB)")
    axs[1].set_ylabel("SSIM")
    axs[2].set_ylabel("Time(s)")
    plt.legend()
    plt.show()


def batch_image_run(images, results_files, plot_filename):
    """Run batch compression experiment on a set of images and plot results."""
    psnr_res, ssim_res, time_res = compress_images(
        images, ratios, results_files
    )
    plot_results_ci(psnr_res, ssim_res, time_res, plot_filename)


def load_high_res_dataset():
    """Load high-resolution image dataset."""

    def rgb_to_grayscale(images):
        """
        images: numpy array of shape (32, 32, 3)
        returns: numpy array of shape (32, 32) in grayscale
        """
        # coefficients for R, G, B
        coeffs = np.array([0.299, 0.587, 0.114]).reshape(1, 1, 3)
        gray = np.sum(images * coeffs, axis=2)
        return gray.astype(images.dtype)

    images = []
    files = glob.glob("lossless-images/*.png")
    for f in files:
        _img = plt.imread(f)
        _img = rgb_to_grayscale(_img)
        images.append(_img)
    return images


def load_med_res_dataset():
    """Load medium-resolution image dataset."""

    images = []
    for i in range(1, 25):
        images.append(plt.imread(f"kodak/{i:04d}.tif"))
    return images


if __name__ == "__main__":
    test_img = scipy.datasets.face(gray=True)
    single_image_run(test_img)
    single_image_runs(test_img)

    high_res_images = load_high_res_dataset()
    batch_image_run(
        high_res_images,
        results_files={
            "psnr": "results/high_res_psnr.pkl",
            "ssim": "results/high_res_ssim.pkl",
            "time": "results/high_res_time.pkl",
        },
        plot_filename="plots/high_res.pdf",
    )
    med_res_images = load_med_res_dataset()
    batch_image_run(
        med_res_images,
        results_files={
            "psnr": "results/med_res_psnr.pkl",
            "ssim": "results/med_res_ssim.pkl",
            "time": "results/med_res_time.pkl",
        },
        plot_filename="plots/med_res.pdf",
    )
