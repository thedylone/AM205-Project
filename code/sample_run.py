"""Sample Run for High-res Dataset and Save Results"""

import os
import pickle
from tqdm.notebook import tqdm
import numpy as np
from compression import run_compression_experiment


ratios = np.logspace(0, 4, 100)[1:]  # Compression ratios from 10^0 to 10^4
"""Compression ratios to test."""


def compress_images(images, crs, results_files):
    """Compress images and save results.
    Args:
        images (list): List of images to compress.
        crs (ndarray): Array of compression ratios.
        results_files (dict): Dictionary of result file paths.
    Returns:
        tuple: PSNR results, SSIM results, time results."""
    if os.path.exists(results_files["psnr"]):
        with open(results_files["psnr"], "rb") as f:
            psnr_results = pickle.load(f)
        with open(results_files["ssim"], "rb") as f:
            ssim_results = pickle.load(f)
        with open(results_files["time"], "rb") as f:
            time_results = pickle.load(f)
        return psnr_results, ssim_results, time_results

    methods = ["DWT", "DFT", "PCA"]
    psnr_results = {m: [] for m in methods}
    ssim_results = {m: [] for m in methods}
    time_results = {m: [] for m in methods}
    for img in tqdm(images, leave=False, desc="images"):
        psnr_res, ssim_res, time_res = run_compression_experiment(img, crs)
        for method in methods:
            psnr_results[method] += psnr_res[method]
            ssim_results[method] += ssim_res[method]
            time_results[method] += time_res[method]
    with open(results_files["psnr"], "wb") as f:
        pickle.dump(psnr_results, f)
    with open(results_files["ssim"], "wb") as f:
        pickle.dump(ssim_results, f)
    with open(results_files["time"], "wb") as f:
        pickle.dump(time_results, f)
    return psnr_results, ssim_results, time_results
