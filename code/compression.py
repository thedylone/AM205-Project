"""Image compression using DWT, DFT, and PCA."""

import time
import numpy as np
import matplotlib.pyplot as plt
import pywt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from sklearn.decomposition import PCA
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm.notebook import tqdm


def dwt_compression(
    img: np.ndarray, ratios: np.ndarray, wavelet="bior4.4", plot=False
):
    """Compress image using Discrete Wavelet Transform (DWT).
    Args:
        img (ndarray): Input image.
        ratios (ndarray): Array of compression ratios.
        wavelet (str): Wavelet type to use.
        plot (bool): Whether to plot the reconstructed images.
    Returns:
        tuple (tuple): CRs, PSNR values, SSIM values, times."""
    start = time.time()
    coeffs = pywt.wavedec2(img, wavelet)  # DWT
    end = time.time()
    c_arr, c_slices = pywt.coeffs_to_array(coeffs)

    # Thresholding
    flat = c_arr.flatten()

    dwt_ratio = np.zeros(ratios.shape)
    dwt_psnr = np.zeros(ratios.shape)
    dwt_ssim = np.zeros(ratios.shape)
    dwt_time = np.full_like(ratios, end - start, dtype=float)

    pbar = tqdm(enumerate(ratios), total=len(ratios), leave=False)
    for i, ratio in pbar:
        pbar.set_description(f"ratio: {ratio:.2f}")
        kf = int(len(flat) / ratio)
        thresh = np.partition(np.abs(flat), -kf)[-kf]
        c_thr = pywt.threshold(c_arr, thresh, mode="hard")

        # Reconstruction
        img_dwt = pywt.waverec2(
            pywt.array_to_coeffs(c_thr, c_slices, output_format="wavedec2"),
            wavelet,
        )

        if plot:
            plt.imshow(img_dwt, cmap="gray")
            plt.axis("off")
            plt.show()

        dwt_ratio[i] = img.size / kf
        dwt_psnr[i] = psnr(img, img_dwt, data_range=255)
        dwt_ssim[i] = ssim(img, img_dwt, data_range=255)

    return dwt_ratio, dwt_psnr, dwt_ssim, dwt_time


def dft_compression(img, ratios, plot=False):
    """Compress image using Discrete Fourier Transform (DFT).
    Args:
        img (ndarray): Input image.
        ratios (ndarray): Array of compression ratios.
        plot (bool): Whether to plot the reconstructed images.
    Returns:
        tuple (tuple): CRs, PSNR values, SSIM values, times."""
    start = time.time()
    freqs = fftshift(fft2(img))
    end = time.time()

    # Thresholding
    flat = np.abs(freqs).flatten()

    dft_ratio = np.zeros(ratios.shape)
    dft_psnr = np.zeros(ratios.shape)
    dft_ssim = np.zeros(ratios.shape)
    dft_time = np.full_like(ratios, end - start, dtype=float)

    pbar = tqdm(enumerate(ratios), total=len(ratios), leave=False)
    for i, ratio in pbar:
        pbar.set_description(f"ratio: {ratio:.2f}")
        kf = int(len(flat) / ratio)
        thresh = np.partition(flat, -kf)[-kf]
        freqs_thr = pywt.threshold(freqs, thresh, mode="hard")

        img_dft = np.real(ifft2(ifftshift(freqs_thr)))
        if plot:
            plt.imshow(img_dft, cmap="gray")
            plt.axis("off")
            plt.show()

        dft_ratio[i] = img.size / kf
        dft_psnr[i] = psnr(img, img_dft, data_range=255)
        dft_ssim[i] = ssim(img, img_dft, data_range=255)

    return dft_ratio, dft_psnr, dft_ssim, dft_time


def pca_compression(img, ratios, plot=False):
    """Compress image using Principal Component Analysis (PCA).
    Args:
        img (ndarray): Input image.
        ratios (ndarray): Array of compression ratios.
        plot (bool): Whether to plot the reconstructed images.
    Returns:
        tuple (tuple): CRs, PSNR values, SSIM values, times."""
    h, w = img.shape
    pca_ratio, pca_psnr, pca_ssim, pca_time = [], [], [], []
    seen = set()

    for ratio in (pbar := tqdm(ratios, total=len(ratios), leave=False)):
        pbar.set_description(f"ratio: {ratio:.2f}")
        n_components = max(1, int(min(h, w, img.size / (h + w)) / ratio))
        if n_components in seen:
            continue
        seen.add(n_components)
        pca = PCA(n_components=n_components)
        start = time.time()
        Z = pca.fit_transform(img)
        end = time.time()
        pca_time.append(end - start)
        img_pca = pca.inverse_transform(Z)

        if plot:
            plt.imshow(img_pca, cmap="gray")
            plt.axis("off")
            plt.show()

        pca_ratio.append(img.size / (n_components * (h + w)))
        pca_psnr.append(psnr(img, img_pca, data_range=255))
        pca_ssim.append(ssim(img, img_pca, data_range=255))

    return pca_ratio, pca_psnr, pca_ssim, pca_time


def run_compression_experiment(img, ratios):
    """Run compression experiments using DWT, DFT, and PCA.
    Args:
        img (ndarray): Input image.
        ratios (ndarray): Array of compression ratios.
    Returns:
        tuple: PSNR results, SSIM results, time results."""

    def crop_to_even(img):
        h, w = img.shape[:2]
        new_h = h if h % 2 == 0 else h - 1
        new_w = w if w % 2 == 0 else w - 1
        return img[:new_h, :new_w]

    img = crop_to_even(img)
    psnr_results, ssim_results, time_results = {}, {}, {}
    c_func = {
        "DWT": dwt_compression,
        "DFT": dft_compression,
        "PCA": pca_compression,
    }

    for method in (pbar := tqdm(c_func.keys(), leave=False)):
        pbar.set_description(f"Method - {method}")
        crs, psnrs, ssims, times = c_func[method](img, ratios)
        psnr_results[method] = (crs, psnrs)
        ssim_results[method] = (crs, ssims)
        time_results[method] = (crs, times)

    return psnr_results, ssim_results, time_results
