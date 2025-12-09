"""Statistic functions for image analysis"""

import numpy as np
from sklearn.decomposition import PCA


def rgb_to_grayscale(images):
    """images: numpy array of shape (N, H, W, 3)
    returns: numpy array of shape (N, H, W) in grayscale"""
    # coefficients for R, G, B
    coeffs = np.array([0.299, 0.587, 0.114]).reshape(1, 1, 1, 3)
    gray = np.sum(images * coeffs, axis=3)
    return gray.astype(images.dtype)


def covariance_vs_distance_2d(cov_matrix, h=32, w=32, bin_size=1.0):
    """cov_matrix: (H*W, H*W) pixel covariance matrix
    Returns:
        distances: array of distance bin centers
        cov_by_dist: averaged covariance values in each bin"""

    coords = np.array([(i, j) for i in range(h) for j in range(w)])
    diffs = coords[:, None, :] - coords[None, :, :]

    dx = np.abs(diffs[..., 0])
    dy = np.abs(diffs[..., 1])

    dx = np.minimum(dx, h - dx)
    dy = np.minimum(dy, w - dy)

    dists = np.sqrt(dx**2 + dy**2)

    # maximum possible distance on 32x32 grid
    max_dist = np.sqrt(0.25 * (h - 1) ** 2 + 0.25 * (w - 1) ** 2)

    # distance bins
    num_bins = int(np.ceil(max_dist / bin_size))
    edges = np.linspace(0, max_dist, num_bins + 1)

    cov_by_dist = np.zeros(num_bins)
    counts = np.zeros(num_bins)

    # assign each pixel pair to a distance bin
    bin_idx = np.digitize(dists, edges) - 1  # 0..num_bins-1

    for b in range(num_bins):
        mask = bin_idx == b
        cov_by_dist[b] = np.mean(cov_matrix[mask])
        counts[b] = np.sum(mask)

    # distance = midpoint of each bin
    dist_centers = 0.5 * (edges[:-1] + edges[1:])

    return dist_centers, cov_by_dist, counts


def retrieve_singular_values(data_batch, n_components=300):
    """data_batch: numpy array of shape (N, H, W, C)"""
    x = data_batch.reshape(len(data_batch), -1)

    # Normalize
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean

    # PCA (partial, because full is expensive)
    pca = PCA(n_components=n_components)  # enough to show low-rank structure
    pca.fit(x_centered)

    return pca.singular_values_
