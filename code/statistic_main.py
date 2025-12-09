"""Statistical analysis main script for CIFAR-10 and STL-10 datasets."""

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from statistic_functions import (
    rgb_to_grayscale,
    covariance_vs_distance_2d,
    retrieve_singular_values,
)

# 1. Load CIFAR-10 and STL-10 as numpy array
transform = T.Compose([T.ToTensor()])
dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
dataset_stl = torchvision.datasets.STL10(
    root="./data", split="train", download=True, transform=transform
)

# Convert to array (50000, W, H , 3)
data_batch = torch.stack([img for img, _ in dataset]).numpy()
data_batch = data_batch.transpose(0, 2, 3, 1)
data_batch_stl = torch.stack([img for img, _ in dataset_stl]).numpy()
data_batch_stl = data_batch_stl.transpose(0, 2, 3, 1)

gray_images = rgb_to_grayscale(data_batch)
gray_images_stl = rgb_to_grayscale(data_batch_stl)

mean = np.sum(gray_images, axis=0)
mean = mean / 50000
mean_stl = np.sum(gray_images_stl, axis=0)
mean_stl = mean_stl / 5000

centered_data = gray_images - mean
flattened_data = centered_data.reshape(50000, 32 * 32)

centered_data_stl = gray_images_stl - mean_stl
flattened_data_stl = centered_data_stl.reshape(5000, 96 * 96)

cov = flattened_data.T @ flattened_data
cov = cov / 50000

cov_stl = flattened_data_stl.T @ flattened_data_stl
cov_stl = cov_stl / 5000

dist, cov_values, counts = covariance_vs_distance_2d(cov)
dist_stl, cov_values_stl, counts_stl = covariance_vs_distance_2d(
    cov_stl, 96, 96, 1.0
)
plt.figure(figsize=(4, 4))
plt.plot(dist, cov_values, color="C1", label="CIFAR-10")
plt.plot(dist_stl, cov_values_stl, label="STL-10")
plt.xlabel("Euclidean distance (pixels)")
plt.ylabel("Average covariance")
plt.grid(True)
# plt.title("2D Pixel Covariance vs Spatial Distance")
plt.legend()
plt.show()

singular_values = retrieve_singular_values(data_batch)
singular_values_stl = retrieve_singular_values(data_batch_stl)

# Singular Values
plt.figure(figsize=(4, 4))
plt.loglog(singular_values, color="C1", label="CIFAR-10")
plt.loglog(singular_values_stl, label="STL-10")
# plt.title("Singular value spectrum (loglog scale)")
plt.xlabel("Component index")
plt.ylabel("Singular value")
plt.grid(True)
plt.legend()
plt.show()
