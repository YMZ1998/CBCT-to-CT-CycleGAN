import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_random_images(folder, num_samples=5):
    image_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    images = [Image.open(os.path.join(folder, f)) for f in selected_files]
    return images


def load_random_npy_files(folder, num_samples=5):
    npy_files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    selected_files = random.sample(npy_files, min(num_samples, len(npy_files)))
    arrays = [np.load(os.path.join(folder, f)) for f in selected_files]
    for i in range(len(arrays)):
        arrays[i] = normalize_to_range(arrays[i])
    return arrays


def normalize_to_range(data):
    data = np.clip(data, -800, 1500)
    return data

def plot_images(ct_images, cbct_images):
    num_samples = min(len(ct_images), len(cbct_images))
    plt.figure(figsize=(10, 3 * num_samples))

    for i in range(num_samples):
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(ct_images[i], cmap='gray')
        plt.title(f'CT Image {i + 1}')
        plt.axis('off')

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(cbct_images[i], cmap='gray')
        plt.title(f'CBCT Image {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataset_dir = r'../datasets/thorax-512'
    trainA_dir = os.path.join(dataset_dir, 'train', 'A')
    trainB_dir = os.path.join(dataset_dir, 'train', 'B')

    num_samples = 4
    cbct_images = load_random_npy_files(trainA_dir, num_samples)
    ct_images = load_random_npy_files(trainB_dir, num_samples)

    plot_images(ct_images, cbct_images)
