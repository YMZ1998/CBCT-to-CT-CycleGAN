import os
import numpy as np
import random
import matplotlib.pyplot as plt
from visdom import Visdom


def load_npy_files(folder_path):
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {folder_path}")

    images = []
    for file in npy_files[::3]:
        file_path = os.path.join(folder_path, file)
        img_array = np.load(file_path)
        images.append(img_array)

    return np.stack(images)


def plot_slice(ct_slice, cbct_slice):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(ct_slice, cmap='gray')
    plt.title('CT Slice')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cbct_slice, cmap='gray')
    plt.title('CBCT Slice')
    plt.axis('off')

    plt.show()


def visualize_random_slices(ct_data, cbct_data, num_samples=3):
    depth = ct_data.shape[0]
    slice_indices = random.sample(range(depth), num_samples)



    for idx in slice_indices:
        ct_slice = ct_data[idx]
        cbct_slice = cbct_data[idx]

        ct_slice = normalize_to_range(ct_slice)
        cbct_slice = normalize_to_range(cbct_slice)

        plot_slice(ct_slice, cbct_slice)


def normalize_to_range(data):
    data = np.clip(data, -1000, 1000)
    data_min, data_max = np.min(data), np.max(data)
    print(data_min, data_max)
    data = (data - data_min) / (data_max - data_min + 1e-8)
    return data


def visualize_with_visdom(ct_data, cbct_data, num_samples=3):
    viz = Visdom()

    if not viz.check_connection():
        raise ConnectionError("Visdom server not running. Please start the server.")

    depth = min(ct_data.shape[0], cbct_data.shape[0])
    slice_indices = random.sample(range(depth), num_samples)

    for idx in slice_indices:
        ct_slice = ct_data[idx]
        cbct_slice = cbct_data[idx]

        ct_slice_normalized = normalize_to_range(ct_slice)
        cbct_slice_normalized = normalize_to_range(cbct_slice)

        viz.image(
            ct_slice_normalized,
            opts=dict(title=f'CT Slice {idx}', caption='CT Slice'),
        )

        viz.image(
            cbct_slice_normalized,
            opts=dict(title=f'CBCT Slice {idx}', caption='CBCT Slice'),
        )


def check():
    data_dir = r'D:\Data\cbct_ct\pelvis\2PA055'
    # data_dir = r'D:\Data\cbct_ct\brain\2BA004'
    ct_path = os.path.join(data_dir, 'ct')
    cbct_path = os.path.join(data_dir, 'cbct')

    ct_data = load_npy_files(ct_path)
    cbct_data = load_npy_files(cbct_path)

    print(f"CT shape: {ct_data.shape}")
    print(f"CBCT shape: {cbct_data.shape}")

    visualize_random_slices(ct_data, cbct_data, num_samples=5)
    # visualize_with_visdom(ct_data, cbct_data, num_samples=5)


if __name__ == '__main__':
    check()
