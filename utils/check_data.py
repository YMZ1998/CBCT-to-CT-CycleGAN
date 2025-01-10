import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image


def load_npy_data(file_path):
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"{file_path} does not exist.")


def load_png_images(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    images = []
    for file in image_files:
        img = Image.open(os.path.join(folder_path, file))
        img_array = np.array(img)  # 转换为 NumPy 数组
        images.append(img_array)
    return np.stack(images)


def plot_slice(ct_slice, cbct_slice, mask_slice):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(ct_slice, cmap='gray')
    plt.title('CT Slice')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cbct_slice, cmap='gray')
    plt.title('CBCT Slice')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mask_slice, cmap='gray')
    plt.title('Mask Slice')
    plt.axis('off')

    plt.show()


def plot_overlay(ct_slice, mask_slice):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(ct_slice, cmap='gray')
    plt.title('CT Slice')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(mask_slice, cmap='jet', alpha=0.5)  # 使用 jet 颜色映射，alpha 控制透明度
    plt.title('CT + Mask Overlay')
    plt.axis('off')

    plt.show()


def visualize_random_slices(ct_data, cbct_data, mask_data, num_samples=3):
    depth = ct_data.shape[0]
    slice_indices = random.sample(range(depth), num_samples)

    for idx in slice_indices:
        ct_slice = ct_data[idx]
        cbct_slice = cbct_data[idx]
        mask_slice = mask_data[idx]

        plot_slice(ct_slice, cbct_slice, mask_slice)

        # plot_overlay(ct_slice, mask_slice)


def check():
    data_dir = r'D:\Data\cbct_ct\brain\2BA006'
    ct_path = os.path.join(data_dir, 'ct')
    cbct_path = os.path.join(data_dir, 'cbct')
    mask_path = os.path.join(data_dir, 'mask')

    ct_data = load_png_images(ct_path)
    cbct_data = load_png_images(cbct_path)
    mask_data = load_png_images(mask_path)

    print(f"CT shape: {ct_data.shape}")
    print(f"CBCT shape: {cbct_data.shape}")
    print(f"Mask shape: {mask_data.shape}")

    visualize_random_slices(ct_data, cbct_data, mask_data, num_samples=5)


if __name__ == '__main__':
    check()
