import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

def load_npy_data(file_path):
    """
    加载 .npy 文件并返回 NumPy 数组。

    Args:
        file_path (str): .npy 文件路径。

    Returns:
        np.ndarray: 加载的数据。
    """
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"{file_path} does not exist.")

def load_png_images(folder_path):
    """
    从文件夹中加载 PNG 图像并返回 NumPy 数组。

    Args:
        folder_path (str): 包含 PNG 文件的文件夹路径。

    Returns:
        np.ndarray: 图像数据，形状为 (num_images, height, width)。
    """
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    images = []
    for file in image_files:
        img = Image.open(os.path.join(folder_path, file))
        img_array = np.array(img)  # 转换为 NumPy 数组
        images.append(img_array)
    return np.stack(images)

def plot_slice(ct_slice, cbct_slice, mask_slice):
    """
    可视化 CT、CBCT 和 Mask 的切片。

    Args:
        ct_slice (np.ndarray): CT 切片，形状为 (height, width)。
        cbct_slice (np.ndarray): CBCT 切片，形状为 (height, width)。
        mask_slice (np.ndarray): Mask 切片，形状为 (height, width)。
    """
    plt.figure(figsize=(15, 5))

    # 显示 CT 切片
    plt.subplot(1, 3, 1)
    plt.imshow(ct_slice, cmap='gray')
    plt.title('CT Slice')
    plt.axis('off')

    # 显示 CBCT 切片
    plt.subplot(1, 3, 2)
    plt.imshow(cbct_slice, cmap='gray')
    plt.title('CBCT Slice')
    plt.axis('off')

    # 显示 Mask 切片
    plt.subplot(1, 3, 3)
    plt.imshow(mask_slice, cmap='gray')
    plt.title('Mask Slice')
    plt.axis('off')

    plt.show()

def plot_overlay(ct_slice, mask_slice):
    """
    将 Mask 叠加到 CT 切片上并可视化。

    Args:
        ct_slice (np.ndarray): CT 切片，形状为 (height, width)。
        mask_slice (np.ndarray): Mask 切片，形状为 (height, width)。
    """
    plt.figure(figsize=(10, 5))

    # 显示 CT 切片
    plt.subplot(1, 2, 1)
    plt.imshow(ct_slice, cmap='gray')
    plt.title('CT Slice')
    plt.axis('off')

    # 显示叠加效果
    plt.subplot(1, 2, 2)
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(mask_slice, cmap='jet', alpha=0.5)  # 使用 jet 颜色映射，alpha 控制透明度
    plt.title('CT + Mask Overlay')
    plt.axis('off')

    plt.show()

def visualize_random_slices(ct_data, cbct_data, mask_data, num_samples=3):
    """
    随机抽取并可视化 CT、CBCT 和 Mask 的切片。

    Args:
        ct_data (np.ndarray): CT 数据，形状为 (depth, height, width)。
        cbct_data (np.ndarray): CBCT 数据，形状为 (depth, height, width)。
        mask_data (np.ndarray): Mask 数据，形状为 (depth, height, width)。
        num_samples (int): 抽取的切片数量。
    """
    depth = ct_data.shape[0]
    slice_indices = random.sample(range(depth), num_samples)

    for idx in slice_indices:
        ct_slice = ct_data[idx]
        cbct_slice = cbct_data[idx]
        mask_slice = mask_data[idx]

        # 可视化单一切片
        plot_slice(ct_slice, cbct_slice, mask_slice)

        # 可视化叠加效果
        # plot_overlay(ct_slice, mask_slice)

def main():
    # 数据路径
    data_dir = r'D:\Data\cbct_ct\2BA006'  # 替换为你的数据路径
    ct_path = os.path.join(data_dir, 'ct')  # CT 数据路径
    cbct_path = os.path.join(data_dir, 'cbct')  # CBCT 数据路径
    mask_path = os.path.join(data_dir, 'mask')  # Mask 数据路径

    ct_data = load_png_images(ct_path)
    cbct_data = load_png_images(cbct_path)
    mask_data = load_png_images(mask_path)

    print(f"CT shape: {ct_data.shape}")
    print(f"CBCT shape: {cbct_data.shape}")
    print(f"Mask shape: {mask_data.shape}")

    # 可视化随机切片
    visualize_random_slices(ct_data, cbct_data, mask_data, num_samples=5)

if __name__ == '__main__':
    main()