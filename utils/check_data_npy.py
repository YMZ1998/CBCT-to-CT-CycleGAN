import os
import numpy as np
import random
import matplotlib.pyplot as plt
from visdom import Visdom


def load_npy_files(folder_path):
    """
    加载文件夹中的所有 .npy 文件并堆叠为 NumPy 数组
    """
    # 获取文件夹中所有 .npy 文件
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    # 检查是否有 .npy 文件
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {folder_path}")

    # 加载所有 .npy 文件并堆叠
    images = []
    for file in npy_files:
        file_path = os.path.join(folder_path, file)
        img_array = np.load(file_path)  # 加载 .npy 文件
        images.append(img_array)

    return np.stack(images)  # 堆叠为 NumPy 数组


def plot_slice(ct_slice, cbct_slice):
    """可视化 CT 和 CBCT 切片"""
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
    """随机可视化 CT 和 CBCT 的切片"""
    depth = ct_data.shape[0]
    slice_indices = random.sample(range(depth), num_samples)

    for idx in slice_indices:
        ct_slice = ct_data[idx]
        cbct_slice = cbct_data[idx]

        plot_slice(ct_slice, cbct_slice)


def normalize_to_range(data, target_min=0, target_max=1):
    """
    将数据归一化到指定范围 [target_min, target_max]
    """
    data_min, data_max = np.min(data), np.max(data)
    return (data - data_min) / (data_max - data_min + 1e-8) * (target_max - target_min) + target_min


def visualize_with_visdom(ct_data, cbct_data, num_samples=3):
    """
    使用 visdom 可视化 CT 和 CBCT 的随机切片
    """
    # 初始化 visdom
    viz = Visdom()

    # 检查 visdom 服务器是否连接
    if not viz.check_connection():
        raise ConnectionError("Visdom server not running. Please start the server.")

    depth = min(ct_data.shape[0], cbct_data.shape[0])
    slice_indices = random.sample(range(depth), num_samples)

    for idx in slice_indices:
        ct_slice = ct_data[idx]
        cbct_slice = cbct_data[idx]

        # 归一化到 [0, 1] 范围
        ct_slice_normalized = normalize_to_range(ct_slice)
        cbct_slice_normalized = normalize_to_range(cbct_slice)

        # 显示 CT 切片
        viz.image(
            ct_slice_normalized,
            opts=dict(title=f'CT Slice {idx}', caption='CT Slice'),
        )

        # 显示 CBCT 切片
        viz.image(
            cbct_slice_normalized,
            opts=dict(title=f'CBCT Slice {idx}', caption='CBCT Slice'),
        )


def check():
    """主函数：加载数据并可视化"""
    data_dir = r'D:\Data\cbct_ct\pelvis_internals2\0014'
    ct_path = os.path.join(data_dir, 'ct')
    cbct_path = os.path.join(data_dir, 'cbct')

    # 加载 CT 和 CBCT 数据
    ct_data = load_npy_files(ct_path)
    cbct_data = load_npy_files(cbct_path)

    print(f"CT shape: {ct_data.shape}")
    print(f"CBCT shape: {cbct_data.shape}")

    # 可视化随机切片
    # visualize_random_slices(ct_data, cbct_data, num_samples=5)
    visualize_with_visdom(ct_data, cbct_data, num_samples=5)


if __name__ == '__main__':
    check()
