import os
import random
import matplotlib.pyplot as plt
from PIL import Image


def load_random_images(folder, num_samples=5):
    """
    从文件夹中随机加载指定数量的图像。

    Args:
        folder (str): 图像文件夹路径。
        num_samples (int): 需要加载的图像数量。

    Returns:
        list: 包含图像数据的列表。
    """
    image_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    images = [Image.open(os.path.join(folder, f)) for f in selected_files]
    return images


def plot_images(ct_images, cbct_images):
    """
    可视化 CT 和 CBCT 图像。

    Args:
        ct_images (list): CT 图像列表。
        cbct_images (list): CBCT 图像列表。
    """
    num_samples = min(len(ct_images), len(cbct_images))
    plt.figure(figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # 显示 CT 图像
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(ct_images[i], cmap='gray')
        plt.title(f'CT Image {i + 1}')
        plt.axis('off')

        # 显示 CBCT 图像
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(cbct_images[i], cmap='gray')
        plt.title(f'CBCT Image {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # 数据集路径
    dataset_dir = r'./datasets/cbct2ct'
    trainA_dir = os.path.join(dataset_dir, 'train', 'A')
    trainB_dir = os.path.join(dataset_dir, 'train', 'B')

    # 随机加载图像
    num_samples = 3  # 抽样数量
    cbct_images = load_random_images(trainA_dir, num_samples)
    ct_images = load_random_images(trainB_dir, num_samples)

    # 可视化图像
    plot_images(ct_images, cbct_images)


if __name__ == '__main__':
    main()
