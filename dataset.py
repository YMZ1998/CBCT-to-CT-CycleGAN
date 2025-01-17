import glob
import os
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


def normalize(data, anatomy='pelvis'):
    if anatomy == 'pelvis':
        data_min, data_max = -1000, 2000

    elif anatomy == 'brain':
        data_min, data_max = -1000, 2000

    data = np.clip(data, data_min, data_max)
    # data_min, data_max = np.min(data), np.max(data)
    # print(data_min, data_max)

    return (data - data_min) / (data_max - data_min + 1e-8)


class NpyDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', anatomy='pelvis'):
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        self.unaligned = unaligned
        self.anatomy = anatomy

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode, '*.npy')))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode, '*.npy')))

    def __getitem__(self, index):
        item_A = np.load(self.files_A[index % len(self.files_A)]).astype(np.float32)

        item_A = normalize(item_A, self.anatomy)
        item_A = Image.fromarray(item_A)
        if self.transform:
            item_A = self.transform(item_A)

        if self.unaligned:
            # print(self.files_B[random.randint(0, len(self.files_B) - 1)])
            item_B = np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]).astype(np.float32)
        else:
            item_B = np.load(self.files_B[index % len(self.files_B)]).astype(np.float32)

        item_B = normalize(item_B, self.anatomy)
        item_B = Image.fromarray(item_B)

        if self.transform:
            item_B = self.transform(item_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))[:]
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))[:]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def test_npy_dataset():
    root = './datasets/pelvis'
    # root = './datasets/brain'

    transforms_ = [
        transforms.Resize(int(512), Image.BILINEAR),
        # transforms.RandomCrop(256),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]

    dataset = NpyDataset(root=root, transforms_=transforms_, unaligned=True, mode='train', anatomy='pelvis')
    # dataset = ImageDataset(root=root, transforms_=transforms_, unaligned=True, mode='train')

    print(f"数据集大小: {len(dataset)}")

    sample = dataset[0]
    print(f"样本 A 的形状: {sample['A'].shape}")
    print(f"样本 B 的形状: {sample['B'].shape}")
    print(f"样本 A 的最大值: {sample['A'].max()}")
    print(f"样本 A 的最小值: {sample['A'].min()}")
    print(f"样本 B 的最大值: {sample['B'].max()}")
    print(f"样本 B 的最小值: {sample['B'].min()}")

    def plot_sample(sample):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(sample['A'].squeeze(), cmap='gray')  # 假设是单通道图像
        plt.title('A')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(sample['B'].squeeze(), cmap='gray')  # 假设是单通道图像
        plt.title('B')
        plt.axis('off')

        plt.show()

    for i in range(5):
        sample = dataset[i]
        plot_sample(sample)

    # 使用 DataLoader 加载数据
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 遍历 DataLoader
    for i, batch in enumerate(dataloader):
        print(f"批次 {i + 1}:")
        print(f"A 的形状: {batch['A'].shape}")
        print(f"B 的形状: {batch['B'].shape}")
        print(f"A 的最大值: {batch['A'].max()}")
        print(f"A 的最小值: {batch['A'].min()}")
        if i == 3:  # 只检查第一个批次
            break


# 运行测试
if __name__ == '__main__':
    test_npy_dataset()
