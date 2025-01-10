import os
import shutil
import random


def copy_files(src_dir, dst_dir, prefix, file_list):
    """
    将文件从源目录复制到目标目录，并重命名。

    Args:
        src_dir (str): 源目录路径。
        dst_dir (str): 目标目录路径。
        prefix (str): 文件名前缀。
        file_list (list): 要复制的文件名列表。
    """
    os.makedirs(dst_dir, exist_ok=True)
    for i, file_name in enumerate(file_list):
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, f"{prefix}_{i}.png")
        shutil.copy(src_path, dst_path)


def prepare_cyclegan_dataset(data_dir, output_dir, train_ratio=0.8):
    """
    将 CT 和 CBCT 数据组织成 CycleGAN 数据集格式。

    Args:
        data_dir (str): 数据根目录（包含患者子目录）。
        output_dir (str): 输出数据集目录。
        train_ratio (float): 训练集比例。
    """
    # 获取所有患者目录
    patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # 初始化文件列表
    ct_files = []
    cbct_files = []

    # 遍历每个患者目录
    for patient_dir in patient_dirs[:10]:
        print(f"Processing patient directory: {patient_dir}")
        ct_dir = os.path.join(patient_dir, 'ct')
        cbct_dir = os.path.join(patient_dir, 'cbct')

        # 获取 CT 和 CBCT 文件
        if os.path.exists(ct_dir) and os.path.exists(cbct_dir):
            ct_files.extend([os.path.join(patient_dir, 'ct', f) for f in os.listdir(ct_dir) if f.endswith('.png')])
            cbct_files.extend(
                [os.path.join(patient_dir, 'cbct', f) for f in os.listdir(cbct_dir) if f.endswith('.png')])

    # 确保 CT 和 CBCT 文件数量一致
    assert len(ct_files) == len(cbct_files), "CT and CBCT files must have the same length."

    # 随机划分训练集和测试集
    num_samples = len(ct_files)
    num_train = int(num_samples * train_ratio)
    indices = list(range(num_samples))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # 获取训练集和测试集文件路径
    ct_train_files = [ct_files[i] for i in train_indices]
    cbct_train_files = [cbct_files[i] for i in train_indices]
    ct_test_files = [ct_files[i] for i in test_indices]
    cbct_test_files = [cbct_files[i] for i in test_indices]

    # 复制训练集
    copy_files('', os.path.join(output_dir, 'train', 'A'), 'cbct', cbct_train_files)
    copy_files('', os.path.join(output_dir, 'train', 'B'), 'ct', ct_train_files)

    # 复制测试集
    copy_files('', os.path.join(output_dir, 'test', 'A'), 'cbct', cbct_test_files)
    copy_files('', os.path.join(output_dir, 'test', 'B'), 'ct', ct_test_files)


def main():
    # 数据路径
    data_dir = r'D:\Data\cbct_ct'  # 数据根目录
    output_dir = r'./datasets/cbct2ct'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 准备 CycleGAN 数据集
    prepare_cyclegan_dataset(data_dir, output_dir)

    print(f"CycleGAN dataset saved to {output_dir}")


if __name__ == '__main__':
    main()
