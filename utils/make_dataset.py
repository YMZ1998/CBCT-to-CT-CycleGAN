import os
import shutil
import random
import sys

from tqdm import tqdm
from utils import remove_and_create_dir

suffix='npy'


def copy_files(src_dir, dst_dir, prefix, file_list):
    os.makedirs(dst_dir, exist_ok=True)
    for i, file_name in tqdm(enumerate(file_list), file=sys.stdout, total=len(file_list)):
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, f"{prefix}_{i}.{suffix}")
        shutil.copy(src_path, dst_path)


def split_dataset(file_list, train_ratio):
    num_samples = len(file_list)
    num_train = int(num_samples * train_ratio)
    indices = list(range(num_samples))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    return [file_list[i] for i in train_indices], [file_list[i] for i in test_indices]


def prepare_cyclegan_dataset(data_dir, output_dir, train_ratio=0.8):
    patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    ct_files = []
    cbct_files = []

    for patient_dir in tqdm(patient_dirs, file=sys.stdout):
        ct_dir = os.path.join(patient_dir, 'ct')
        cbct_dir = os.path.join(patient_dir, 'cbct')

        if os.path.exists(ct_dir) and os.path.exists(cbct_dir):
            ct_files.extend([os.path.join(patient_dir, 'ct', f) for f in os.listdir(ct_dir) if f.endswith(f'.{suffix}')])
            cbct_files.extend(
                [os.path.join(patient_dir, 'cbct', f) for f in os.listdir(cbct_dir) if f.endswith(f'.{suffix}')])

    print(len(ct_files), len(cbct_files))
    # 分别划分 CT 和 CBCT 数据集
    ct_train_files, ct_test_files = split_dataset(ct_files, train_ratio)
    cbct_train_files, cbct_test_files = split_dataset(cbct_files, train_ratio)

    copy_files('', os.path.join(output_dir, 'train', 'A'), 'cbct', cbct_files)
    copy_files('', os.path.join(output_dir, 'train', 'B'), 'ct', ct_files)

    copy_files('', os.path.join(output_dir, 'test', 'A'), 'cbct', cbct_test_files)
    copy_files('', os.path.join(output_dir, 'test', 'B'), 'ct', ct_test_files)


def main():
    data_dir = r'D:\Data\cbct_ct\thorax'
    output_dir = r'../datasets/thorax-512'
    # data_dir = r'D:\Data\cbct_ct\brain'
    # output_dir = r'../datasets/brain'

    remove_and_create_dir(output_dir)
    prepare_cyclegan_dataset(data_dir, output_dir)

    print(f"CycleGAN dataset saved to {output_dir}")


if __name__ == '__main__':
    main()
