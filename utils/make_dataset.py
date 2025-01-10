import os
import shutil
import random
import sys

from tqdm import tqdm


def copy_files(src_dir, dst_dir, prefix, file_list):
    os.makedirs(dst_dir, exist_ok=True)
    for i, file_name in tqdm(enumerate(file_list), file=sys.stdout, total=len(file_list)):
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, f"{prefix}_{i}.png")
        shutil.copy(src_path, dst_path)


def prepare_cyclegan_dataset(data_dir, output_dir, train_ratio=0.8):
    patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    ct_files = []
    cbct_files = []

    for patient_dir in tqdm(patient_dirs, file=sys.stdout):
        ct_dir = os.path.join(patient_dir, 'ct')
        cbct_dir = os.path.join(patient_dir, 'cbct')

        if os.path.exists(ct_dir) and os.path.exists(cbct_dir):
            ct_files.extend([os.path.join(patient_dir, 'ct', f) for f in os.listdir(ct_dir) if f.endswith('.png')])
            cbct_files.extend(
                [os.path.join(patient_dir, 'cbct', f) for f in os.listdir(cbct_dir) if f.endswith('.png')])

    assert len(ct_files) == len(cbct_files), "CT and CBCT files must have the same length."

    num_samples = len(ct_files)
    num_train = int(num_samples * train_ratio)
    indices = list(range(num_samples))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    ct_train_files = [ct_files[i] for i in train_indices]
    cbct_train_files = [cbct_files[i] for i in train_indices]
    ct_test_files = [ct_files[i] for i in test_indices]
    cbct_test_files = [cbct_files[i] for i in test_indices]

    copy_files('', os.path.join(output_dir, 'train', 'A'), 'cbct', cbct_train_files)
    copy_files('', os.path.join(output_dir, 'train', 'B'), 'ct', ct_train_files)

    copy_files('', os.path.join(output_dir, 'test', 'A'), 'cbct', cbct_test_files)
    copy_files('', os.path.join(output_dir, 'test', 'B'), 'ct', ct_test_files)


def main():
    data_dir = r'D:\Data\cbct_ct\brain'
    output_dir = r'../datasets/cbct2ct'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    prepare_cyclegan_dataset(data_dir, output_dir)

    print(f"CycleGAN dataset saved to {output_dir}")


if __name__ == '__main__':
    main()
