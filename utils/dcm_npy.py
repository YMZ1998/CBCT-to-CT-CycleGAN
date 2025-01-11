import os
import shutil
import sys

import numpy as np
import pydicom
from tqdm import tqdm


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def normalize_to_range(data, target_min=-1000, target_max=2000):
    """将数据归一化到指定范围"""
    # data_min, data_max = np.min(data), np.max(data)
    # print(data_min, data_max)
    data = np.clip(data, target_min, target_max).astype(np.int16)
    # data = (data - data_min) / (data_max - data_min + 1e-8)
    return data


def load_dicom(dicom_path):
    # 读取 DICOM 文件
    dicom = pydicom.dcmread(dicom_path)

    # 解压数据（如果压缩）
    if hasattr(dicom, 'is_decompressed') and not dicom.is_decompressed:
        dicom.decompress()

    # 获取像素数组
    pixel_array = dicom.pixel_array

    # 应用 Rescale Slope 和 Rescale Intercept
    rescale_slope = dicom.RescaleSlope if 'RescaleSlope' in dicom else 1
    rescale_intercept = dicom.RescaleIntercept if 'RescaleIntercept' in dicom else 0
    pixel_array = pixel_array * rescale_slope + rescale_intercept

    # 处理无效值（假设无效值为 -32768）
    invalid_value = -32768
    pixel_array[pixel_array == invalid_value] = np.nan

    return pixel_array


def convert_dicom_to_npy(dicom_path, output_path):
    """将 DICOM 文件转换为 .npy 文件，并归一化至 0-2000"""
    pixel_array = load_dicom(dicom_path)
    pixel_array = normalize_to_range(pixel_array)
    # print(pixel_array.max())
    # print(pixel_array.dtype)
    np.save(output_path, pixel_array)


def convert_folder_dicom_to_npy(input_folder, output_folder):
    """将文件夹中的 DICOM 文件批量转换为 .npy 文件"""
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".dcm") and 'CT' in filename:
            dicom_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".dcm", ".npy"))
            convert_dicom_to_npy(dicom_path, output_path)


def transfer_folders(path_in, path_out, mode):
    """批量处理文件夹"""
    print(f'Transferring folder: {path_in}')
    for file in tqdm(os.listdir(path_in), file=sys.stdout):
        file_path_in = os.path.join(path_in, file)
        file_path_out = os.path.join(path_out, file, mode)
        os.makedirs(file_path_out, exist_ok=True)
        convert_folder_dicom_to_npy(file_path_in, file_path_out)


if __name__ == '__main__':
    path_in = r"D:\Data\CBCT2CT\TrainingSet"
    path_out = r'D:\Data\cbct_ct\pelvis_internals2'
    remove_and_create_dir(path_out)
    transfer_folders(os.path.join(path_in, 'Trainingset_CBCT'), path_out, 'cbct')
    transfer_folders(os.path.join(path_in, 'Trainingset_CT'), path_out, 'ct')
