import os
import shutil
import sys

import numpy as np
import tqdm
import SimpleITK as sitk
from PIL import Image


def pad_image(image, target_size=(256, 256)):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    pad_height = max(target_size[0] - original_size[1], 0)
    pad_width = max(target_size[1] - original_size[0], 0)

    new_origin = [
        image.GetOrigin()[0] - pad_width * original_spacing[0] / 2,
        image.GetOrigin()[1] - pad_height * original_spacing[1] / 2,
        image.GetOrigin()[2] if len(original_size) > 2 else 0
    ]

    # print(np.min(sitk.GetArrayFromImage(image)))
    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetPadLowerBound([0, 0, 0])
    pad_filter.SetPadUpperBound([pad_width, pad_height, 0])
    if np.min(sitk.GetArrayFromImage(image)) < 0:
        pad_filter.SetConstant(-1024)
    else:
        pad_filter.SetConstant(0)  # 填充值为 0

    padded_image = pad_filter.Execute(image)
    padded_image.SetOrigin(new_origin)

    return padded_image


def resample_image(image, target_size=(256, 256)):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1]),
        original_spacing[2] if len(original_size) > 2 else 1  # 保持深度方向不变
    ]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize([target_size[0], target_size[1], original_size[2] if len(original_size) > 2 else 1])
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(sitk.sitkLinear)  # 线性插值

    resampled_image = resample_filter.Execute(image)

    return resampled_image


def save_png_images(file_path, ct, cbct, mask):
    ct_dir = os.path.join(file_path, 'ct')
    cbct_dir = os.path.join(file_path, 'cbct')
    mask_dir = os.path.join(file_path, 'mask')

    os.makedirs(ct_dir, exist_ok=True)
    os.makedirs(cbct_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(len(cbct)):
        ct_img = normalize_to_uint8(ct[i])
        cbct_img = normalize_to_uint8(cbct[i])
        mask_img = normalize_to_uint8(mask[i])

        Image.fromarray(ct_img).save(os.path.join(ct_dir, f"{i}.png"))
        Image.fromarray(cbct_img).save(os.path.join(cbct_dir, f"{i}.png"))
        Image.fromarray(mask_img).save(os.path.join(mask_dir, f"{i}.png"))


def normalize_to_uint8(data):
    # print(data.max(), data.min())
    data_normalized = (data - data.min()) / (2000 - data.min() + 1e-8) * 255
    data_normalized = np.clip(data_normalized, 0, 255)
    return data_normalized.astype(np.uint8)


def load_images(file_path):
    try:
        cbct = sitk.GetArrayFromImage(sitk.ReadImage(file_path + '/cbct.nii.gz'))
        ct = sitk.GetArrayFromImage(sitk.ReadImage(file_path + '/ct.nii.gz'))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(file_path + '/mask.nii.gz'))
        return cbct, ct, mask
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None


def transfer_folder(path_in, path_out, target_size=(256, 256)):
    if os.path.exists(path_out):
        shutil.rmtree(path_out)
    os.makedirs(path_out, exist_ok=True)

    for file in tqdm.tqdm(os.listdir(path_in)[:], file=sys.stdout):
        file_path_in = os.path.join(path_in, file)
        file_path_out = os.path.join(path_out, file)

        cbct, ct, mask = load_images(file_path_in)
        if cbct is None or ct is None or mask is None:
            continue

        os.makedirs(file_path_out, exist_ok=True)

        ct_image = sitk.GetImageFromArray(ct)
        cbct_image = sitk.GetImageFromArray(cbct)
        mask_image = sitk.GetImageFromArray(mask)

        ct_padded = pad_image(ct_image, target_size=target_size)
        cbct_padded = pad_image(cbct_image, target_size=target_size)
        mask_padded = pad_image(mask_image, target_size=target_size)

        ct_resampled = resample_image(ct_padded, target_size=target_size)
        cbct_resampled = resample_image(cbct_padded, target_size=target_size)
        mask_resampled = resample_image(mask_padded, target_size=target_size)

        ct_resampled_np = sitk.GetArrayFromImage(ct_resampled)
        cbct_resampled_np = sitk.GetArrayFromImage(cbct_resampled)
        mask_resampled_np = sitk.GetArrayFromImage(mask_resampled)

        # 保存为 PNG 格式
        save_png_images(file_path_out, ct_resampled_np, cbct_resampled_np, mask_resampled_np)


def transfer_one_case(path, result, target_size=(256, 256)):
    if os.path.exists(result):
        shutil.rmtree(result)
    os.makedirs(result, exist_ok=True)

    image = sitk.ReadImage(path)
    ct_padded = pad_image(image, target_size=target_size)
    ct_resampled = resample_image(ct_padded, target_size=target_size)
    ct_resampled_np = sitk.GetArrayFromImage(ct_resampled)

    for i in tqdm.tqdm(range(len(ct_resampled_np))):
        ct_img = normalize_to_uint8(ct_resampled_np[i])
        Image.fromarray(ct_img).save(os.path.join(result, f"{i + 1:04d}.png"))


if __name__ == '__main__':
    anatomy = 'pelvis'
    path_in = os.path.join(r'D:\Data\SynthRAD\Task2', anatomy)
    path_out = os.path.join(r'D:\Data\cbct_ct', anatomy)
    # transfer_folder(path_in, path_out)

    path = r'../test_data/pelvis.nii.gz'
    result = r'../test_data/pelvis'
    transfer_one_case(path, result)
