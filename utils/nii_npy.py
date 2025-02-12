import os
import shutil
import sys

import SimpleITK as sitk
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from skimage.transform import resize

target_size=(512, 512)

def pad_image(image):
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
    pad_filter.SetConstant(int(np.min(sitk.GetArrayFromImage(image))))
    # print(np.min(sitk.GetArrayFromImage(image)))

    padded_image = pad_filter.Execute(image)
    padded_image.SetOrigin(new_origin)

    return padded_image


def resample_image(image):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1]),
        original_spacing[2]
    ]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize([target_size[0], target_size[1], original_size[2] if len(original_size) > 2 else 1])
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(sitk.sitkLinear)

    resampled_image = resample_filter.Execute(image)

    return resampled_image

def resize_image(image):
    resized_image = resize(image, target_size, mode='constant', preserve_range=True)
    return resized_image.astype(np.int16)

def save_images(file_path, ct, cbct):
    ct_dir = os.path.join(file_path, 'ct')
    cbct_dir = os.path.join(file_path, 'cbct')

    os.makedirs(ct_dir, exist_ok=True)
    os.makedirs(cbct_dir, exist_ok=True)

    ct = ct[15:-15]
    cbct = cbct[10:-10]

    for i in range(len(ct)):
        if np.unique(ct[i]).size == 1:
            continue
        ct_img = normalize_to_uint8(ct[i])
        ct_img = resize_image(ct_img)
        np.save(os.path.join(ct_dir, f"{i}.npy"), ct_img)

    for i in range(len(cbct)):
        if np.unique(cbct[i]).size == 1:
            continue
        cbct_img = normalize_to_uint8(cbct[i])
        cbct_img = resize_image(cbct_img)
        np.save(os.path.join(cbct_dir, f"{i}.npy"), cbct_img)


def normalize_to_uint8(data):
    # print(data.max(), data.min())
    data = np.clip(data, -1000, 2000).astype(np.int16)
    return data


def load_images(file_path):
    try:
        cbct_path = os.path.join(file_path, 'cbct.nii.gz')
        ct_path = os.path.join(file_path, 'ct.nii.gz')

        if not os.path.exists(cbct_path):
            raise FileNotFoundError(f"文件 {cbct_path} 不存在。")
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"文件 {ct_path} 不存在。")

        cbct = sitk.GetArrayFromImage(sitk.ReadImage(cbct_path))
        ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
        if int(np.min(cbct)) == 0:
            cbct = cbct - 1024
        # mask = sitk.GetArrayFromImage(sitk.ReadImage(file_path + '/mask.nii.gz'))
        return cbct, ct, None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None


def transfer_folder(path_in, path_out):
    if os.path.exists(path_out):
        shutil.rmtree(path_out)
    os.makedirs(path_out, exist_ok=True)

    for file in tqdm.tqdm(os.listdir(path_in)[:], file=sys.stdout):
        file_path_in = os.path.join(path_in, file)
        file_path_out = os.path.join(path_out, file)

        cbct, ct, mask = load_images(file_path_in)

        os.makedirs(file_path_out, exist_ok=True)

        ct_image = sitk.GetImageFromArray(ct)
        cbct_image = sitk.GetImageFromArray(cbct)
        # mask_image = sitk.GetImageFromArray(mask)

        # ct_image = pad_image(ct_image, target_size=target_size)
        # cbct_image = pad_image(cbct_image, target_size=target_size)
        # mask_padded = pad_image(mask_image, target_size=target_size)

        ct_image = resample_image(ct_image)
        cbct_image = resample_image(cbct_image)
        # print(ct_resampled.GetSize(), cbct_resampled.GetSize())
        # mask_resampled = resample_image(mask_padded, target_size=target_size)

        ct_resampled_np = sitk.GetArrayFromImage(ct_image)
        cbct_resampled_np = sitk.GetArrayFromImage(cbct_image)
        # mask_resampled_np = sitk.GetArrayFromImage(mask_resampled)

        save_images(file_path_out, ct_resampled_np, cbct_resampled_np)


def transfer_one_case(path, result):
    if os.path.exists(result):
        shutil.rmtree(result)
    os.makedirs(result, exist_ok=True)

    image = sitk.ReadImage(path)
    ct_padded = pad_image(image)
    ct_resampled = resample_image(ct_padded)
    ct_resampled_np = sitk.GetArrayFromImage(ct_resampled)

    for i in tqdm.tqdm(range(len(ct_resampled_np))):
        ct_img = normalize_to_uint8(ct_resampled_np[i])
        np.save(os.path.join(os.path.join(result, f"{i + 1:04d}.npy")), ct_img)

    data = np.load(os.path.join(result, '0001.npy'))
    print(data.shape)
    data_min, data_max = np.min(data), np.max(data)
    # data = ((data - data_min) / (data_max - data_min + 1e-8) * 255).astype(np.uint8)
    plt.figure()
    plt.imshow(data, cmap='gray')
    plt.show()


if __name__ == '__main__':
    # anatomy = 'pelvis'
    # path_in = os.path.join(r'D:\Data\SynthRAD\Task2', anatomy)
    # path_out = os.path.join(r'D:\Data\cbct_ct', anatomy)

    path_in = r'D:\Data\CBCT2CT2\Data'
    path_out = os.path.join(r'D:\Data\cbct_ct', 'thorax')
    transfer_folder(path_in, path_out)

    # path = r'../test_data/pelvis.nii.gz'
    # result = r'../test_data/pelvis'
    # transfer_one_case(path, result)
