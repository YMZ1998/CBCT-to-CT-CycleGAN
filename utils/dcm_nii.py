import os

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from utils import remove_and_create_dir


def load_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    # 检查每一帧的像素值
    # slice = image[:, :, -1]
    # slice_array = sitk.GetArrayFromImage(slice)
    # print(f"Slice: Min = {np.min(slice_array)}, Max = {np.max(slice_array)}")
    return image


def resample_image(image, target_spacing, new_size):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    resampled_image = resampler.Execute(image)
    return resampled_image


def pad_or_crop_image(image, target_size):
    original_size = image.GetSize()

    # 计算每个维度的填充或裁剪量
    lower_bound = [0, 0, 0]  # 裁剪的下边界
    upper_bound = [0, 0, 0]  # 裁剪的上边界
    pad_lower = [0, 0, 0]  # 填充的下边界
    pad_upper = [0, 0, 0]  # 填充的上边界

    for i in range(3):
        size_diff = target_size[i] - original_size[i]

        if size_diff > 0:
            # 需要填充
            pad_lower[i] = size_diff // 2
            pad_upper[i] = size_diff - pad_lower[i]
        else:
            # 需要裁剪
            lower_bound[i] = (original_size[i] - target_size[i]) // 2
            upper_bound[i] = original_size[i] - target_size[i] - lower_bound[i]

    # Z 轴保持不变
    pad_lower[2] = 0
    pad_upper[2] = 0
    lower_bound[2] = 0
    upper_bound[2] = 0

    # 填充或裁剪
    if any(pad_lower) or any(pad_upper):
        # 填充
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound(pad_lower)
        pad_filter.SetPadUpperBound(pad_upper)
        pad_filter.SetConstant(-1000)  # 填充值（CT 的 air 值）
        image = pad_filter.Execute(image)
    elif any(lower_bound) or any(upper_bound):
        # 裁剪
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize(lower_bound)
        crop_filter.SetUpperBoundaryCropSize(upper_bound)
        image = crop_filter.Execute(image)

    return image


def process_folder(input_dir, output_dir, target_spacing, mode='cbct'):
    patient_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))][:]

    for dir_name in tqdm(patient_folders):
        dicom_dir = os.path.join(input_dir, dir_name)
        # print(f"Processing: {dicom_dir}")
        try:
            image = load_dicom_series(dicom_dir)

            original_size = image.GetSize()
            original_spacing = image.GetSpacing()
            new_size = [
                int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / target_spacing[2])))
            ]

            image = resample_image(image, target_spacing, new_size)
            image = pad_or_crop_image(image, [512, 512, 0])

            # image.SetOrigin((0, 0, 0))

            os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
            output_path = os.path.join(output_dir, dir_name, mode + ".nii.gz")
            sitk.WriteImage(image, output_path)
            # print(f"Saved resampled image to: {output_path}")

        except Exception as e:
            print(f"Error processing {dicom_dir}: {e}")


def check_dcm():
    # path_in = r"D:\Data\CBCT2CT\TrainingSet\Trainingset_CBCT"
    path_in = r"D:\Data\CBCT2CT2\CBCT"
    for p in os.listdir(path_in):
        dicom_dir = os.path.join(path_in, p)
        image = load_dicom_series(dicom_dir)
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        print(p, original_size, original_spacing)


def check_nii():
    # path_in = r"D:\Data\CBCT2CT\Data"
    # path_in = r"D:\Data\SynthRAD\Task2\brain"
    path_in = r"D:\Data\CBCT2CT2\Data"
    for p in os.listdir(path_in):
        print("-" * 30)
        path = os.path.join(path_in, p, 'cbct.nii.gz')
        image = sitk.ReadImage(path)
        print(p, image.GetSize(), image.GetSpacing(), image.GetOrigin())
        path2 = os.path.join(path_in, p, 'ct.nii.gz')
        image2 = sitk.ReadImage(path2)
        print(p, image2.GetSize(), image2.GetSpacing(), image2.GetOrigin())
        print("-" * 30)


def transfer_folder():
    input_dir = r"D:\Data\CBCT2CT2\CT"
    input_dir2 = r"D:\Data\CBCT2CT2\CBCT"
    output_dir = r"D:\Data\CBCT2CT2\Data"

    target_spacing = [0.8, 0.8, 2.]
    target_spacing2 = [0.8, 0.8, 1.1]

    remove_and_create_dir(output_dir)

    process_folder(input_dir, output_dir, target_spacing, mode='ct')
    process_folder(input_dir2, output_dir, target_spacing2, mode='cbct')


if __name__ == '__main__':
    # check_dcm()
    check_nii()
    # transfer_folder()
