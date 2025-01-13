import os

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


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


def process_folder(input_dir, output_dir, target_spacing, mode='cbct'):
    for root, dirs, files in os.walk(input_dir):
        for dir_name in tqdm(dirs):
            dicom_dir = os.path.join(root, dir_name)
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

                # image.SetOrigin((0, 0, 0))

                os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
                output_path = os.path.join(output_dir, dir_name, mode + ".nii.gz")
                sitk.WriteImage(image, output_path)
                # print(f"Saved resampled image to: {output_path}")

            except Exception as e:
                print(f"Error processing {dicom_dir}: {e}")


def check_dcm():
    path_in = r"D:\Data\CBCT2CT\TrainingSet\Trainingset_CBCT"
    for p in os.listdir(path_in):
        dicom_dir = os.path.join(path_in, p)
        image = load_dicom_series(dicom_dir)
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        print(p, original_size, original_spacing)


def check_nii():
    # path_in = r"D:\Data\CBCT2CT\Data"
    path_in = r"D:\Data\SynthRAD\Task2\brain"
    for p in os.listdir(path_in):
        path = os.path.join(path_in, p, 'cbct.nii.gz')
        image = sitk.ReadImage(path)
        print(p, image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())


if __name__ == '__main__':
    # check_dcm()
    # check_nii()
    input_dir = r"D:\Data\CBCT2CT\TrainingSet\Trainingset_CT"
    input_dir2 = r"D:\Data\CBCT2CT\TrainingSet\Trainingset_CBCT"
    output_dir = r"D:\Data\CBCT2CT\Data"
    target_spacing = [1.0, 1.0, 1.0]

    from utils import remove_and_create_dir

    remove_and_create_dir(output_dir)

    process_folder(input_dir, output_dir, target_spacing, mode='ct')
    process_folder(input_dir2, output_dir, target_spacing, mode='cbct')
