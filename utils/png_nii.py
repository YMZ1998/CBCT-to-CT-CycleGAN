import os
import numpy as np
import SimpleITK as sitk
from PIL import Image


def load_png_sequence(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    images = []

    for file in image_files:
        img = Image.open(os.path.join(folder_path, file))
        img_array = np.array(img)
        images.append(img_array)

    volume = np.stack(images, axis=0)
    return volume


def save_as_nifti(volume, output_path, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    sitk_image = sitk.GetImageFromArray(volume)

    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)

    sitk.WriteImage(sitk_image, output_path)


def png_to_nii(folder_path, output_path, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    volume = load_png_sequence(folder_path)

    save_as_nifti(volume, output_path, spacing, origin)


if __name__ == '__main__':
    folder_path = r'./test_data/brain_a2b'
    output_path = r'./test_data/brain_predict.nii.gz'

    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)

    png_to_nii(folder_path, output_path, spacing, origin)
