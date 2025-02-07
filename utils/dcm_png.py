import os
import shutil
import sys

import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_dicom(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    pixel_array = dicom.pixel_array
    return pixel_array, dicom


def normalize_to_uint8(data):
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8) * 255
    return data_normalized.astype(np.uint8)


def save_as_png(pixel_array, output_path, target_size=(256, 256)):
    image = Image.fromarray(pixel_array)

    if target_size is not None:
        image = image.resize(target_size, Image.BICUBIC)

    image.save(output_path)


def convert_dicom_to_png(dicom_path, output_path):
    pixel_array, dicom = load_dicom(dicom_path)

    pixel_array_normalized = normalize_to_uint8(pixel_array)

    save_as_png(pixel_array_normalized, output_path)


def convert_folder_dicom_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".dcm") and 'CT' in filename:
            dicom_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".dcm", ".png"))
            convert_dicom_to_png(dicom_path, output_path)


def transfer_folder(path_in, path_out, mode='cbct'):
    print('Transferring folder:', path_in)
    for file in tqdm(os.listdir(path_in)[:], file=sys.stdout):
        file_path_in = os.path.join(path_in, file)
        file_path_out = os.path.join(path_out, file, mode)

        os.makedirs(file_path_out, exist_ok=True)
        convert_folder_dicom_to_png(file_path_in, file_path_out)


if __name__ == '__main__':
    path_in = r"D:\Data\CBCT2CT\TrainingSet"
    path_out = r'D:\Data\cbct_ct\pelvis_internals'
    remove_and_create_dir(path_out)
    transfer_folder(os.path.join(path_in, 'Trainingset_CBCT'), path_out, 'cbct')
    transfer_folder(os.path.join(path_in, 'Trainingset_CT'), path_out, 'ct')
