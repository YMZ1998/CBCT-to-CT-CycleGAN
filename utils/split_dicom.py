import os
import pydicom
from tqdm import tqdm

from utils import remove_and_create_dir


def split_dicom_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, filename)

        if not filename.lower().endswith(".dcm"):
            print(f"Skipping non-DICOM file: {filename}")
            continue

        try:
            dicom_data = pydicom.dcmread(file_path)

            patient_id = dicom_data.get("PatientID", "UnknownPatient")

            if "FM" in file_path:
                patient_folder = os.path.join(output_folder, patient_id + "-FM")
            else:
                patient_folder = os.path.join(output_folder, patient_id + "-M")
            if not os.path.exists(patient_folder):
                os.makedirs(patient_folder)

            output_path = os.path.join(patient_folder, filename)
            dicom_data.save_as(output_path)

            # print(f"Saved {filename} to {output_path}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    # input_folder = r"D:\Data\CBCT2CT2\CT-FM"
    # input_folder2 = r"D:\Data\CBCT2CT2\CT-M"
    # output_folder = r"D:\Data\CBCT2CT2\CT"
    input_folder = r"D:\Data\CBCT2CT2\FM"
    input_folder2 = r"D:\Data\CBCT2CT2\M"
    output_folder = r"D:\Data\CBCT2CT2\CBCT"
    remove_and_create_dir(output_folder)

    split_dicom_files(input_folder, output_folder)
    split_dicom_files(input_folder2, output_folder)
