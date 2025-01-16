import os
import pydicom
from tqdm import tqdm

from utils import remove_and_create_dir


def split_dicom_files(input_folder, output_folder):
    """
    将输入文件夹中的 DICOM 文件按患者 ID 分类保存到输出文件夹中。

    参数:
    input_folder (str): 包含 DICOM 文件的输入文件夹路径。
    output_folder (str): 用于保存分类后的 DICOM 文件的输出文件夹路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, filename)

        # 检查文件是否为 DICOM 文件
        if not filename.lower().endswith(".dcm"):
            print(f"Skipping non-DICOM file: {filename}")
            continue

        try:
            # 读取 DICOM 文件
            dicom_data = pydicom.dcmread(file_path)

            # 提取患者 ID
            patient_id = dicom_data.get("PatientID", "UnknownPatient")

            # 创建患者文件夹
            if "FM" in file_path:
                patient_folder = os.path.join(output_folder, patient_id + "-FM")
            else:
                patient_folder = os.path.join(output_folder, patient_id + "-M")
            if not os.path.exists(patient_folder):
                os.makedirs(patient_folder)

            # 保存 DICOM 文件到对应的文件夹
            output_path = os.path.join(patient_folder, filename)
            dicom_data.save_as(output_path)

            # print(f"Saved {filename} to {output_path}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")


# 示例用法
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
