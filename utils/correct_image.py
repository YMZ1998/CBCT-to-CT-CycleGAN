import SimpleITK as sitk
import numpy as np


def bias_field_correction(pseudo_ct, num_iterations=[50,50], convergence_threshold=0.001, shrink_factor=2):
    """
    使用 N4 偏置场校正对伪 CT 数据进行强度校正。

    参数:
        pseudo_ct (numpy.ndarray): 伪 CT 数据（2D 或 3D）。
        num_iterations (int): 每个分辨率级别的迭代次数，默认为 50。
        convergence_threshold (float): 收敛阈值，默认为 0.001。
        shrink_factor (int): 图像缩小因子，默认为 2。

    返回:
        numpy.ndarray: 校正后的伪 CT 数据。
    """
    # 将 numpy 数组转换为 SimpleITK 图像
    image = sitk.GetImageFromArray(pseudo_ct)

    # 将图像像素类型转换为 float32
    if image.GetPixelID() != sitk.sitkFloat32:
        image = sitk.Cast(image, sitk.sitkFloat32)

    # 创建 N4 偏置场校正滤波器
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # 设置 N4 校正参数
    corrector.SetMaximumNumberOfIterations(num_iterations)
    corrector.SetConvergenceThreshold(convergence_threshold)
    # corrector.SetShrinkFactor(shrink_factor)

    # 执行 N4 偏置场校正
    corrected_image = corrector.Execute(image)

    # 将 SimpleITK 图像转换回 numpy 数组
    corrected_ct = sitk.GetArrayFromImage(corrected_image)
    sitk.WriteImage(corrected_image, r'../installer/result/corrected_ct.nii.gz')
    return corrected_ct


# 示例调用
if __name__ == "__main__":
    pseudo_ct = r'../installer/result/predict.nii.gz'
    pseudo_ct = sitk.ReadImage(pseudo_ct)
    pseudo_ct = sitk.GetArrayFromImage(pseudo_ct)

    # 执行 N4 偏置场校正
    corrected_ct = bias_field_correction(pseudo_ct)

    # 打印校正前后的数据范围
    print("校正前强度范围:", np.min(pseudo_ct), np.max(pseudo_ct))
    print("校正后强度范围:", np.min(corrected_ct), np.max(corrected_ct))