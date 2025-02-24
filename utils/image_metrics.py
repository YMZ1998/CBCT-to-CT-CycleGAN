import os
from typing import Optional

import SimpleITK as sitk
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop


class ImageMetrics:
    def __init__(self, dynamic_range=[-1024, 1024]):
        # Use fixed wide dynamic range
        self.dynamic_range = dynamic_range

    def score_patient(self, gt_array, pred_array, mask_array):

        # Calculate image metrics
        mae_value = self.mae(gt_array,
                             pred_array,
                             mask_array)

        psnr_value = self.psnr(gt_array,
                               pred_array,
                               mask_array,
                               use_population_range=True)

        ssim_value = self.ssim(gt_array,
                               pred_array,
                               mask_array)
        return {
            'mae': mae_value,
            'ssim': ssim_value,
            'psnr': psnr_value
        }

    def mae(self,
            gt: np.ndarray,
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

        mae_value = np.sum(np.abs(gt * mask - pred * mask)) / mask.sum()
        return float(mae_value)

    def psnr(self,
             gt: np.ndarray,
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: Optional[bool] = False) -> float:
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

        if use_population_range:
            dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]

            # Clip gt and pred to the dynamic range
            gt = np.where(gt < self.dynamic_range[0], self.dynamic_range[0], gt)
            gt = np.where(gt > self.dynamic_range[1], self.dynamic_range[1], gt)
            pred = np.where(pred < self.dynamic_range[0], self.dynamic_range[0], pred)
            pred = np.where(pred > self.dynamic_range[1], self.dynamic_range[1], pred)
        else:
            dynamic_range = gt.max() - gt.min()

        # apply mask
        gt = gt[mask == 1]
        pred = pred[mask == 1]
        psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range)
        return float(psnr_value)

    def ssim(self,
             gt: np.ndarray,
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None) -> float:
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

            # Mask gt and pred
            gt = np.where(mask == 0, min(self.dynamic_range), gt)
            pred = np.where(mask == 0, min(self.dynamic_range), pred)

        # Make values non-negative
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
        ssim_value_full, ssim_map = structural_similarity(gt, pred, data_range=dynamic_range, full=True)

        if mask is not None:
            # Follow skimage implementation of calculating the mean value:
            # crop(ssim_map, pad).mean(dtype=np.float64), with pad=3 by default.
            pad = 3
            ssim_value_masked = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)
            return float(ssim_value_masked)
        else:
            return float(ssim_value_full)


def compute_metrics(origin_ct_path, predict_path, mask_path=None):
    metrics = ImageMetrics()
    gt = sitk.ReadImage(origin_ct_path)
    pred = sitk.ReadImage(predict_path)
    if mask_path is not None:
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)
    else:
        mask_array = None
    gt_array = sitk.GetArrayFromImage(gt)
    pred_array = sitk.GetArrayFromImage(pred)
    metrics = metrics.score_patient(gt_array, pred_array, mask_array)
    print(metrics)
    return metrics['mae'], metrics['psnr'], metrics['ssim']


def compute_val_metrics():
    origin_ct_path = '../test_data/brain_1/ct.nii.gz'
    # predict_path = origin_ct_path.replace('ct', 'cbct')
    # origin_ct_path = '../test_data/pelvis4.nii.gz'
    predict_path = "../test_data/predict.nii.gz"
    metrics = compute_metrics(origin_ct_path, predict_path)
    return metrics['mae'], metrics['psnr'], metrics['ssim']


def compute_val_metrics2():
    result_path = "../result"
    src_path = r"D:\Data\SynthRAD\Task2\brain"

    # 用于存储所有样本的指标
    mae_list = []
    psnr_list = []
    ssim_list = []

    print(len(os.listdir(result_path)))

    for p in os.listdir(result_path)[:20]:
        print("-" * 100)
        origin_ct_path = os.path.join(src_path, p, 'ct.nii.gz')
        mask_path = os.path.join(src_path, p, 'mask.nii.gz')
        predict_path = os.path.join(result_path, p, 'predict.nii.gz')

        mae, psnr, ssim = compute_metrics(origin_ct_path, predict_path, mask_path)
        print(mae, psnr, ssim)

        print(f"{p} - MAE: {mae:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
        print("-" * 100)

        mae = float(mae)
        psnr = float(psnr)
        ssim = float(ssim)

        mae_list.append(mae)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
    psnr_mean, psnr_std = np.mean(psnr_list), np.std(psnr_list)
    ssim_mean, ssim_std = np.mean(ssim_list), np.std(ssim_list)

    print(f"MAE  - Mean: {mae_mean:.4f}, Std Dev: {mae_std:.4f}")
    print(f"PSNR - Mean: {psnr_mean:.4f}, Std Dev: {psnr_std:.4f}")
    print(f"SSIM - Mean: {ssim_mean:.4f}, Std Dev: {ssim_std:.4f}")

    return {
        "MAE": {"mean": mae_mean, "std_dev": mae_std},
        "PSNR": {"mean": psnr_mean, "std_dev": psnr_std},
        "SSIM": {"mean": ssim_mean, "std_dev": ssim_std},
    }


if __name__ == '__main__':
    compute_val_metrics2()
