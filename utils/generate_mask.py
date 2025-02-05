import SimpleITK as sitk
import numpy as np
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.measure import label


def normalize(img, min_, max_):
    return (img - min_) / (max_ - min_)


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def get_3d_mask(img):
    mask = np.zeros(img.shape).astype(int)

    otsu_threshold = threshold_otsu(img)

    mask[img > otsu_threshold] = 1

    selem = morphology.ball(3)

    mask = morphology.binary_dilation(mask, selem)
    mask = morphology.binary_dilation(mask, selem)
    mask = morphology.binary_erosion(mask, selem)
    mask = morphology.binary_dilation(mask, selem)
    # mask = morphology.binary_dilation(mask, selem)

    remove_holes = morphology.remove_small_holes(mask, area_threshold=500)

    largest_cc = getLargestCC(remove_holes)

    return img, largest_cc.astype(int)


if __name__ == '__main__':

    # case = 'brain'
    case = 'pelvis4'

    ct_path=f'../test_data/{case}.nii.gz'
    mask_path=ct_path.replace('.nii.gz', '_mask.nii.gz')

    ct = sitk.ReadImage(ct_path)
    ct_array = sitk.GetArrayFromImage(ct)

    ct_array, mask_array = get_3d_mask(ct_array)
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(ct)
    sitk.WriteImage(mask, mask_path)

    # mask = sitk.ReadImage(mask_path)
    # mask_array=sitk.GetArrayFromImage(mask)

    predict = sitk.ReadImage(f'../test_data/predict.nii.gz')
    predict_array = sitk.GetArrayFromImage(predict)
    predict_array[mask_array == 0] = -1000
    predict = sitk.GetImageFromArray(predict_array)
    predict.CopyInformation(ct)
    sitk.WriteImage(predict, f'../test_data/predict_with_mask.nii.gz')
