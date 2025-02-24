import os
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk

from evaluation import remove_and_create_dir


def load_image(image_path):
    if os.path.exists(image_path):
        image = sitk.ReadImage(image_path)
        return sitk.GetArrayFromImage(image)
    else:
        print(f"Warning: {image_path} does not exist.")
        return None


def save_visualization(cbct, origin_ct, mask, predict, save_path):
    """Visualize and save images."""
    fig, axes = plt.subplots(1, 4, figsize=(8, 2), dpi=100, tight_layout=True)
    axes = axes.ravel()

    axes[0].imshow(cbct, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("CBCT")

    axes[1].imshow(origin_ct, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Original CT")

    axes[2].imshow(mask, cmap="gray" if mask is not None else "hot")
    axes[2].axis("off")
    axes[2].set_title("Mask")

    axes[3].imshow(np.where(mask == 0, -1000, predict), cmap="gray")  # Apply mask if it exists
    axes[3].axis("off")
    axes[3].set_title("Prediction")

    # Save the figure
    plt.subplots_adjust(top=0.85)
    # plt.show()
    plt.savefig(save_path, dpi=100)
    plt.clf()
    plt.close(fig)


def visualize_output(origin_cbct_path, origin_ct_path, predict_path, mask_path):
    origin_cbct_array = load_image(origin_cbct_path)
    origin_ct_array = load_image(origin_ct_path)
    pred_array = load_image(predict_path)
    mask_array = load_image(mask_path) if mask_path else None

    for i in range(10, origin_cbct_array.shape[0], 5):
        print(f"Processing slice {i}...")
        save_visualization(origin_cbct_array[i], origin_ct_array[i], mask_array[i], pred_array[i],
                           save_path=f'../figure/{i}.png')


if __name__ == '__main__':
    result_path = "../result"
    src_path = r"D:\Data\SynthRAD\Task2\brain"

    remove_and_create_dir("../figure")

    if os.path.isdir(result_path):
        p = os.listdir(result_path)[20]
        print("-" * 100)

        origin_ct_path = os.path.join(src_path, p, 'ct.nii.gz')
        origin_cbct_path = os.path.join(src_path, p, 'cbct.nii.gz')
        mask_path = os.path.join(src_path, p, 'mask.nii.gz')
        predict_path = os.path.join(result_path, p, 'predict.nii.gz')

        visualize_output(origin_cbct_path, origin_ct_path, predict_path, mask_path)
    else:
        print(f"Error: {result_path} is not a valid directory.")
