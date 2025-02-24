import os
import numpy as np
import nibabel as nib
from skimage.transform import resize

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image



def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).

    Remove outliers voxels first, then min-max scale.

    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image


def extract_middle_slices(nifti_files, num_slices=80, target_shape=(256, 256)):
    all_slices = []

    for nifti_file in nifti_files:
        img = nib.load(nifti_file)

        data = img.get_fdata()
        data = irm_min_max_preprocess(data)

        # Calculate the indices for the middle slices
        start_slice = (data.shape[2] - num_slices) // 2
        end_slice = start_slice + num_slices


        # Initialize an empty list to store 2D slices
        slices = []

        # Iterate through the slices of the NIfTI volume
        for slice_idx in range(start_slice, end_slice):
            slice_2d = data[:, :, slice_idx]
            # Resize to the target shape (256x256)
            slice_2d = resize(slice_2d, target_shape, mode='constant', anti_aliasing=True)
            slices.append(slice_2d)


        all_slices.append(slices)

    return np.concatenate(all_slices)

def save_as_numpy_array(data, save_path):
    np.save(save_path, data)


if __name__ == "__main__":
    # Path to directory containing NIfTI files
    nifti_directory = "nifti_files_dir_path"
    nifti_files = [os.path.join(nifti_directory, f) for f in os.listdir(nifti_directory) if f.endswith('.nii.gz')]
    nifti_files = sorted(nifti_files)

    # Extract middle slices and concatenate
    middle_slices = extract_middle_slices(nifti_files)

    # Save as Numpy array
    save_path =r"save_dir_path\contrast.npy"
    save_as_numpy_array(middle_slices, save_path)

    print("Middle slices saved as Numpy array.")
