from time import strftime
import time
import os
import argparse
import nibabel as nib
from nibabel import load as load_nii
import numpy as np
from skimage.filters import threshold_otsu
from voxelmorph_model import VoxelMorph


def print_message(message):
    """
    Function to print a message with a custom specification
    :param message: Message to be printed
    :return: None.
    """
    c = color_codes()
    dashes = ''.join(['-'] * (len(message) + 11))
    print(dashes)
    print(
        '%s[%s]%s %s' %
        (c['c'], strftime("%H:%M:%S", time.localtime()), c['nc'], message)
    )
    print(dashes)


def parse_args():
    """
    Function to control the arguments of the python script when called from the
    command line.
    :return: Dictionary with the argument values
    """
    parser = argparse.ArgumentParser(
        description='Run the longitudinal MS lesion segmentation docker.'
    )
    parser.add_argument(
        '-f', '--files-path',
        dest='dataset_path',
        default='/home/owner/data/OASIS2Reoriented',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-s', '--save_name',
        dest='model_name',
        default='model',
        help='Name of the file to store the model weights.'
    )
    parser.add_argument(
        '-i', '--posIDX',
        dest='index',
        type=int, default=0,
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-g', '--gpu',
        dest='gpu_id',
        type=int, default=0,
        help='GPU id number'
    )
    return vars(parser.parse_args())


def color_codes():
    """
    Function that returns a custom dictionary with ASCII codes related to
    colors.
    :return: Custom dictionary with ASCCI codes for terminal colors.
    """
    codes = {
        'nc': '\033[0m',
        'b': '\033[1m',
        'k': '\033[0m',
        '0.25': '\033[30m',
        'dgy': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'gc': '\033[32m;0m',
        'bg': '\033[32;1m',
        'y': '\033[33m',
        'c': '\033[36m',
        '0.75': '\033[37m',
        'lgy': '\033[37m',
    }
    return codes


def get_dirs(path):
    """
    Function to get the folder name of the patients given a path.
    :param path: Folder where the patients should be located.
    :return: List of patient names.
    """
    # All patients (full path)
    patient_paths = sorted(
        filter(
            lambda d: os.path.isdir(os.path.join(path, d)),
            os.listdir(path)
        )
    )
    # Patients used during training
    return patient_paths


def cnn_registration(
        d_path=None,
        verbose=1,
):
    """
        Function that applies a CNN-based registration approach. The goal of
        this network is to find the atrophy deformation, and how it affects the
        lesion mask, manually segmented on the baseline image.
        :param d_path: Path where the whole database is stored. If not specified,
        it will be read from the command parameters.
        :param verbose: Verbosity level.
        :return: None.
        """

    # Init
    c = color_codes()
    if d_path is None:
        d_path = parse_args()['dataset_path']
    patients = get_dirs(d_path)
    index = parse_args()['index']
    patient = patients[index]

    time_str = strftime("%H:%M:%S")
    print('\n%s[%s]%s CNN registration' % (
        c['c'], time_str, c['nc']
    ))

    global_start = time.time()

    # Main loop
    if verbose > 0:
        print(
            '%s[%s]%s Testing CNN with patient %s %s(%d/%d)%s' %
            (
                c['c'], strftime("%H:%M:%S"),
                c['g'], patient,
                c['c'], index + 1, len(patients), c['nc']
            )
        )

    # Create the network and run it.
    reg_net = VoxelMorph().cuda()
    reg_net.load_model(
        os.path.join(d_path, patient, parse_args()['model_name'])
    )

    # Baseline image (testing)
    source_nii = load_nii(
        os.path.join(
            d_path, patient, patient + '_MR1.nii'
        )
    )
    source_image = np.squeeze(source_nii.get_data())
    source_shape = source_nii.get_data().shape

    # Follow-up image (testing)
    target_image = np.squeeze(
        load_nii(
            os.path.join(
                d_path, patient, patient + '_MR2.nii'
            )
        ).get_data()
    )

    # Brain mask
    source_otsu = threshold_otsu(target_image)
    target_otsu = threshold_otsu(target_image)
    brain_bin = np.logical_or(
        source_image > source_otsu, target_image > target_otsu
    )
    brain_mask = np.reshape(
        brain_bin.astype(np.int8),
        (1, 1) + brain_bin.shape
    )

    # Normalised images
    source_mu = np.mean(source_image[brain_bin])
    source_sigma = np.std(source_image[brain_bin])
    source_image = np.reshape(source_image, (1, 1) + source_image.shape)
    norm_source = (source_image - source_mu) / source_sigma

    target_mu = np.mean(target_image[brain_bin])
    target_sigma = np.std(target_image[brain_bin])
    target_image = np.reshape(target_image, (1, 1) + target_image.shape)
    norm_target = (target_image - target_mu) / target_sigma

    df = reg_net.get_deformation(norm_source, norm_target)
    source_moved = reg_net.transform_image(norm_source, norm_target)
    mask_moved = reg_net.transform_mask(norm_source, norm_target, brain_mask)

    source_mov = source_moved[0] * source_sigma + source_mu
    source_nii.get_data()[:] = np.reshape(
        source_mov,
        source_shape
    )
    source_nii.to_filename(
        os.path.join(d_path, patient, 'voxelmorph_moved.nii.gz')
    )
    mask_nii = nib.Nifti1Image(
        np.reshape(mask_moved[0], source_shape),
        source_nii.get_qform(),
        source_nii.get_header()
    )
    mask_nii.to_filename(
        os.path.join(d_path, patient, 'voxelmorph_mask.nii.gz')
    )

    df_mask = np.repeat(np.expand_dims(brain_mask, -1), 3, -1)
    df_nii = nib.Nifti1Image(
        np.reshape(
            np.moveaxis(df[0], 0, -1) * df_mask,
            source_shape + (-1,)
        ),
        source_nii.get_qform(),
        source_nii.get_header()
    )
    df_nii.to_filename(
        os.path.join(d_path, patient, 'voxelmorph_df.nii.gz')
    )

    # Finished
    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - global_start)
        )
        print_message(
            '%sAll patients finished %s(total time %s)%s' %
            (c['r'], c['b'], time_str, c['nc'])
        )


if __name__ == "__main__":
    cnn_registration()
