from time import strftime
import time
import os
import argparse
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
        '-f', '--old',
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
        '-l', '--lambda',
        dest='lambda',
        type=int, default=1,
        help='Parameter to store the working directory.'
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
    parser.add_argument(
        '-b', '--batch_size',
        dest='batch_size',
        type=int, default=4,
        help='Batch size')
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=150000,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=5000,
        help='Patience for early stopping'
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
    training = patients[:index] + patients[index+1:]

    time_str = strftime("%H:%M:%S")
    print('\n%s[%s]%s CNN registration' % (
        c['c'], time_str, c['nc']
    ))

    global_start = time.time()

    # Main loop
    if verbose > 0:
        print(
            '%s[%s]%s Starting CNN with patient %s %s(%d/%d)%s' %
            (
                c['c'], strftime("%H:%M:%S"),
                c['g'], patient,
                c['c'], index + 1, len(patients), c['nc']
            )
        )

    # Load mask, source and target and expand the dimensions.
    image_names1 = map(
        lambda patient_i: os.path.join(
            d_path, patient_i, patient_i + '_MR1.nii'
        ),
        training
    )
    image_names2 = map(
        lambda patient_i: os.path.join(
            d_path, patient_i, patient_i + '_MR2.nii'
        ),
        training
    )

    # Baseline images
    source_niis = map(
        lambda image_name: load_nii(image_name),
        image_names1
    )
    source_images = map(
        lambda source_nii: np.squeeze(source_nii.get_data()),
        source_niis
    )

    # Follow-up images
    target_niis = map(
        lambda image_name: load_nii(image_name),
        image_names2
    )
    target_images = map(
        lambda source_nii: np.squeeze(source_nii.get_data()),
        target_niis
    )

    # Brain mask
    source_otsu = map(threshold_otsu, source_images)
    target_otsu = map(threshold_otsu, target_images)
    brain_bins = map(
        lambda (th1, th2, im1, im2): np.logical_or(
            im1 > th1, im2 > th2
        ),
        zip(source_otsu, target_otsu, source_images, target_images)
    )
    brain_masks = map(
        lambda brain_mask: brain_mask.astype(np.uint8),
        brain_bins
    )

    # Baseline images norm
    source_mus = map(
        lambda (source, brain): np.mean(source[brain]),
        zip(source_images, brain_bins)
    )
    source_sigmas = map(
        lambda (source, brain): np.std(source[brain]),
        zip(source_images, brain_bins)
    )
    norm_source = map(
        lambda (source, mu, sigma): ((source - mu) / sigma).astype(np.float32),
        zip(source_images, source_mus, source_sigmas)
    )

    # Follow-up images norm
    target_mus = map(
        lambda (source, brain): np.mean(source[brain]),
        zip(target_images, brain_bins)
    )
    target_sigmas = map(
        lambda (source, brain): np.std(source[brain]),
        zip(target_images, brain_bins)
    )
    norm_target = map(
        lambda (source, mu, sigma): ((source - mu) / sigma).astype(np.float32),
        zip(target_images, target_mus, target_sigmas)
    )

    # Create the network and run it.
    reg_net = VoxelMorph(
        device=torch.device('cuda:%d' % parse_args()['gpu_id'] if torch.cuda.is_available() else "cpu")
    ).cuda()
    reg_net.register(
        norm_source,
        norm_target,
        brain_masks,
        parse_args()['batch_size'],
        epochs=parse_args()['epochs'],
        patience=parse_args()['patience'],
        device='cuda:%d' % parse_args()['gpu_id'],
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

    reg_net.save_model(
        os.path.join(d_path, patient, parse_args()['model_name'])
    )
    
if __name__ == "__main__":
    cnn_registration()
