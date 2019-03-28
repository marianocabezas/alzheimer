# Import the required packages
from __future__ import print_function
from scipy import ndimage as nd
import numpy as np
import os
import sys
import time
import re
import traceback
from nibabel import load as load_nii
from time import strftime
from subprocess import call
from scipy.ndimage.morphology import binary_dilation as imdilate
from data_manipulation.generate_features import get_voxels

"""
Utility functions
"""


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


def find_file(name, dirname):
    """

    :param name:
    :param dirname:
    :return:
    """
    result = filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    )

    return os.path.join(dirname, result[0]) if result else None


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


def run_command(command, message=None, stdout=None, stderr=None):
    """
    Function to run and time a shell command using the call function from the
    subprocess module.
    :param command: Command that will be run. It has to comply with the call
    function specifications.
    :param message: Message to be printed before running the command. This is
    an optional parameter and by default its
    None.
    :param stdout: File where the stdout will be redirected. By default we use
    the system's stdout.
    :param stderr: File where the stderr will be redirected. By default we use
    the system's stderr.
    :return:
    """
    if message is not None:
        print_message(message)

    time_f(lambda: call(command), stdout=stdout, stderr=stderr)


def time_f(f, stdout=None, stderr=None):
    """
    Function to time another function.
    :param f: Function to be run. If the function has any parameters, it should
    be passed using the lambda keyword.
    :param stdout: File where the stdout will be redirected. By default we use
    the system's stdout.
    :param stderr: File where the stderr will be redirected. By default we use
    the system's stderr.
    :return: The result of running f.
    """
    # Init
    stdout_copy = sys.stdout
    if stdout is not None:
        sys.stdout = stdout

    start_t = time.time()
    try:
        ret = f()
    except Exception as e:
        ret = None
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('{0}: {1}'.format(type(e).__name__, e.message), file=stderr)
        traceback.print_tb(exc_traceback, file=stderr)
    finally:
        if stdout is not None:
            sys.stdout = stdout_copy

    print(
        time.strftime(
            'Time elapsed = %H hours %M minutes %S seconds',
            time.gmtime(time.time() - start_t)
        )
    )
    return ret


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


"""
Data related functions
"""


def remove_small_regions(img_vol, min_size=3):
    """
    Function that removes blobs with a size smaller than a mininum from a mask
    volume.
    :param img_vol: Mask volume. It should be a numpy array of type bool.
    :param min_size: Mininum size for the blobs.
    :return: New mask without the small blobs.
    """
    blobs, _ = nd.measurements.label(
        img_vol,
        nd.morphology.generate_binary_structure(3, 3)
    )
    labels = filter(bool, np.unique(blobs))
    areas = [np.count_nonzero(np.equal(blobs, l)) for l in labels]
    nu_labels = [l for l, a in zip(labels, areas) if a > min_size]
    nu_mask = reduce(
        lambda x, y: np.logical_or(x, y),
        [np.equal(blobs, l) for l in nu_labels]
    ) if nu_labels else np.zeros_like(img_vol)
    return nu_mask


def best_match(
        fixed, moving,
        fixed_idx, movs,
        c_function=np.correlate,
        verbose=1
):
    """
    Function to find the best match for a given mask in a followup image
    taking into account the intensities of the voxels inside the mask and a set
    of possible translations.
    :param fixed: Fixed image.
    :param moving: Moving image.
    :param fixed_idx: Indexes of the mask for the fixed image.
    :param movs: Possible translations.
    :param c_function: Comparison function. Defaults to cross-correlation.
    :param verbose: Verbosity level.
    :return:
    """

    fixed_idx = np.array(fixed_idx)
    moved_idx = map(lambda mov: fixed_idx + mov, movs)
    fixed_voxels = get_voxels(moving, fixed_idx)
    if moving is not None:
        match_scores = map(
            lambda mov: c_function(fixed_voxels, get_voxels(fixed, mov)),
            moved_idx
        )
    else:
        match_scores = map(
            lambda mov: c_function(get_voxels(fixed, mov)),
            moved_idx
        )
    best_mov = np.argmax(match_scores)
    best_score = match_scores[best_mov]
    best_idx = moved_idx[best_mov]
    if verbose > 1:
        whites = ''.join([' '] * 11)
        print(
            '%s\-The best score was %f (movement (%s))' %
            (whites, best_score, ', '.join(map(str, movs[best_mov]))))
    return best_idx


def improve_mask(image, mask, expansion=1, verbose=1):
    """
    Function that improves a segmentation mask by removing outlier voxels from
    it and adapting the boundary to include voxels that might be part of the
    mask.
    :param image: Image that was segmented.
    :param mask: Original mask.
    :param expansion: Expansion of the bounding box for all dimensions.
    :param verbose: Verbosity level.
    :return: The refined mask of the same shape as mask.
    """

    bounding_box_min = np.min(np.nonzero(mask), axis=1) - expansion
    bounding_box_max = np.max(np.nonzero(mask), axis=1) + 1 + expansion
    bounding_box = map(
        lambda (min_i, max_i): slice(min_i, max_i),
        zip(bounding_box_min, bounding_box_max)
    )

    mask_int = image[mask.astype(bool)]
    bb_int = image[bounding_box]
    mask_mu = np.mean(mask_int)
    nu_mask = bb_int > mask_mu

    fixed_mask = np.zeros_like(image)
    fixed_mask[bounding_box][nu_mask] = True

    if verbose > 1:
        whites = ''.join([' '] * 11)
        print(
            '%s\-Intensity min = %f from %d pixels to %d pixels' % (
                whites,
                mask_mu,
                len(mask_int), np.sum(nu_mask)
            )
        )
    return np.logical_and(fixed_mask, mask)


def get_mask(mask_name, dilate=0, dtype=np.uint8):
    # Lesion mask
    mask_image = load_nii(mask_name).get_data().astype(dtype)
    if dilate > 0:
        mask_image = imdilate(
            mask_image,
            iterations=dilate
        ).astype(dtype)

    return mask_image


def get_normalised_image(image_name, mask, dtype=np.float32):
    mask_bin = mask > 0
    image = load_nii(image_name).get_data()
    image_mu = np.mean(image[mask_bin])
    image_sigma = np.std(image[mask_bin])
    norm_image = (image - image_mu) / image_sigma

    return norm_image.astype(dtype)


def is_dir(path, name):
    return os.path.isdir(os.path.join(path, name))


def get_atrophy_cases(d_path, mask, lesion_tags, dilate):
    patients = get_dirs(d_path)

    patient_paths = map(lambda p: os.path.join(d_path, p), patients)

    max_timepoint = map(
        lambda p_path: len(
            filter(
                lambda f: is_dir(p_path, f),
                os.listdir(p_path)
            )
        ),
        patient_paths
    )
    timepoints_list = map(
        lambda (p_path, max_i): map(
            lambda t: 'flair_time%d-time%d_corrected_matched.nii.gz' % (t, max_i),
            range(1, max_i)
        ),
        zip(patient_paths, max_timepoint)
    )
    for timepoints in timepoints_list:
        timepoints.append('flair_corrected.nii.gz')

    brain_names = map(
        lambda (p_path, max_i): os.path.join(p_path, 'time%d' % max_i, mask),
        zip(patient_paths, max_timepoint)
    )
    masks = map(get_mask, brain_names)

    lesion_names = map(
        lambda p_path: find_file('(' + '|'.join(lesion_tags) + ')', p_path),
        patient_paths
    )
    lesions = map(
        lambda name: get_mask(name, dilate),
        lesion_names
    )

    norm_cases = map(
        lambda (p, mask_i, max_i, timepoints_i): map(
            lambda name: get_normalised_image(
                os.path.join(p, 'time%d' % max_i, name), mask_i
            ),
            timepoints_i
        ),
        zip(patient_paths, masks, max_timepoint, timepoints_list)
    )

    return norm_cases, lesions, masks


def get_newlesion_cases(
        d_path,
        brain_name, lesion_name, source_name, target_name
):
    patients = get_dirs(d_path)
    patient_paths = map(lambda p: os.path.join(d_path, p), patients)
    brain_names = map(
        lambda p_path: os.path.join(p_path, brain_name),
        patient_paths
    )
    brains = map(get_mask, brain_names)
    lesion_names = map(
        lambda p_path: os.path.join(p_path, lesion_name),
        patient_paths
    )
    lesions = map(get_mask, lesion_names)
    norm_source = map(
        lambda (p, mask_i): get_normalised_image(
            os.path.join(p, source_name), mask_i
        ),
        zip(patient_paths, brains)
    )
    norm_target = map(
        lambda (p, mask_i): get_normalised_image(
            os.path.join(p, target_name), mask_i
        ),
        zip(patient_paths, brains)
    )

    return norm_source, norm_target, lesions