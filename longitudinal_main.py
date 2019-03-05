"""
The main file running inside the docker (the starting point)
"""
# Import the required packages
from __future__ import print_function
from scipy import ndimage as nd
import numpy as np
import torch
import argparse
import os
import sys
import time
import re
import traceback
from itertools import product
from nibabel import load as load_nii
import SimpleITK as sitk
import nibabel as nib
from time import strftime
from subprocess import call
from scipy.ndimage.morphology import binary_erosion as erode
from scipy.ndimage.morphology import binary_dilation as imdilate
from sklearn.metrics import mean_squared_error
from data_manipulation.sitk import itkn4, itkhist_match, itksubtraction, itkdemons
from data_manipulation.generate_features import get_mask_voxels, get_voxels
from data_manipulation.information_theory import bidirectional_mahalanobis, normalized_mutual_information
from models import LongitudinalNet

# torch.backends.cudnn.benchmark = True

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


def parse_args():
    """
    Function to control the arguments of the python script when called from the
    command line.
    :return: Dictionary with the argument values
    """
    parser = argparse.ArgumentParser(
        description='Run the longitudinal MS lesion segmentation docker.'
    )
    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        '-d', '--dataset-path',
        dest='dataset_path',
        default='/home/mariano/DATA/Australia60m/Workstation',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-D', '--seg-dataset-path',
        dest='seg_dataset_path',
        default='/home/mariano/DATA/VH',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-l', '--lambda',
        dest='lambda',
        type=float, default=1,
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-L', '--losses-list',
        dest='loss_idx',
        nargs='+', type=int, default=[0, 1],
        help='List of loss indices. '
             '0: Global subtraction gradient\n'
             '1: Lesion subtraction gradient\n'
             '2: Global cross-correlation\n'
             '3: Lesion cross-correlation\n'
             '4: Global mean squared error\n'
             '5: Lesion mahalanobis distance between timepoints\n'
             '6: Lesion histogram difference\n'
             '7: Deformation regularization\n'
             '8: Modulo maximisation'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=100,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=50,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '-r', '--dilate-radius',
        dest='dilate',
        type=int, default=2,
        help='Number of dilate repetitions (equivalent to radius)'
    )
    parser.add_argument(
        '-g', '--gpu',
        dest='gpu_id',
        type=int, default=0,
        help='GPU id number'
    )
    parser.add_argument(
        '--data-smooth',
        dest='data_smooth',
        action='store_true', default=False,
        help='Whether or not to apply a smoothing layer to the input'
    )
    parser.add_argument(
        '--df-smooth',
        dest='df_smooth',
        action='store_true', default=False,
        help='Whether or not to apply a smoothing layer to the '
             'deformation field'
    )
    parser.add_argument(
        '--trainable-smooth',
        dest='train_smooth',
        action='store_true', default=False,
        help='Whether or not to make the smoothing trainable'
    )
    return vars(parser.parse_args())


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


"""
> Main functions (options)
"""


def initial_analysis(
        d_path=None,
        pd_tags=list(['DP', 'PD', 'pd', 'dp']),
        t1_tags=list(['MPRAGE', 'mprage', 'MPR', 'mpr', 'T1', 't1']),
        t2_tags=list(['T2', 't2']),
        flair_tags=list(['FLAIR', 'flair', 'dark_fluid', 'dark_fluid']),
        verbose=1,
):
    """
    Function that applies some processing based on our paper:
        M. Cabezas, J.F. Corral, A. Oliver, Y. Diez, M. Tintore, C. Auger,
        X. Montalban, X. Llado, D. Pareto, A. Rovira.
        'Automatic multiple sclerosis lesion detection in brain MRI by FLAIR
        thresholding'
        In American Journal of Neuroradiology, Volume 37(10), 2016,
        Pages 1816-1823
        http://dx.doi.org/10.3174/ajnr.A4829
    :param d_path: Path where the whole database is stored. If not specified,
    it will be read from the command parameters.
    :param pd_tags: PD tags for the preprocessing method in itk_tools.
    :param t1_tags: T1 tags for the preprocessing method in itk_tools.
    :param t2_tags: T2 tags for the preprocessing method in itk_tools.
    :param flair_tags: FLAIR tags for the preprocessing method in itk_tools.
    :param verbose: Verbosity level.
    :return: None.
    """

    c = color_codes()

    # Init
    if d_path is None:
        d_path = parse_args()['dataset_path']
    robex = os.path.join(parse_args()['tools_path'], 'ROBEX', 'runROBEX.sh')
    tags = [pd_tags, t1_tags, t2_tags, flair_tags]
    images = ['pd', 't1', 't2', 'flair']
    patients = get_dirs(d_path)

    time_str = strftime("%H:%M:%S")
    print('\n%s[%s]%s Longitudinal analysis' % (
        c['c'], time_str, c['nc']
    ))

    global_start = time.time()

    # Main loop
    for i, patient in enumerate(patients):
        patient_start = time.time()
        print(
            '%s[%s]%s Starting preprocessing with patient %s %s(%d/%d)%s' %
            (
                c['c'], strftime("%H:%M:%S"),
                c['g'], patient,
                c['c'], i + 1, len(patients),
                c['nc']
            )
        )
        patient_path = os.path.join(d_path, patient)

        timepoints = get_dirs(os.path.join(d_path, patient))

        if verbose > 0:
            print('/-------------------------------\\')
            print('|         Preprocessing         |')
            print('\\-------------------------------/')

        # First we skull strip all the timepoints using ROBEX. We could
        #  probably use another method that has better integration with
        #  our code...
        for folder in timepoints:

            full_folder = os.path.join(patient_path, folder) + '/'

            ''' ROBEX '''
            if verbose > 0:
                print('\t-------------------------------')
                print('\t            ROBEX              ')
                print('\t-------------------------------')
            # Check if there is a brainmask
            brainmask_name = find_file('brainmask.nii.gz', full_folder)
            if brainmask_name is None:
                brainmask_name = os.path.join(full_folder, 'brainmask.nii.gz')
                brain_name = os.path.join(full_folder, 'stripped.nii.gz')
                base_pd = filter(
                    lambda x: not os.path.isdir(x) and re.search(
                        r'(\w|\W)*(t1|T1)(\w|\W)', x
                    ),
                    os.listdir(full_folder)
                )[0]
                base_image = os.path.join(full_folder, base_pd)
                run_command(
                    [robex, base_image, brain_name, brainmask_name],
                    'ROBEX skull stripping - %s (%s)' % (folder, patient),
                )
                if find_file('stripped.nii.gz', full_folder):
                    os.remove(brain_name)

            # The next step is to apply bias correction
            """N4 preprocessing"""
            if verbose > 0:
                print('\t-------------------------------')
                print('\t        Bias correction        ')
                print('\t-------------------------------')
            original_files = map(
                lambda t: find_file('(' + '|'.join(t) + ')', full_folder), tags
            )
            map(
                lambda (b, im): itkn4(
                    b, full_folder, im,
                    max_iters=200, levels=4, verbose=verbose
                ),
                filter(
                    lambda (b, _): b is not None,
                    zip(original_files, images)
                )
            )

        # Followup setup
        followup = timepoints[-1]
        followup_path = os.path.join(patient_path, followup)

        bm_tag = 'brainmask.nii.gz'
        brains = map(
            lambda t: load_nii(
                os.path.join(patient_path, t, bm_tag)
            ).get_data(),
            timepoints
        )
        brain = reduce(np.logical_or, brains)
        brain_nii = load_nii(
            os.path.join(patient_path, timepoints[-1], 'brainmask.nii.gz')
        )
        brain_nii.get_data()[:] = brain
        brain_nii.to_filename(
            os.path.join(followup_path, 'union_brainmask.nii.gz')
        )

        followup_files = map(
            lambda im: find_file(im + '_corrected.nii.gz', followup_path),
            images
        )
        for baseline in timepoints[:-1]:
            image_tag = baseline + '-' + followup
            baseline_path = os.path.join(patient_path, baseline)
            baseline_files = map(
                lambda im: find_file(im + '_corrected.nii.gz', baseline_path),
                images
            )

            # Now it's time to histogram match everything to the last timepoint.
            """Histogram matching"""
            if verbose > 0:
                print('\t-------------------------------')
                print('\t       Histogram matching      ')
                print('\t-------------------------------')
            map(
                lambda (f, b, im): itkhist_match(
                    f, b,
                    followup_path, im + '_' + image_tag,
                    verbose=verbose
                ),
                filter(
                    lambda (f, b, im): b is not None and f is not None,
                    zip(followup_files, baseline_files, images)
                )
            )

            """Subtraction"""
            if verbose > 0:
                print('/-------------------------------\\')
                print('|          Subtraction          |')
                print('\\-------------------------------/')
            hm_tag = '_corrected_matched.nii.gz'
            baseline_files = map(
                lambda im: find_file(
                    im + '_' + image_tag + hm_tag,
                    followup_path
                ),
                images
            )

            map(
                lambda (f, b, im): itksubtraction(
                    f, b,
                    followup_path, im + '_' + image_tag,
                    verbose=verbose
                ),
                filter(
                    lambda (f, b, im): b is not None and f is not None,
                    zip(followup_files, baseline_files, images)
                )
            )

            """Deformation"""
            if verbose > 0:
                print('/-------------------------------\\')
                print('|          Deformation          |')
                print('\\-------------------------------/')
            mask = sitk.ReadImage(
                os.path.join(followup_path, 'union_brainmask.nii.gz')
            )
            map(
                lambda (f, b, im): itkdemons(
                    f, b, mask,
                    followup_path, im + '_' + image_tag,
                    verbose=verbose
                ),
                filter(
                    lambda (f, b, im): f is not None and b is not None,
                    zip(baseline_files, followup_files, images)
                )
            )

        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - patient_start)
        )
        print(
            '%sPatient %s finished%s (total time %s)\n' %
            (c['r'], patient, c['nc'], time_str)
        )

    time_str = time.strftime(
        '%H hours %M minutes %S seconds',
        time.gmtime(time.time() - global_start)
    )
    print_message(
        '%sAll patients finished %s(total time %s)%s' %
        (c['r'],  c['b'], time_str, c['nc'])
    )


def naive_registration(
        d_path=None,
        image='flair',
        lesion_tags=list(['_bin', 'lesion', 'lesionMask']),
        fixed_folder='time6',
        moving_folder='time1',
        c_function=lambda x, y: -bidirectional_mahalanobis(x, y),
        width=5,
        dim=3,
        refine=False,
        expansion=1,
        verbose=1,
):
    """
    Function that applies a lesion by lesion naive registration. The idea is
    that after 3 years, the atrophy of the MS patient is noticeable and that
    causes a movement on the chronic MS lesions (due to the ventricle
    expansion). The assumption is that we can use a naive registration
    approach that open a sliding window around the baseline mask and tries to
    find the best match in terms of intensity similarity on the follow-up
    image.
    :param d_path: Path where the whole database is stored. If not specified,
    it will be read from the command parameters.
    :param image: Image name that will be used for matching.
    :param lesion_tags: Tags that may be contained on the lesion mask filename.
    :param fixed_folder: Folder that contains the fixed image.
    :param moving_folder: Folder that contains the moving image.
    :param c_function: Function that compares two matrices and returns the
    value of their similarity.
    :param width: Width of the sliding window.
    :param dim: Number of dimensions of the fixed and moving images.
    :param refine: Whether to refine the original mask or not. The refined mask
    is based on the gaussian distribution of the voxels.
    inside the original mask and the bounding box of the mask with a user
    defined expansion.
    :param expansion: expansion of the bounding box for the refinement.
    :param verbose: Verbosity level.
    :return: None.
    """

    c = color_codes()

    # Init
    if d_path is None:
        d_path = parse_args()['dataset_path']
    movs = map(
        lambda (e, s): list(product(range(0, e, s), repeat=dim)),
        [(-width, -1), (width, 1)]
    )
    movs = map(
        lambda lmovs: filter(lambda mov: np.linalg.norm(mov) <= width, lmovs),
        movs
    )
    np_movs = np.unique(np.concatenate(movs), axis=0)
    patients = get_dirs(d_path)

    time_str = strftime("%H:%M:%S")
    print('\n%s[%s]%s Naive registration' % (
        c['c'], time_str, c['nc']
    ))

    global_start = time.time()

    # Main loop
    for i, patient in enumerate(patients):
        patient_start = time.time()
        if verbose > 0:
            print(
                '%s[%s]%s Starting preprocessing with patient %s %s(%d/%d)%s' %
                (
                    c['c'], strftime("%H:%M:%S"),
                    c['g'], patient,
                    c['c'], i + 1, len(patients), c['nc']
                )
            )

        patient_path = os.path.join(d_path, patient)

        fixed_name = image + '_corrected.nii.gz'
        fixed = load_nii(
            os.path.join(patient_path, fixed_folder, fixed_name)
        ).get_data()
        moving_tag = '_' + moving_folder + '-' + fixed_folder
        moving_name = image + moving_tag + '_corrected_matched.nii.gz'
        moving = load_nii(
            os.path.join(patient_path, fixed_folder, moving_name)
        ).get_data()

        mask_name = find_file('(' + '|'.join(lesion_tags) + ')', patient_path)
        if verbose > 1:
            print(
                '%s-Using image %s%s%s' %
                (
                    ''.join([' '] * 11),
                    c['b'], mask_name,
                    c['nc']
                )
            )
        if mask_name is not None:
            strel = np.ones([3] * dim)
            mask_nii = load_nii(mask_name)
            mask = mask_nii.get_data()
            labels, nlabels = nd.label(mask, structure=strel)

            if verbose > 1:
                print(
                    '%s-Refining the %s%d%s lesions' %
                    (
                        ''.join([' '] * 11),
                        c['b'], nlabels,
                        c['nc']
                    )
                )

            if refine:
                masks = map(
                    lambda l: improve_mask(
                        moving, labels == l,
                        expansion, verbose
                    ),
                    range(1, nlabels + 1)
                )

                refined_mask = reduce(np.logical_or, masks)
                mask_nii.get_data()[:] = refined_mask
                mask_nii.to_filename(
                    os.path.join(patient_path, 'refined_mask.nii.gz')
                )
            else:
                masks = map(lambda l: labels == l, range(1, nlabels + 1))

            if verbose > 1:
                print(
                    '%s-Moving %s%d%s lesions' %
                    (
                        ''.join([' '] * 11),
                        c['b'], nlabels,
                        c['nc']
                    )
                )

            c_functions = {
                'nmi': lambda x, y: normalized_mutual_information(x, y, 32),
                'mahal': lambda x, y: -bidirectional_mahalanobis(x, y),
                'mse': lambda x, y: -mean_squared_error(x, y)
            }

            for n_f, c_f in c_functions.items():

                matches = map(
                    lambda mask: best_match(
                        fixed, moving,
                        get_mask_voxels(mask), np_movs,
                        c_function=c_f,
                        verbose=verbose
                    ),
                    masks
                )

                nulesion_idx = np.concatenate(matches, axis=0)
                numask = np.zeros_like(mask)
                for idx in nulesion_idx:
                    numask[tuple(idx)] = True

                mask_nii.get_data()[:] = numask
                if refine:
                    mask_nii.to_filename(
                        os.path.join(patient_path, '%s_ref_mask.nii.gz' % n_f)
                    )
                else:
                    mask_nii.to_filename(
                        os.path.join(patient_path, '%s_mask.nii.gz' % n_f)
                    )

        if verbose > 0:
            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - patient_start)
            )
            print(
                '%sPatient %s finished%s (total time %s)\n' %
                (c['r'], patient, c['nc'], time_str)
            )

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - global_start)
        )
        print_message(
            '%sAll patients finished %s(total time %s)%s' %
            (c['r'], c['b'], time_str, c['nc'])
        )


def deformationbased_registration(
        d_path=None,
        image='flair_time1-time6_multidemons_deformation.nii.gz',
        lesion_tags=list(['_bin', 'lesion', 'lesionMask']),
        fixed_folder='time6',
        vector_op=np.max,
        dim=3,
        verbose=1,
):
    """
    Function that applies a translation based solely on a deformation field. It
    basically moves the mask using the average deformation inside the mask.
    :param d_path: Path where the whole database is stored. If not specified,
    it will be read from the command parameters.
    :param image: Image name of the deformation image
    :param lesion_tags: Tags that may be contained on the lesion mask filename.
    :param fixed_folder: Folder that contains the fixed image..
    :param vector_op: Operation used on the deformation field of each lesion.
    :param dim: Number of dimensions of the fixed and moving images.
    :param verbose: Verbosity level.
    :return: None.
    """
    c = color_codes()

    # Init
    if d_path is None:
        d_path = parse_args()['dataset_path']
    patients = get_dirs(d_path)

    time_str = strftime("%H:%M:%S")
    print('\n%s[%s]%s Naive registration' % (
        c['c'], time_str, c['nc']
    ))

    global_start = time.time()

    # Main loop
    for i, patient in enumerate(patients):
        patient_start = time.time()
        if verbose > 0:
            print(
                '%s[%s]%s Starting deformation with patient %s %s(%d/%d)%s' %
                (
                    c['c'], strftime("%H:%M:%S"),
                    c['g'], patient,
                    c['c'], i + 1, len(patients), c['nc']
                )
            )

        patient_path = os.path.join(d_path, patient)
        defo_path = os.path.join(patient_path, fixed_folder)

        # Deformation loading
        defo_name = find_file(image, defo_path)
        defo = np.moveaxis(np.squeeze(load_nii(defo_name).get_data()), -1, 0)

        mask_name = find_file('(' + '|'.join(lesion_tags) + ')', patient_path)
        if mask_name is not None:
            mask_nii = load_nii(mask_name)
            mask = mask_nii.get_data()

            strel = np.ones([3] * dim)
            label_im, nlabels = nd.label(mask, structure=strel)
            labels = range(1, nlabels + 1)

            l_masks = map(lambda l_i: label_im == l_i, labels)
            lb_masks = map(
                lambda (m_i, in_i): np.logical_xor(m_i, in_i),
                zip(l_masks, map(lambda m_i: erode(m_i, strel), l_masks)))

            l_voxels = map(get_mask_voxels, l_masks)
            vectors = map(
                lambda lm: np.stack(
                    map(
                        lambda d: d[lm],
                        defo
                    ),
                    axis=0
                ),
                lb_masks
            )

            v_ops = {
                'max': np.max,
                'min': np.min,
                'mean': np.mean,
                'median': np.median
            }

            for n_op, v_op in v_ops.items():
                v_means = map(lambda v: np.round(v_op(v, axis=-1)), vectors)
                if verbose > 1:
                    whites = ''.join([' '] * 11)
                    for mov in v_means:
                        print(
                            '%s\-Defo movement was (%s)' %
                            (whites, ', '.join(map(str, mov)))
                        )

                matches = map(lambda (v, m): v + m, zip(l_voxels, v_means))
                nulesion_idx = np.concatenate(matches, axis=0)

                numask = np.zeros_like(mask)
                for idx in nulesion_idx:
                    numask[tuple(idx.astype(np.int))] = True

                mask_nii.get_data()[:] = numask
                mask_nii.to_filename(
                    os.path.join(patient_path, 'defo_mask_%s.nii.gz' % n_op)
                )

        if verbose > 0:
            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - patient_start)
            )
            print(
                '%sPatient %s finished%s (total time %s)\n' %
                (c['r'], patient, c['nc'], time_str)
            )

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - global_start)
        )
        print_message(
            '%sAll patients finished %s(total time %s)%s' %
            (c['r'], c['b'], time_str, c['nc'])
        )


def subtraction_registration(
        d_path=None,
        image='flair_time1-time6_subtraction.nii.gz',
        lesion_tags=list(['_bin', 'lesion', 'lesionMask']),
        fixed_folder='time6',
        width=5,
        dim=3,
        verbose=1,
):
    """
    Function that applies a lesion by lesion naive registration. The idea is
     that after 3 years, the atrophy of the MS patient is noticeable and that
     causes a movement on the chronic MS lesions (due to the ventricle
     expansion). The assumption is that we can use a naive registration
     approach that open a sliding window around the baseline mask and tries to
     find the best match in terms of intensity similarity on the follow-up
     image.
    :param d_path: Path where the whole database is stored. If not specified,
    it will be read from the command parameters.
    :param image: Image name that will be used for matching.
    :param lesion_tags: Tags that may be contained on the lesion mask filename.
    :param fixed_folder: Folder that contains the fixed image.
    :param width: Width of the sliding window.
    :param dim: Number of dimensions of the fixed and moving images.
    :param verbose: Verbosity level.
    :return: None.
    """

    c = color_codes()

    # Init
    if d_path is None:
        d_path = parse_args()['dataset_path']
    movs = map(
        lambda (e, s): list(product(range(0, e, s), repeat=dim)),
        [(-width, -1), (width, 1)]
    )
    movs = map(
        lambda lmovs: filter(lambda mov: np.linalg.norm(mov) <= width, lmovs),
        movs
    )
    np_movs = np.unique(np.concatenate(movs), axis=0)
    patients = get_dirs(d_path)

    time_str = strftime("%H:%M:%S")
    print('\n%s[%s]%s Naive registration' % (
        c['c'], time_str, c['nc']
    ))

    global_start = time.time()

    # Main loop
    for i, patient in enumerate(patients):
        patient_start = time.time()
        if verbose > 0:
            print(
                '%s[%s]%s Starting deformation with patient %s %s(%d/%d)%s' %
                (
                    c['c'], strftime("%H:%M:%S"),
                    c['g'], patient,
                    c['c'], i + 1, len(patients), c['nc']
                )
            )

        patient_path = os.path.join(d_path, patient)
        defo_path = os.path.join(patient_path, fixed_folder)

        # Subtraction loading
        sub_name = find_file(image, defo_path)
        sub = load_nii(sub_name).get_data()

        mask_name = find_file('(' + '|'.join(lesion_tags) + ')', patient_path)
        if verbose > 1:
            print(
                '%s-Using image %s%s%s' %
                (
                    ''.join([' '] * 11),
                    c['b'], mask_name,
                    c['nc']
                )
            )
        if mask_name is not None:
            strel = np.ones([3] * dim)
            mask_nii = load_nii(mask_name)
            mask = mask_nii.get_data()
            labels, nlabels = nd.label(mask, structure=strel)

            if verbose > 1:
                print(
                    '%s-Moving %s%d%s lesions' %
                    (
                        ''.join([' '] * 11),
                        c['b'], nlabels,
                        c['nc']
                    )
                )

            c_functions = {
                'std': np.std,
                'mean': np.mean,
            }

            for n_f, c_f in c_functions.items():

                matches = map(
                    lambda l: best_match(
                        sub, None,
                        get_mask_voxels(labels == l), np_movs,
                        c_function=c_f,
                        verbose=verbose
                    ),
                    range(1, nlabels + 1)
                )

                nulesion_idx = np.concatenate(matches, axis=0)
                numask = np.zeros_like(mask)
                for idx in nulesion_idx:
                    numask[tuple(idx)] = True

                mask_nii.get_data()[:] = numask
                mask_nii.to_filename(
                    os.path.join(patient_path, 'sub_mask_%s.nii.gz' % n_f)
                )

        if verbose > 0:
            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - patient_start)
            )
            print(
                '%sPatient %s finished%s (total time %s)\n' %
                (c['r'], patient, c['nc'], time_str)
            )

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - global_start)
        )
        print_message(
            '%sAll patients finished %s(total time %s)%s' %
            (c['r'], c['b'], time_str, c['nc'])
        )


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


def cnn_registration(
        d_path=None,
        lesion_tags=list(['_bin', 'lesion', 'lesionMask']),
        folder='time6',
        mask='union_brainmask.nii.gz',
        source_name='flair_moved.nii.gz',
        target_name='flair_processed.nii.gz',
        brain_name='brainmask.nii.gz',
        lesion_name='gt_mask.nii',
        verbose=1,
):
    """
        Function that applies a CNN-based registration approach. The goal of
        this network is to find the atrophy deformation, and how it affects the
        lesion mask, manually segmented on the baseline image.
        :param d_path: Path where the whole database is stored. If not
         specified,it will be read from the command parameters.
        :param lesion_tags: Tags that may be contained on the lesion mask
         filename.
        :param folder: Folder that contains the images.
        :param mask: Brainmask name.
        :param source_name: Name of the source image from the segmentation
         dataset.
        :param target_name: Name of the target image from the segmentation
         dataset.
        :param brain_name: Name of the brainmask image from the segmentation
         dataset.
        :param lesion_name: Name of the lesionmask image from the segmentation
         dataset.
        :param verbose: Verbosity level.
        :return: None.
        """

    c = color_codes()

    # Init
    if d_path is None:
        d_path = parse_args()['dataset_path']
    patients = get_dirs(d_path)
    gpu = parse_args()['gpu_id']
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%d' % gpu if cuda else 'cpu')
    torch.backends.cudnn.benchmark = True

    time_str = strftime("%H:%M:%S")
    print('\n%s[%s]%s CNN registration' % (
        c['c'], time_str, c['nc']
    ))
    timepoints = map(
        lambda t: 'flair_time%d-time6_corrected_matched.nii.gz' % t,
        range(1, 6)
    )
    timepoints.append('flair_corrected.nii.gz')

    global_start = time.time()

    # Main loop
    patient_paths = map(lambda p: os.path.join(d_path, p), patients)
    brain_names = map(
        lambda p_path: os.path.join(p_path, folder, mask),
        patient_paths
    )
    masks = map(get_mask, brain_names)

    lesion_names = map(
        lambda p_path: find_file('(' + '|'.join(lesion_tags) + ')', p_path),
        patient_paths
    )
    lesions = map(
        lambda name: get_mask(name, parse_args()['dilate']),
        lesion_names
    )

    norm_cases = map(
        lambda (p, mask_i): map(
            lambda name: get_normalised_image(
                os.path.join(p, folder, name), mask_i
            ),
            timepoints
        ),
        zip(patient_paths, masks)
    )

    vhpath = parse_args()['seg_dataset_path']
    vhpatients = get_dirs(vhpath)
    vhpatient_paths = map(lambda p: os.path.join(vhpath, p), vhpatients)
    vhbrain_names = map(
        lambda p_path: os.path.join(p_path, brain_name),
        vhpatient_paths
    )
    brains = map(get_mask, vhbrain_names)
    vhlesion_names = map(
        lambda p_path: os.path.join(p_path, lesion_name),
        vhpatient_paths
    )
    vhlesions = map(get_mask, vhlesion_names)
    norm_source = map(
        lambda (p, mask_i): get_normalised_image(
            os.path.join(p, source_name), mask_i
        ),
        zip(vhpatient_paths, brains)
    )
    norm_target = map(
        lambda (p, mask_i): get_normalised_image(
            os.path.join(p, target_name), mask_i
        ),
        zip(vhpatient_paths, brains)
    )

    if verbose > 0:
        print(
            '%s[%s]%s Training CNN with all timepoints + segmentation%s' %
            (
                c['c'], strftime("%H:%M:%S"),
                c['g'], c['nc']
            )
        )

    loss_idx = parse_args()['loss_idx']
    epochs = parse_args()['epochs']
    patience = parse_args()['patience']
    lambda_v = parse_args()['lambda']
    data_smooth = parse_args()['data_smooth']
    df_smooth = parse_args()['df_smooth']
    train_smooth = parse_args()['train_smooth']
    smooth_s = '.'.join(
        filter(
            None,
            [
                'data' if data_smooth else None,
                'df' if df_smooth else None,
                'tr' if train_smooth else None,
            ]
        )
    )
    model_name = os.path.join(
        d_path,
        'mixedlong_model_%s_loss%s_l%.2fe%dp%d.mdl' % (
            smooth_s, '+'.join(map(str, loss_idx)), lambda_v, epochs, patience
        )
    )

    training_start = time.time()

    reg_net = LongitudinalNet(
        loss_idx=loss_idx,
        lambda_d=lambda_v,
        device=device,
        data_smooth=data_smooth,
        df_smooth=df_smooth,
        trainable_smooth=train_smooth
    )
    try:
        reg_net.load_model(model_name)
    except IOError:
        reg_net.fit(
            norm_source,
            norm_target,
            vhlesions,
            norm_cases,
            lesions,
            masks,
            epochs=epochs,
            patience=patience
        )

    reg_net.save_model(model_name)

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - training_start)
        )
        print(
            '%sTraining finished%s (total time %s)\n' %
            (c['r'], c['nc'], time_str)
        )

    for i, patient in enumerate(patients):
        patient_start = time.time()
        if verbose > 0:
            print(
                '%s[%s]%s Starting testing with patient %s %s(%d/%d)%s' %
                (
                    c['c'], strftime("%H:%M:%S"),
                    c['g'], patient,
                    c['c'], i + 1, len(patients), c['nc']
                )
            )
        # Load mask, source and target and expand the dimensions.
        patient_path = os.path.join(d_path, patient)
        source_name = os.path.join(
            patient_path, folder, 'flair_time1-time6_corrected_matched.nii.gz'
        )
        target_name = os.path.join(
            patient_path, folder, 'flair_corrected.nii.gz'
        )
        brain_name = os.path.join(patient_path, folder, mask)
        mask_name = find_file('(' + '|'.join(lesion_tags) + ')', patient_path)
        if verbose > 1:
            print(
                '%s-Using image %s%s%s' %
                (
                    ''.join([' '] * 11),
                    c['b'], mask_name,
                    c['nc']
                )
            )

        # Brain mask
        brain_mask = load_nii(brain_name).get_data()
        brain_bin = brain_mask > 0

        # Lesion mask
        lesion = load_nii(mask_name).get_data()
        mask_image = np.reshape(lesion, (1, 1) + lesion.shape)

        # Baseline image (testing)
        source_nii = load_nii(source_name)
        source_image = source_nii.get_data()
        source_mu = np.mean(source_image[brain_bin])
        source_sigma = np.std(source_image[brain_bin])
        source_image = np.reshape(source_image, (1, 1) + source_image.shape)
        norm_source = (source_image - source_mu) / source_sigma

        # Follow-up image (testing)
        target_image = load_nii(target_name).get_data()
        target_mu = np.mean(target_image[brain_bin])
        target_sigma = np.std(target_image[brain_bin])
        target_image = np.reshape(target_image, (1, 1) + target_image.shape)
        norm_target = (target_image - target_mu) / target_sigma

        # Test the network
        source_mov, mask_mov, df = reg_net.transform(
            norm_source, norm_target, mask_image
        )

        source_mov = source_mov[0] * source_sigma + source_mu
        source_nii.get_data()[:] = source_mov * brain_mask
        source_nii.to_filename(
            os.path.join(patient_path, 'cnn_defo.nii.gz')
        )
        mask_nii = nib.Nifti1Image(
            mask_mov[0],
            source_nii.get_qform(),
            source_nii.get_header()
        )
        mask_nii.to_filename(
            os.path.join(patient_path, 'cnn_defo_mask.nii.gz')
        )

        df_mask = np.repeat(np.expand_dims(brain_mask, -1), 3, -1)
        df_nii = nib.Nifti1Image(
            np.moveaxis(df[0], 0, -1) * df_mask,
            source_nii.get_qform(),
            source_nii.get_header()
        )
        df_nii.to_filename(
            os.path.join(patient_path, 'cnn_defo_df.nii.gz')
        )

        # Patient done
        if verbose > 0:
            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - patient_start)
            )
            print(
                '%sPatient %s finished%s (total time %s)\n' %
                (c['r'], patient, c['nc'], time_str)
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


def main():
    # initial_analysis()
    # naive_registration(verbose=2)
    # naive_registration(refine=True, verbose=2)
    cnn_registration(verbose=2)
    # deformationbased_registration(verbose=2)
    # subtraction_registration(image='m60_flair', verbose=2)


if __name__ == "__main__":
    main()
