"""
The main file running inside the docker (the starting point)
"""
# Import the required packages
from __future__ import print_function
from scipy import ndimage as nd
import numpy as np
import argparse
import os
import sys
import time
import re
import traceback
from itertools import product
from nibabel import load as load_nii
import SimpleITK as sitk
from time import strftime
from subprocess import call
from data_manipulation.sitk import itkn4, itkhist_match, itksubtraction, itkdemons
from data_manipulation.generate_features import get_mask_voxels


"""
Utility functions
"""


def color_codes():
    """
    Function that returns a custom dictionary with ASCII codes related to colors.
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
    Function to control the arguments of the python script when called from the command line.
    :return: Dictionary with the argument values
    """
    parser = argparse.ArgumentParser(description='Run the longitudinal MS lesion segmentation docker.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-f', '--old',
        dest='dataset_path', default='/home/mariano/DATA/Australia60m/Workstation',
        help='Parameter to store the working directory.'
    )
    group.add_argument(
        '-t', '--tools',
        dest='tools_path', default='/home/mariano/alzheimer/',
        help='Parameter to store the tools directory.'
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
    print('%s[%s]%s %s' % (c['c'], strftime("%H:%M:%S", time.localtime()), c['nc'], message))
    print(dashes)


def run_command(command, message=None, stdout=None, stderr=None):
    """
    Function to run and time a shell command using the call function from the subprocess module.
    :param command: Command that will be run. It has to comply with the call function specifications.
    :param message: Message to be printed before running the command. This is an optional parameter and by default its
     None.
    :param stdout: File where the stdout will be redirected. By default we use the system stdout.
    :param stderr: File where the stderr will be redirected. By default we use the system stderr.
    :return:
    """
    if message is not None:
        print_message(message)

    time_f(lambda: call(command), stdout=stdout, stderr=stderr)


def time_f(f, stdout=None, stderr=None):
    """
    Function to time another function.
    :param f: Function to be run. If the function has any parameters, it should be passed
     using the lambda keyword.
    :param stdout: Stdout file to write all prints to file. By default the system stdout will be used.
    :param stderr: Stderr file to write all prints to file. By default the system stderr will be used.
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

    print(time.strftime('Time elapsed = %H hours %M minutes %S seconds', time.gmtime(time.time() - start_t)))
    return ret


def get_dirs(path):
    """
    Function to get the folder name of the patients given a path.
    :param path: Folder where the patients should be located.
    :return: List of patient names.
    """
    # All patients (full path)
    patient_paths = sorted(filter(lambda d: os.path.isdir(os.path.join(path, d)), os.listdir(path)))
    # Patients used during training
    return patient_paths


"""
Data related functions
"""


def remove_small_regions(img_vol, min_size=3):
    """
    Function that removes blobs with a size smaller than a mininum from a mask volume.
    :param img_vol: Mask volume. It should be a numpy array of type bool.
    :param min_size: Mininum size for the blobs.
    :return: New mask without the small blobs.
    """
    blobs, _ = nd.measurements.label(img_vol, nd.morphology.generate_binary_structure(3, 3))
    labels = filter(bool, np.unique(blobs))
    areas = [np.count_nonzero(np.equal(blobs, l)) for l in labels]
    nu_labels = [l for l, a in zip(labels, areas) if a > min_size]
    nu_mask = reduce(lambda x, y: np.logical_or(x, y),
                     [np.equal(blobs, l) for l in nu_labels]) if nu_labels else np.zeros_like(img_vol)
    return nu_mask


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
        M. Cabezas, J.F. Corral, A. Oliver, Y. Diez, M. Tintore, C. Auger, X. Montalban,
         X. Llado, D. Pareto, A. Rovira.
        'Automatic multiple sclerosis lesion detection in brain MRI by FLAIR thresholding'
        In American Journal of Neuroradiology, Volume 37(10), 2016, Pages 1816-1823
        http://dx.doi.org/10.3174/ajnr.A4829
    :param d_path: Path where the whole database is stored. If not specified, it will be read from the
     config file.
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
        print('%s[%s]%s Starting preprocessing with patient %s %s(%d/%d)%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], patient, c['c'], i + 1, len(patients), c['nc']
        ))
        patient_path = os.path.join(d_path, patient)

        timepoints = get_dirs(os.path.join(d_path, patient))

        if verbose > 0:
            print('/-------------------------------\\')
            print('|         Preprocessing         |')
            print('\\-------------------------------/')

        # First we skull strip all the timepoints using ROBEX. We could probably use another method that
        # has better integration with our code...
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
                    lambda x: not os.path.isdir(x) and re.search(r'(\w|\W)*(t1|T1)(\w|\W)', x),
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
            original_files = map(lambda t: find_file('(' + '|'.join(t) + ')', full_folder), tags)
            map(
                lambda (b, im): itkn4(b, full_folder, im, max_iters=200, levels=4, verbose=verbose),
                filter(lambda (b, _): b is not None, zip(original_files, images))
            )

        # Followup setup
        followup = timepoints[-1]
        followup_path = os.path.join(patient_path, followup)

        bm_tag = 'brainmask.nii.gz'
        brains = map(lambda t: load_nii(os.path.join(patient_path, t, bm_tag)).get_data(), timepoints)
        brain = reduce(np.logical_or, brains)
        brain_nii = load_nii(os.path.join(patient_path, timepoints[-1], 'brainmask.nii.gz'))
        brain_nii.get_data()[:] = brain
        brain_nii.to_filename(os.path.join(followup_path, 'union_brainmask.nii.gz'))

        followup_files = map(lambda im: find_file(im + '_corrected.nii.gz', followup_path), images)
        for baseline in timepoints[:-1]:
            image_tag = baseline + '-' + followup
            baseline_path = os.path.join(patient_path, baseline)
            baseline_files = map(lambda im: find_file(im + '_corrected.nii.gz', baseline_path), images)

            # Now it's time to histogram match everything to the last timepoint.
            """Histogram matching"""
            if verbose > 0:
                print('\t-------------------------------')
                print('\t       Histogram matching      ')
                print('\t-------------------------------')
            map(
                lambda (f, b, im): itkhist_match(f, b, followup_path, im + '_' + image_tag, verbose=verbose),
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
            baseline_files = map(lambda im: find_file(im + '_' + image_tag + hm_tag, followup_path), images)

            map(
                lambda (f, b, im): itksubtraction(f, b, followup_path, im + '_' + image_tag, verbose=verbose),
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
            mask = sitk.ReadImage(os.path.join(followup_path, 'union_brainmask.nii.gz'))
            map(
                lambda (f, b, im): itkdemons(f, b, mask, followup_path, im + '_' + image_tag, verbose=verbose),
                filter(
                    lambda (f, b, im): f is not None and b is not None,
                    zip(baseline_files, followup_files, images)
                )
            )

        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - patient_start)
        )
        print('%sPatient %s finished%s (total time %s)\n' % (c['r'], patient, c['nc'], time_str))

    time_str = time.strftime('%H hours %M minutes %S seconds', time.gmtime(time.time() - global_start))
    print_message('%sAll patients finished %s(total time %s)%s' % (c['r'],  c['b'], time_str, c['nc']))


def naive_registration(
        d_path=None,
        image='flair_corrected.nii.gz',
        lesion_tags=list(['_bin', 'lesion', 'lesionMask']),
        fixed_folder='time6',
        moving_folder='time1',
        width=5,
        dim=3,
        verbose=1,
):
    """
    Function that applies a lesion by lesion naive registration. The idea is that after 3 years,
    the atrophy of the MS patient is noticeable and that causes a movement on the chronic MS lesions
    (due to the ventricle expansion). The assumption is that we can use a naive registration approach
    that open a sliding window around the baseline mask and tries to find the best match in tems
    of intensity similarity on the follow-up image.
    :param d_path: Path where the whole database is stored. If not specified, it will be read from the
     config file.
    :param image: Image name that will be used for matching.
    :param lesion_tags: Tags that may be contained on the lesion mask filename.
    :param fixed_folder: Folder that contains the fixed image.
    :param moving_folder: Folder that contains the moving image.
    :param width: Width of the sliding window.
    :param dim: Number of dimensions of the fixed and moving images.
    :param verbose: Verbosity level.
    :return: None.
    """

    c = color_codes()

    # Init
    if d_path is None:
        d_path = parse_args()['dataset_path']
    mov_pair = [(-width, -1), (width, 1)]
    movs = np.unique(np.concatenate(map(lambda (e, s): list(product(range(0, e, s), repeat=dim)), mov_pair)), axis=0)
    fixed = load_nii(os.path.join(d_path, fixed_folder, image)).get_data()
    moving = load_nii(os.path.join(d_path, moving_folder, image)).get_data()

    mask_name = find_file(lesion_tags, d_path)
    if mask_name is not None:
        mask = load_nii(mask_name)
        labels, nlabels = nd.label(mask)
        match_scores = map(
            lambda vox: map(
                lambda mov: (np.correlate(fixed[vox], moving[vox + movs]), vox + movs),
                movs
            ),
            map(lambda l: np.array(get_mask_voxels(labels == l)), range(1, nlabels + 1))
        )


def main():
    initial_analysis()


if __name__ == "__main__":
    main()
