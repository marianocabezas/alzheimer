"""
The main file running inside the docker (the starting point)
"""
# Import the required packages
from __future__ import print_function
import numpy as np
import torch
import argparse
import os
import time
import re
from nibabel import load as load_nii
import nibabel as nib
from time import strftime
from data_manipulation.sitk import itkn4, itkhist_match
from data_manipulation.metrics import dsc_det, tp_fraction_det, fp_fraction_det
from data_manipulation.metrics import dsc_seg, tp_fraction_seg, fp_fraction_seg
from data_manipulation.metrics import true_positive_det, num_regions, num_voxels
from models import NewLesionsNet
from utils import color_codes, get_dirs, find_file, run_command, print_message
from utils import get_mask, get_normalised_image


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
        '-d', '--dataset-path',
        dest='dataset_path',
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
        nargs='+', type=int, default=[2],
        help='List of loss indices. '
             '0: Global subtraction gradient\n'
             '1: Global cross-correlation\n'
             '2: Global mean squared error\n'
             '3: Deformation regularization\n'
             '4: Modulo maximisation\n'
             '5: Global mutual information\n'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=25,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=10,
        help='Patience for early stopping'
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


def new_lesions(
        d_path=None,
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
    net_name = 'newlesions'
    model_name = '%s_model_%s_loss%s_l%.2fe%dp%d.mdl' % (
        net_name,
        smooth_s + '_' if smooth_s else '',
        '+'.join(map(str, loss_idx)),
        lambda_v, epochs, patience
    )

    global_start = time.time()

    # Main loop
    for i, patient in enumerate(patients):
        if verbose > 0:
            print(
                '%s[%s]%s Starting training for patient %s %s(%d/%d)%s' %
                (
                    c['c'], strftime("%H:%M:%S"),
                    c['g'], patient,
                    c['c'], i + 1, len(patients), c['nc']
                )
            )

        train_patients = patients[:i] + patients[i + 1:]
        patient_paths = map(lambda p: os.path.join(d_path, p), train_patients)
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
                os.path.join(p, source_name), mask_i, masked=True
            ),
            zip(patient_paths, brains)
        )
        norm_target = map(
            lambda (p, mask_i): get_normalised_image(
                os.path.join(p, target_name), mask_i, masked=True
            ),
            zip(patient_paths, brains)
        )

        training_start = time.time()

        reg_net = NewLesionsNet(
            loss_idx=loss_idx,
            lambda_d=lambda_v,
            device=device,
            data_smooth=data_smooth,
            df_smooth=df_smooth,
            trainable_smooth=train_smooth,
        )
        try:
            reg_net.load_model(os.path.join(d_path, patient, model_name))
        except IOError:
            reg_net.fit(
                norm_source,
                norm_target,
                lesions,
                brains,
                overlap=24,
                val_split=0.1,
                epochs=epochs,
                patience=patience
            )

        reg_net.save_model(os.path.join(d_path, patient, model_name))

        if verbose > 0:
            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - training_start)
            )
            print(
                '%sTraining finished%s (total time %s)\n' %
                (c['r'], c['nc'], time_str)
            )

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

        # Brain mask
        brain = get_mask(os.path.join(patient_path, brain_name), dtype=bool)

        # Lesion mask
        gt = get_mask(os.path.join(patient_path, lesion_name))

        # Baseline image (testing)
        source_nii = load_nii(os.path.join(patient_path, source_name))
        source_image = source_nii.get_data()
        source_mu = np.mean(source_image[brain])
        source_sigma = np.std(source_image[brain])
        norm_source = get_normalised_image(
            os.path.join(patient_path, source_name), brain
        )

        # Follow-up image (testing)
        norm_target = get_normalised_image(
            os.path.join(patient_path, source_name), brain
        )

        sufix = '%s%sloss%s_l%.2fe%dp%d' % (
            smooth_s + '_' if smooth_s else '', net_name,
            '+'.join(map(str, loss_idx)), lambda_v, epochs, patience
        )

        # Test the network
        seg, source_mov, df = reg_net.new_lesions(
            np.reshape(norm_source, (1, 1) + norm_source.shape),
            np.reshape(norm_target, (1, 1) + norm_target.shape)
        )

        lesion = seg[0][1] > 0.5

        source_mov = source_mov[0] * source_sigma + source_mu
        source_nii.get_data()[:] = source_mov * brain
        source_nii.to_filename(
            os.path.join(patient_path, 'moved_%s.nii.gz' % sufix)
        )
        mask_nii = nib.Nifti1Image(
            seg[0][1],
            source_nii.get_qform(),
            source_nii.get_header()
        )
        mask_nii.to_filename(
            os.path.join(patient_path, 'lesion_mask_%s.nii.gz' % sufix)
        )

        df_mask = np.repeat(np.expand_dims(brain, -1), 3, -1)
        df_nii = nib.Nifti1Image(
            np.moveaxis(df[0], 0, -1) * df_mask,
            source_nii.get_qform(),
            source_nii.get_header()
        )
        df_nii.to_filename(
            os.path.join(patient_path, 'deformation _%s.nii.gz' % sufix)
        )

        # Patient done
        if verbose > 0:
            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - training_start)
            )
            print(
                '%sPatient %s finished%s (total time %s)\n' %
                (c['r'], patient, c['nc'], time_str)
            )
            tpfv = tp_fraction_seg(gt, lesion)
            fpfv = fp_fraction_seg(gt, lesion)
            dscv = dsc_seg(gt, lesion)
            tpfl = tp_fraction_det(gt, lesion)
            fpfl = fp_fraction_det(gt, lesion)
            dscl = dsc_det(gt, lesion)
            tp = true_positive_det(lesion, gt)
            gt_d = num_regions(gt)
            lesion_s = num_voxels(lesion)
            gt_s = num_voxels(gt)
            measures = (tpfv, fpfv, dscv, tpfl, fpfl, dscl, tp, gt_d, lesion_s, gt_s)

            print('TPFV FPFV DSCV TPFL FPFL DSCL TPL GTL Voxels GTV')
            print('%f %f %f %f %f %f %d %d %d %d' % measures)

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
    new_lesions(verbose=2)


if __name__ == "__main__":
    main()
