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
from models import NewLesionsNet, NewLesionsUNet
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
        default='/home/mariano/DATA/Longitudinal',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=5,
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


def new_lesions(
        d_path=None,
        images=['pd', 't1', 't2', 'flair'],
        brain_name='union_brainmask.nii.gz',
        lesion_name='longitudinalMask.nii.gz',
        verbose=1,
):
    """
        Function that applies a CNN-based registration approach. The goal of
        this network is to find the atrophy deformation, and how it affects the
        lesion mask, manually segmented on the baseline image.
        :param d_path: Path where the whole database is stored. If not
         specified,it will be read from the command parameters.
        :param images: Names of the images.
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

    source_names = map(lambda im: '%s_moved.nii.gz' % im, images)
    target_names = map(lambda im: '%s_processed.nii.gz' % im, images)

    gpu = parse_args()['gpu_id']
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%d' % gpu if cuda else 'cpu')

    torch.backends.cudnn.benchmark = True

    time_str = strftime("%H:%M:%S")
    print('\n%s[%s]%s CNN registration' % (
        c['c'], time_str, c['nc']
    ))

    epochs = parse_args()['epochs']
    patience = parse_args()['patience']
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
            lambda p_path: os.path.join(
                p_path, 'time2', 'segmentation', brain_name
            ),
            patient_paths
        )
        brains = map(get_mask, brain_names)
        lesion_names = map(
            lambda p_path: os.path.join(p_path, 'time2', lesion_name),
            patient_paths
        )
        lesions = map(get_mask, lesion_names)
        norm_source = map(
            lambda (p, mask_i): np.stack(
                map(
                    lambda im: get_normalised_image(
                        os.path.join(p, 'time2', 'preprocessed', im),
                        mask_i, masked=True
                    ),
                    source_names
                ),
                axis=0
            ),
            zip(patient_paths, brains)
        )
        norm_target = map(
            lambda (p, mask_i): np.stack(
                map(
                    lambda im: get_normalised_image(
                        os.path.join(p, 'time2', 'preprocessed', im),
                        mask_i, masked=True
                    ),
                    target_names
                ),
                axis=0
            ),
            zip(patient_paths, brains)
        )

        training_start = time.time()

        net_name = 'newlesions-unet'
        model_name = '%s_model_%s_e%dp%d.mdl' % (
            net_name, smooth_s + '_' if smooth_s else '', epochs, patience
        )

        seg_net = NewLesionsUNet(n_images=len(images))
        try:
            seg_net.load_model(os.path.join(d_path, patient, model_name))
        except IOError:
            if verbose > 0:
                n_params = sum(
                    p.numel() for p in seg_net.parameters() if p.requires_grad
                )
                print(
                    '%sStarting training wit a unet%s (%d parameters)' %
                    (c['c'], c['nc'], n_params)
                )
            seg_net.fit(
                norm_source,
                norm_target,
                lesions,
                num_workers=16,
                val_split=0.1,
                epochs=epochs,
                patience=patience
            )

        seg_net.save_model(os.path.join(d_path, patient, model_name))

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
        brain = get_mask(os.path.join(
            patient_path, 'time2', 'segmentation', brain_name
        ), dtype=bool)

        # Lesion mask
        gt = get_mask(os.path.join(patient_path, 'time2', lesion_name))

        # Baseline image (testing)
        source_niis = map(
            lambda name: load_nii(
                os.path.join(patient_path, 'time2', 'preprocessed', name)
            ),
            source_names
        )

        norm_source_tst = np.stack(
            map(
                lambda im: get_normalised_image(
                    os.path.join(patient_path, 'time2', 'preprocessed', im),
                    brain, masked=True
                ),
                source_names
            ),
            axis=0
        )
        norm_target_tst = np.stack(
            map(
                lambda im: get_normalised_image(
                    os.path.join(patient_path, 'time2', 'preprocessed', im),
                    brain, masked=True
                ),
                target_names
            ),
            axis=0
        )

        sufix = 'seg_%s%s.e%dp%d' % (
            smooth_s + '_' if smooth_s else '', net_name, epochs, patience
        )

        seg = seg_net.new_lesions(
            np.expand_dims(norm_source_tst, axis=0),
            np.expand_dims(norm_target_tst, axis=0)
        )

        lesion_unet = seg[0][1] > 0.5

        for j, s_i in enumerate(seg[0]):
            mask_nii = nib.Nifti1Image(
                s_i,
                source_niis[0].get_qform(),
                source_niis[0].get_header()
            )
            mask_nii.to_filename(
                os.path.join(
                    patient_path, 'lesion_mask_%s_s%d.nii.gz' % (sufix, j)
                )
            )

        net_name = 'newlesions-vm'
        model_name = '%s_model_%s_e%dp%d.mdl' % (
            net_name, smooth_s + '_' if smooth_s else '', epochs, patience
        )
        reg_net = NewLesionsNet(
            device=device,
            data_smooth=data_smooth,
            df_smooth=df_smooth,
            trainable_smooth=train_smooth,
            n_images=len(images)
        )
        try:
            reg_net.load_model(os.path.join(d_path, patient, model_name))
        except IOError:
            if verbose > 0:
                n_params = sum(
                    p.numel() for p in seg_net.parameters() if p.requires_grad
                )
                print(
                    '%sStarting training with VoxelMorph%s (%d parameters)' %
                    (c['c'], c['nc'], n_params)
                )
            reg_net.fit(
                norm_source,
                norm_target,
                lesions,
                num_workers=16,
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

        source_images = map(lambda nii: nii.get_data(), source_niis)
        source_mus = map(lambda im: np.mean(im[brain]), source_images)
        source_sigmas = map(lambda im: np.std(im[brain]), source_images)

        sufix = 'vm_%s%s.e%dp%d' % (
            smooth_s + '_' if smooth_s else '', net_name, epochs, patience
        )

        # Test the network
        seg, source_mov, df = reg_net.new_lesions(
            np.expand_dims(norm_source_tst, axis=0),
            np.expand_dims(norm_target_tst, axis=0)
        )

        lesion_vm = seg[0][1] > 0.5

        for j, (nii, mov, mu, sigma) in enumerate(
                zip(source_niis, source_mov[0], source_mus, source_sigmas)
        ):
            source_mov = mov * sigma + mu
            nii.get_data()[:] = source_mov * brain
            nii.to_filename(
                os.path.join(patient_path, 'moved_%s_im%d.nii.gz' % (sufix, j))
            )
        for j, s_i in enumerate(seg[0]):
            mask_nii = nib.Nifti1Image(
                s_i,
                source_niis[0].get_qform(),
                source_niis[0].get_header()
            )
            mask_nii.to_filename(
                os.path.join(
                    patient_path, 'lesion_mask_%s_s%d.nii.gz' % (sufix, j)
                )
            )

        df_mask = np.repeat(np.expand_dims(brain, -1), 3, -1)
        df_nii = nib.Nifti1Image(
            np.moveaxis(df[0], 0, -1) * df_mask,
            source_niis[0].get_qform(),
            source_niis[0].get_header()
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

            print('Netname TPFV FPFV DSCV TPFL FPFL DSCL TPL GTL Vox GTV')

            tpfv = tp_fraction_seg(gt, lesion_unet)
            fpfv = fp_fraction_seg(gt, lesion_unet)
            dscv = dsc_seg(gt, lesion_unet)
            tpfl = tp_fraction_det(gt, lesion_unet)
            fpfl = fp_fraction_det(gt, lesion_unet)
            dscl = dsc_det(gt, lesion_unet)
            tp = true_positive_det(lesion_unet, gt)
            gt_d = num_regions(gt)
            lesion_s = num_voxels(lesion_unet)
            gt_s = num_voxels(gt)
            measures = (tpfv, fpfv, dscv, tpfl, fpfl, dscl, tp, gt_d, lesion_s, gt_s)

            print(
                'Unet    %.2f %.2f %.2f %.2f %.2f %.2f '
                '%03d %03d %03d %03d' % measures
            )

            tpfv = tp_fraction_seg(gt, lesion_vm)
            fpfv = fp_fraction_seg(gt, lesion_vm)
            dscv = dsc_seg(gt, lesion_vm)
            tpfl = tp_fraction_det(gt, lesion_vm)
            fpfl = fp_fraction_det(gt, lesion_vm)
            dscl = dsc_det(gt, lesion_vm)
            tp = true_positive_det(lesion_vm, gt)
            gt_d = num_regions(gt)
            lesion_s = num_voxels(lesion_vm)
            gt_s = num_voxels(gt)
            measures = (tpfv, fpfv, dscv, tpfl, fpfl, dscl, tp, gt_d, lesion_s, gt_s)

            print(
                'VMnet   %.2f %.2f %.2f %.2f %.2f %.2f '
                '%d %d %d %d' % measures
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
    new_lesions(verbose=2)


if __name__ == "__main__":
    main()
