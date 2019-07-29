from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
from models import BratsSegmentationNet, BratsSegmentationHybridNet
from utils import color_codes, get_dirs, find_file, run_command, print_message
from utils import get_mask, get_normalised_image
from nibabel import save as save_nii
from nibabel import load as load_nii
from data_manipulation.metrics import dsc_seg


def color_codes():
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


def count_params(pmodel):
    return sum(p.numel() for p in pmodel.parameters() if p.requires_grad)


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')

    # Mode selector
    parser.add_argument(
        '-d', '--file-directory',
        dest='loo_dir', default='/home/mariano/DATA/Brats19TrainingData',
        help='Option to use leave-one-out. The second parameter is the '
             'folder with all the patients.'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '-f', '--filters',
        dest='filters',
        type=int,  default=20,
        help='Number of starting filters (30 for NNNet)'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=2,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '-s', '--sampling-rate',
        dest='sampling_rate',
        type=int,  default=1,
        help='Number of epochs'
    )
    parser.add_argument(
        '-y', '--use-hybrid',
        dest='hybrid',
        default=False, action='store_true',
        help='Whether to use a hybrid net. Default is False'
    )
    parser.add_argument(
        '-b', '--blocks',
        dest='blocks',
        type=int, default=3,
        help='Number of blocks (or depth)'
    )

    options = vars(parser.parse_args())

    return options


def main():
    # Init
    c = color_codes()
    options = parse_inputs()
    epochs = options['epochs']
    patience = options['patience']
    sampling_rate = options['sampling_rate']
    filters = options['filters']
    hybrid = options['hybrid']
    depth = options['blocks']
    mode_s = '-hybrid' if hybrid else ''
    images = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']

    # Prepare the sufix that will be added to the results for the net and images

    print(
        '%s[%s] %s<BRATS 2019 pipeline%s testing>%s' % (
            c['c'], strftime("%H:%M:%S"), c['y'], mode_s, c['nc']
        )
    )
    d_path = options['loo_dir']
    patients = get_dirs(d_path)
    np.random.seed(42)
    patients = np.random.permutation(patients).tolist()

    ''' <Segmentation task> '''
    n_folds = 5
    print(
        '%s[%s] %sStarting cross-validation (segmentation) - %d folds%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )

    depth_s = '-f%d' % depth
    filters_s = '-f%d' % filters
    sampling_rate_s = '-sr%d' % sampling_rate if sampling_rate > 1 else ''
    net_name = 'brats2019-nnunet-%s%s%s%s' % (
        mode_s, sampling_rate_s, filters_s, depth_s
    )

    for i in range(n_folds):
        print(
            '%s[%s] %sFold %s(%s%d%s%s/%d)%s' % (
                c['c'], strftime("%H:%M:%S"), c['g'],
                c['c'], c['b'], i + 1, c['nc'], c['c'], n_folds, c['nc']
            )
        )
        ''' Training '''
        ini_p = len(patients) * i / n_folds
        end_p = len(patients) * (i + 1) / n_folds

        # Training data
        train_patients = patients[:ini_p] + patients[end_p:]
        brain_names = map(
            lambda p: os.path.join(
                d_path, p, p + '_t1.nii.gz'
            ),
            train_patients
        )
        brains = map(get_mask, brain_names)
        lesion_names = map(
            lambda p: os.path.join(d_path, p, p + '_seg.nii.gz'),
            train_patients
        )
        train_y = map(get_mask, lesion_names)
        for yi in train_y:
            yi[yi == 4] = 3
        train_x = map(
            lambda (p, mask_i): np.stack(
                map(
                    lambda im: get_normalised_image(
                        os.path.join(d_path, p, p + im),
                        mask_i, masked=True
                    ),
                    images
                ),
                axis=0
            ),
            zip(train_patients, brains)
        )

        # Testing data
        test_patients = patients[ini_p:end_p]
        patient_paths = map(lambda p: os.path.join(d_path, p), test_patients)
        brain_names = map(
            lambda p: os.path.join(
                d_path, p, p + '_t1.nii.gz'
            ),
            test_patients
        )
        brains_test = map(get_mask, brain_names)
        test_x = map(
            lambda (p, mask_i): np.stack(
                map(
                    lambda im: get_normalised_image(
                        os.path.join(d_path, p, p + im),
                        mask_i,
                    ),
                    images
                ),
                axis=0
            ),
            zip(test_patients, brains_test)
        )

        print(
            'Training / testing samples = %d / %d' % (
                len(train_x), len(test_x)
            )
        )

        # Training itself
        model_name = '%s_f%d.mdl' % (net_name, i)
        if hybrid:
            net = BratsSegmentationHybridNet(filters=filters, depth=depth)
        else:
            net = BratsSegmentationNet(depth=depth)
        try:
            net.load_model(os.path.join(d_path, model_name))
        except IOError:
            n_params = sum(
                p.numel() for p in net.parameters() if p.requires_grad
            )
            print(
                '%sStarting training with a unet%s (%d parameters)' %
                (c['c'], c['nc'], n_params)
            )

            # Image wise training
            if hybrid:
                net.fit(
                    train_x, train_y, rois=brains,
                    val_split=0.1, epochs=epochs, patience=patience,
                    num_workers=16,
                )
            else:
                net.fit(
                    train_x, train_y, rois=brains,
                    val_split=0.1, epochs=epochs, patience=patience,
                    batch_size=1, num_workers=16,
                    sample_rate=sampling_rate
                )

            net.save_model(os.path.join(d_path, model_name))

        # Testing data
        pred_y = net.segment(test_x)

        for (path_i, p_i, pred_i) in zip(patient_paths, test_patients, pred_y):
            seg_i = np.argmax(pred_i, axis=0)
            seg_i[seg_i == 3] = 4

            niiname = os.path.join(path_i, p_i + '_seg.nii.gz')
            nii = load_nii(niiname)
            seg = nii.get_data()

            dsc = map(
                lambda label: dsc_seg(seg == label, seg_i == label), [1, 2, 4]
            )

            nii.get_data()[:] = seg_i
            save_nii(nii, os.path.join(path_i, p_i + '.nii.gz'))

            print(
                'Patient %s: %s' % (p_i, ' / '.join(map(str, dsc)))
            )

        # uncert_y, pred_y = net.uncertainty(test_x, steps=20)
        # for (path_i, p_i, pred_i, uncert_i) in zip(
        #         patient_paths, test_patients, pred_y, uncert_y
        # ):
        #     seg_i = np.argmax(pred_i, axis=0)
        #     seg_i[seg_i == 3] = 4
        #
        #     niiname = os.path.join(path_i, p_i + '_seg.nii.gz')
        #     nii = load_nii(niiname)
        #     seg = nii.get_data()
        #
        #     dsc = map(
        #         lambda label: dsc_seg(seg == label, seg_i == label), [1, 2, 4]
        #     )
        #
        #     nii.get_data()[:] = seg_i
        #     save_nii(nii, os.path.join(path_i, p_i + '_uncert-seg.nii.gz'))
        #
        #     niiname = os.path.join(path_i, p_i + '_flair.nii.gz')
        #     nii = load_nii(niiname)
        #     nii.get_data()[:] = uncert_i
        #     save_nii(nii, os.path.join(path_i, p_i + '_uncert.nii.gz'))
        #
        #     print(
        #         'Patient %s: %s' % (p_i, ' / '.join(map(str, dsc)))
        #     )

        print(
            '%s[%s] %sStarting training (%ssegmentation%s)%s' % (
                c['c'], strftime("%H:%M:%S"),
                c['g'], c['b'], c['nc'] + c['g'], c['nc']
            )
        )


if __name__ == '__main__':
    main()
