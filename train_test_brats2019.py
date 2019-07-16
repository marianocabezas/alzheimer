from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
from models import BratsSegmentationNet
from utils import color_codes, get_dirs, find_file, run_command, print_message
from utils import get_mask, get_normalised_image


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
        '-f', '--file-directory',
        dest='loo_dir', default='/home/mariano/DATA/Brats19TrainingData',
        help='Option to use leave-one-out. The second parameter is the '
             'folder with all the patients.'
    )
    parser.add_argument(
        '-b', '--batch_size',
        dest='batch_size',
        type=int, default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=50,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=5,
        help='Patience for early stopping'
    )

    options = vars(parser.parse_args())

    return options


def main():
    # Init
    c = color_codes()
    options = parse_inputs()
    batch_size = options['batch_size']
    epochs = options['epochs']
    patience = options['patience']
    sampling_rate = options['sampling_rate']
    images = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']

    # Prepare the sufix that will be added to the results for the net and images

    print(
        '%s[%s] %s<BRATS 2019 pipeline testing>%s' % (
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )
    d_path = options['loo_dir']
    patients = get_dirs(d_path)

    ''' <Segmentation task> '''
    n_folds = 5
    print(
        '%s[%s] %sStarting cross-validation (segmentation) - %d folds%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )

    sampling_rate_s = '-sr%d' % sampling_rate if sampling_rate > 1 else ''
    net_name = 'brats2019-nnunet_grouped%s' % sampling_rate_s

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
        patient_paths = map(lambda p: os.path.join(d_path, p), train_patients)
        brain_names = map(
            lambda (p_path, p): os.path.join(
                p_path, p + '_t1.nii.gz'
            ),
            zip(patient_paths, train_patients)
        )
        brains = map(get_mask, brain_names)
        lesion_names = map(
            lambda (p_path, p): os.path.join(p_path, p + '_seg.nii.gz'),
            zip(patient_paths, train_patients)
        )
        train_y = map(get_mask, lesion_names)
        for yi in train_y:
            yi[yi == 4] = 3
        train_x = map(
            lambda (p_path, p, mask_i): np.stack(
                map(
                    lambda im: get_normalised_image(
                        os.path.join(p_path, p + im),
                        mask_i,
                    ),
                    images
                ),
                axis=0
            ),
            zip(patient_paths, train_patients, brains)
        )

        # Testing data
        test_patients = patients[ini_p:end_p]
        patient_paths = map(lambda p: os.path.join(d_path, p), test_patients)
        lesion_names = map(
            lambda (p_path, p): os.path.join(p_path, p + '_seg.nii.gz'),
            zip(patient_paths, test_patients)
        )
        test_y = map(get_mask, lesion_names)
        test_x = map(
            lambda (p_path, p, mask_i): np.stack(
                map(
                    lambda im: get_normalised_image(
                        os.path.join(p_path, p + im),
                        mask_i,
                    ),
                    images
                ),
                axis=0
            ),
            zip(patient_paths, test_patients, brains)
        )

        print(
            'Training / testing samples = %d / %d' % (
                len(train_x), len(test_x)
            )
        )

        # Training itself
        model_name = '%s_f%d.mdl' % (net_name, i)
        net = BratsSegmentationNet()
        try:
            net.load_model(os.path.join(d_path, model_name))
        except IOError:
            n_params = sum(
                p.numel() for p in net.parameters() if p.requires_grad
            )
            print(
                '%sStarting training wit a unet%s (%d parameters)' %
                (c['c'], c['nc'], n_params)
            )

            net.fit(
                train_x, train_y,
                val_split=0.1, epochs=epochs, patience=patience,
                batch_size=batch_size, num_workers=16,
                sample_rate=sampling_rate
            )

            net.save_model(os.path.join(d_path, model_name))

        # Testing data

        print(
            '%s[%s] %sStarting training (%ssegmentation%s)%s' % (
                c['c'], strftime("%H:%M:%S"),
                c['g'], c['b'], c['nc'] + c['g'], c['nc']
            )
        )


if __name__ == '__main__':
    main()
