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
        type=int,  default=5,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=1,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '--no-flair',
        action='store_false', dest='use_flair', default=True,
        help='Don''t use FLAIR'
    )
    parser.add_argument(
        '--flair',
        action='store', dest='flair', default='_flair.nii.gz',
        help='FLAIR sufix name'
    )
    parser.add_argument(
        '--no-t1',
        action='store_false', dest='use_t1', default=True,
        help='Don''t use T1'
    )
    parser.add_argument(
        '--t1',
        action='store', dest='t1', default='_t1.nii.gz',
        help='T1 sufix name'
    )
    parser.add_argument(
        '--no-t1ce',
        action='store_false', dest='use_t1ce', default=True,
        help='Don''t use T1 with contrast'
    )
    parser.add_argument(
        '--t1ce',
        action='store', dest='t1ce', default='_t1ce.nii.gz',
        help='T1 with contrast enchancement sufix name'
    )
    parser.add_argument(
        '--no-t2',
        action='store_false', dest='use_t2', default=True,
        help='Don''t use T2'
    )
    parser.add_argument(
        '--t2',
        action='store', dest='t2', default='_t2.nii.gz',
        help='T2 sufix name'
    )
    parser.add_argument(
        '--labels',
        action='store', dest='labels', default='_seg.nii.gz',
        help='Labels image sufix'
    )

    options = vars(parser.parse_args())

    return options


def get_names(sufix, path):
    options = parse_inputs()
    if path is None:
        path = options['loo_dir']

    directories = filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])
    patients = sorted(directories)

    return map(lambda p: os.path.join(p, p.split('/')[-1] + sufix), patients)


def get_names_from_path(path=None):
    options = parse_inputs()
    # Prepare the names
    flair_names = get_names(options['flair'], path) if options['use_flair'] else None
    t2_names = get_names(options['t2'], path) if options['use_t2'] else None
    t1_names = get_names(options['t1'], path) if options['use_t1'] else None
    t1ce_names = get_names(options['t1ce'], path) if options['use_t1ce'] else None

    label_names = np.array(get_names(options['labels'], path))
    image_names = np.stack(filter(None, [flair_names, t2_names, t1_names, t1ce_names]), axis=1)

    return image_names, label_names


def main():
    # Init
    c = color_codes()
    options = parse_inputs()
    batch_size = options['batch_size']
    epochs = options['epochs']
    patience = options['patience']
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
    for i in range(n_folds):
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

        # Training itself
        print(
            '%s[%s] %sFold %s(%s%d%s%s/%d)%s' % (
                c['c'], strftime("%H:%M:%S"), c['g'],
                c['c'], c['b'], i + 1, c['nc'], c['c'], n_folds, c['nc']
            )
        )

        net = BratsSegmentationNet()

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
            batch_size=batch_size, sample_rate=10, num_workers=16
        )

        # Testing data

        print(
            '%s[%s] %sStarting training (%ssegmentation%s)%s' % (
                c['c'], strftime("%H:%M:%S"),
                c['g'], c['b'], c['nc'] + c['g'], c['nc']
            )
        )


if __name__ == '__main__':
    main()
