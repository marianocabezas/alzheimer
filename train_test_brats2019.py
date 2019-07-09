from __future__ import print_function
import argparse
import os
import csv
from time import strftime
import numpy as np
from nibabel import load as load_nii
from skimage.transform import resize
from sklearn import decomposition
import torch
from models import BratsSegmentationNet, BratsSurvivalNet


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
        type=int, default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=20,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=5,
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

    # Prepare the sufix that will be added to the results for the net and images
    train_data, train_labels = get_names_from_path()

    print(
        '%s[%s] %s<BRATS 2019 pipeline testing>%s' % (
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )
    # Block center computation

    ''' <Segmentation task> '''
    # n_folds = len(tst_simage_names)
    n_folds = 5
    print(
        '%s[%s] %sStarting cross-validation (segmentation) - %d folds%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )
    for i in range(n_folds):
        ''' Training '''
        ini_p = len(train_data) * i / n_folds
        end_p = len(train_data) * (i + 1) / n_folds

        # Training data
        print(
            train_data[:ini_p, :].shape, train_data[end_p:, :].shape,
            train_data[:ini_p, :].dtype(), train_data[end_p:, :].dtype()
        )
        train_x = np.concatenate(
            train_data[:ini_p, :], train_data[end_p:, :]
        )
        train_y = np.concatenate(
            train_labels[:ini_p, :], train_labels[end_p:, :]
        )

        # Testing data
        test_x = train_data[ini_p:end_p]
        test_y = train_labels[ini_p:end_p]

        # Training itself
        print(
            '%s[%s] %sFold %s(%s%d%s%s/%d)%s' % (
                c['c'], strftime("%H:%M:%S"), c['g'],
                c['c'], c['b'], i + 1, c['nc'], c['c'], n_folds, c['nc']
            )
        )

        net = BratsSegmentationNet()

        net.fit(
            train_x, train_y,
            val_split=0.1, criterion='dsc',
            epochs=epochs, patience=patience, batch_size=batch_size
        )

        print(
            '%s[%s] %sStarting training (%ssegmentation%s)%s' % (
                c['c'], strftime("%H:%M:%S"),
                c['g'], c['b'], c['nc'] + c['g'], c['nc']
            )
        )


if __name__ == '__main__':
    main()
