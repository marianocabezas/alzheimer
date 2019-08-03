from __future__ import print_function
import argparse
import os
import csv
from time import strftime
import numpy as np
from models import BratsSegmentationNet, BratsSurvivalNet
from datasets import BoundarySegmentationCroppingDataset
from datasets import BBImageDataset
from utils import color_codes, get_dirs
from utils import get_mask, get_normalised_image
from nibabel import save as save_nii
from nibabel import load as load_nii
from data_manipulation.metrics import dsc_seg
from torch.utils.data import DataLoader


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
    parser.add_argument(
        '-B', '--batch-size',
        dest='batch_size',
        type=int, default=1,
        help='Number of blocks (or depth)'
    )
    parser.add_argument(
        '-t', '--patch-size',
        dest='patch_size',
        type=int, default=None,
        help='Patch size'
    )

    options = vars(parser.parse_args())

    return options


def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

    # s = str(s)
    # if s[0] in ('-', '+'):
    #     return s[1:].isdigit()
    # return s.isdigit()


def get_survival_data():
    # Init
    options = parse_inputs()
    path = options['loo_dir']

    with open(os.path.join(path, 'survival_data.csv')) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        names = csvreader.next()
        survivaldict = {
            p[0]: {
                field: value
                for field, value in zip(names[1:], p[1:])
            }
            for p in csvreader
        }

    final_dict = dict(
        filter(
            lambda (k, v): v['ResectionStatus'] == 'GTR' and isint(v['Survival']),
            survivaldict.items()
        )
    )

    return final_dict


def train_test_seg(net_name, n_folds):
    # Init
    c = color_codes()
    options = parse_inputs()
    epochs = options['epochs']
    patch_size = options['patch_size']
    batch_size = options['batch_size']
    patience = options['patience']
    depth = options['blocks']
    images = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']

    d_path = options['loo_dir']
    patients = get_dirs(d_path)
    np.random.seed(42)
    patients = np.random.permutation(patients).tolist()

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

            # Data split (using numpy) for train and validation.
            # We also compute the number of batches for both training and
            # validation according to the batch size.
            n_samples = len(train_x)

            val_split = 0.1

            n_t_samples = int(n_samples * (1 - val_split))
            n_v_samples = n_samples - n_t_samples

            print(
                'Training / validation samples = %d / %d' % (
                    n_t_samples, n_v_samples
                )
            )

            d_train = train_x[:n_t_samples]
            d_val = train_x[n_t_samples:]

            t_train = train_y[:n_t_samples]
            t_val = train_y[n_t_samples:]

            if brains is not None:
                r_train = brains[:n_t_samples]
                r_val = brains[n_t_samples:]
            else:
                r_train = None
                r_val = None

            num_workers = 16

            # Training
            if patch_size is None:
                # Full image one
                print('Dataset creation images <with validation>')
                train_dataset = BBImageDataset(
                    d_train, t_train, r_train
                )
            else:
                # Unbalanced one
                print('Dataset creation unbalanced patches <with validation>')
                train_dataset = BoundarySegmentationCroppingDataset(
                    d_train, t_train, masks=r_train, patch_size=patch_size,
                )

            print('Dataloader creation <with validation>')
            train_loader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers,
            )

            # Validation
            val_dataset = BBImageDataset(
                d_val, t_val, r_val
            )
            val_loader = DataLoader(
                val_dataset, 1, num_workers=num_workers
            )

            net.fit(train_loader, val_loader, epochs=epochs, patience=patience)

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


def train_test_survival(net_name, n_folds):
    # Init
    c = color_codes()
    options = parse_inputs()
    epochs = options['epochs']
    patch_size = options['patch_size']
    batch_size = options['batch_size']
    patience = options['patience']
    depth = options['blocks']
    images = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']

    d_path = options['loo_dir']
    patients = get_dirs(d_path)
    survival_dict = get_survival_data()
    seg_patients = filter(lambda p: p not in survival_dict.keys(), patients)
    survival_patients = survival_dict.keys()
    np.random.seed(42)
    seg_patients = np.random.permutation(patients).tolist()

    ''' Segmentation training'''
    # The goal here is to pretrain a unique segmentation network for all
    # the survival folds. We only do it once. Then we split the survival
    # patients accordingly.
    net = BratsSurvivalNet()

    # Training data
    print('Loading ROI masks...')
    brain_names = map(
        lambda p: os.path.join(
            d_path, p, p + '_t1.nii.gz'
        ),
        seg_patients
    )
    brains = map(get_mask, brain_names)
    print('Loading labels...')
    lesion_names = map(
        lambda p: os.path.join(d_path, p, p + '_seg.nii.gz'),
        seg_patients
    )
    train_y = map(get_mask, lesion_names)
    for yi in train_y:
        yi[yi == 4] = 3
    print('Loading data...')
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
        zip(seg_patients, brains)
    )

    print(
        'Training segmentation samples = %d' % (
            len(seg_patients)
        )
    )

    # Training itself
    model_name = '%s-init.mdl' % net_name
    try:
        net.load_model(os.path.join(d_path, model_name))
    except IOError:
        n_params = sum(
            p.numel() for p in net.base_model.parameters() if p.requires_grad
        )
        print(
            '%sStarting segmentation training with a unet%s (%d parameters)' %
            (c['c'], c['nc'], n_params)
        )

        # Data split (using numpy) for train and validation.
        # We also compute the number of batches for both training and
        # validation according to the batch size.
        n_samples = len(train_x)

        val_split = 0.1

        n_t_samples = int(n_samples * (1 - val_split))
        n_v_samples = n_samples - n_t_samples

        print(
            'Training / validation samples = %d / %d' % (
                n_t_samples, n_v_samples
            )
        )

        d_train = train_x[:n_t_samples]
        d_val = train_x[n_t_samples:]

        t_train = train_y[:n_t_samples]
        t_val = train_y[n_t_samples:]

        if brains is not None:
            r_train = brains[:n_t_samples]
            r_val = brains[n_t_samples:]
        else:
            r_train = None
            r_val = None

        num_workers = 16

        # Training
        if patch_size is None:
            # Full image one
            print('Dataset creation images <with validation>')
            train_dataset = BBImageDataset(
                d_train, t_train, r_train
            )
        else:
            # Unbalanced one
            print('Dataset creation unbalanced patches <with validation>')
            train_dataset = BoundarySegmentationCroppingDataset(
                d_train, t_train, masks=r_train, patch_size=patch_size,
            )

        print('Dataloader creation <with validation>')
        train_loader = DataLoader(
            train_dataset, batch_size, True, num_workers=num_workers,
        )

        # Validation
        val_dataset = BBImageDataset(
            d_val, t_val, r_val
        )
        val_loader = DataLoader(
            val_dataset, 1, num_workers=num_workers
        )

        print(
            '%s[%s] %sInitial %s%spretraining%s' % (
                c['c'], strftime("%H:%M:%S"), c['g'],
                c['c'], c['b'], c['nc']
            )
        )

        net.fit_seg(train_loader, val_loader, epochs=epochs, patience=patience)

        net.save_model(os.path.join(d_path, model_name))

        for i in range(n_folds):
            print(
                '%s[%s] %sFold %s(%s%d%s%s/%d)%s' % (
                    c['c'], strftime("%H:%M:%S"), c['g'],
                    c['c'], c['b'], i + 1, c['nc'], c['c'], n_folds, c['nc']
                )
            )


def main():
    # Init
    c = color_codes()
    options = parse_inputs()
    patch_size = options['patch_size']
    filters = options['filters']
    hybrid = options['hybrid']
    depth = options['blocks']
    mode_s = '-hybrid' if hybrid else ''

    # Prepare the sufix that will be added to the results for the net and images
    print(
        '%s[%s] %s<BRATS 2019 pipeline%s testing>%s' % (
            c['c'], strftime("%H:%M:%S"), c['y'], mode_s, c['nc']
        )
    )

    ''' <Segmentation task> '''
    n_folds = 5
    print(
        '%s[%s] %sStarting cross-validation (segmentation) - %d folds%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )

    patch_s = '-ps%d' % patch_size if patch_size is not None else ''
    depth_s = '-f%d' % depth
    filters_s = '-f%d' % filters
    net_name = 'brats2019-nnunet-%s%s%s%s' % (
        mode_s, filters_s, depth_s, patch_s
    )

    # train_test_seg(net_name, n_folds)

    ''' <Survival task> '''
    net_name = 'brats2019-survival-%s%s%s%s' % (
        mode_s, filters_s, depth_s, patch_s
    )

    print(
        '%s[%s] %sStarting cross-validation (survival) - %d folds%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )
    train_test_survival(net_name, n_folds)


if __name__ == '__main__':
    main()
