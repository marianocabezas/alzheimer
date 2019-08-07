from __future__ import print_function
import argparse
import os
import csv
from time import strftime
import numpy as np
from models import BratsSegmentationNet, BratsSurvivalNet
from models import BratsNewSegmentationNet
from datasets import BratsSegmentationCroppingDataset, BlocksBBDataset
from datasets import BBImageDataset, BBImageValueDataset
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
        help='Number of samples per batch'
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


def get_images(names):
    options = parse_inputs()
    images = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
    d_path = options['loo_dir']
    print('Loading rois...')
    brain_names = map(
        lambda p: os.path.join(
            d_path, p, p + '_t1.nii.gz'
        ),
        names
    )
    rois = map(get_mask, brain_names)
    print('Loading and normalising images...')
    data = map(
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
        zip(names, rois)
    )

    return rois, data


def get_labels(names):
    options = parse_inputs()
    d_path = options['loo_dir']
    print('Loading labels...')
    lesion_names = map(
        lambda p: os.path.join(d_path, p, p + '_seg.nii.gz'),
        names
    )
    targets = map(get_mask, lesion_names)
    for yi in targets:
        yi[yi == 4] = 3

    return targets


def train_test_seg(net_name, n_folds, val_split=0.1):
    # Init
    c = color_codes()
    options = parse_inputs()
    epochs = options['epochs']
    patience = options['patience']
    depth = options['blocks']
    filters = options['filters']
    images = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']

    d_path = options['loo_dir']
    patients = get_dirs(d_path)

    cbica = filter(lambda p: 'CBICA' in p, patients)
    tcia = filter(lambda p: 'TCIA' in p, patients)
    tmc = filter(lambda p: 'TMC' in p, patients)
    b2013 = filter(lambda p: '2013' in p, patients)

    for i in range(n_folds):
        print(
            '%s[%s] %sFold %s(%s%d%s%s/%d)%s' % (
                c['c'], strftime("%H:%M:%S"), c['g'],
                c['c'], c['b'], i + 1, c['nc'], c['c'], n_folds, c['nc']
            )
        )

        # Training itself
        # Data split (using the patient names) for train and validation.
        # We also compute the number of batches for both training and
        # validation according to the batch size.
        ''' Training '''
        ini_cbica = len(cbica) * i / n_folds
        end_cbica = len(cbica) * (i + 1) / n_folds
        fold_cbica = cbica[:ini_cbica] + cbica[end_cbica:]
        n_fold_cbica = len(fold_cbica)
        n_cbica = int(n_fold_cbica * (1 - val_split))

        ini_tcia = len(tcia) * i / n_folds
        end_tcia = len(tcia) * (i + 1) / n_folds
        fold_tcia = tcia[:ini_tcia] + tcia[end_tcia:]
        n_fold_tcia = len(fold_tcia)
        n_tcia = int(n_fold_tcia * (1 - val_split))

        ini_tmc = len(tmc) * i / n_folds
        end_tmc = len(tmc) * (i + 1) / n_folds
        fold_tmc = tmc[:ini_tmc] + tmc[end_tmc:]
        n_fold_tmc = len(fold_tmc)
        n_tmc = int(n_fold_tmc * (1 - val_split))

        ini_b2013 = len(b2013) * i / n_folds
        end_b2013 = len(b2013) * (i + 1) / n_folds
        fold_b2013 = b2013[:ini_b2013] + b2013[end_b2013:]
        n_fold_b2013 = len(fold_b2013)
        n_b2013 = int(n_fold_b2013 * (1 - val_split))

        training_n = n_fold_cbica + n_fold_tcia + n_fold_tmc + n_fold_b2013
        testing_n = len(patients) - training_n

        print(
            'Training / testing samples = %d / %d' % (
                training_n, testing_n
            )
        )

        model_name = '%s-hybrid_f%d.mdl' % (net_name, i)
        net = BratsNewSegmentationNet(depth=depth, filters=filters)
        # net = BratsSegmentationNet(depth=depth, filters=filters)
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

            num_workers = 4

            # Training
            train_cbica = fold_cbica[:n_cbica]
            train_tcia = fold_tcia[:n_tcia]
            train_tmc = fold_tmc[:n_tmc]
            train_b2013 = fold_b2013[:n_b2013]
            train_patients = train_cbica + train_tcia + train_tmc + train_b2013

            targets = get_labels(train_patients)
            rois, data = get_images(train_patients)

            print('< Training dataset (images) >')
            im_dataset = BBImageDataset(
                data, targets, rois, flip=True
            )

            print('Dataloader creation <train-images>')
            im_loader = DataLoader(
                im_dataset, 1, True,
            )

            print(
                '%d images / %d batches' % (
                    len(im_dataset), len(im_loader)
                )
            )

            print('< Training dataset (patches) >')
            patch_dataset = BlocksBBDataset(
                data, targets, patch_size=32, flip=True
            )

            print('Dataloader creation <train-patches>')
            patch_loader = DataLoader(
                patch_dataset, 64, True
            )

            print(
                '%d patches / %d batches' % (
                    len(patch_dataset), len(patch_loader)
                )
            )

            # Validation
            val_cbica = fold_cbica[n_cbica:]
            val_tcia = fold_tcia[n_tcia:]
            val_tmc = fold_tmc[n_tmc:]
            val_b2013 = fold_b2013[n_b2013:]

            val_patients = val_cbica + val_tcia + val_tmc + val_b2013

            print('< Validation dataset >')
            targets = get_labels(val_patients)
            rois, data = get_images(val_patients)
            print('< Training dataset (images) >')
            val_dataset = BBImageDataset(
                data, targets, rois
            )

            print('Dataloader creation <val>')
            val_loader = DataLoader(
                val_dataset, 1
            )

            # print(
            #     'Training / validation samples = %d / %d' % (
            #         len(train_dataset), len(val_dataset)
            #     )
            # )

            print(
                'Training / validation samples = %d / %d' % (
                    len(patch_dataset) + len(im_dataset), len(val_dataset)
                )
            )

            # net.fit(train_loader, val_loader, epochs=epochs, patience=patience)
            net.fit(im_loader, patch_loader, val_loader, epochs=epochs, patience=patience)

            net.save_model(os.path.join(d_path, model_name))

        # Testing data
        test_cbica = cbica[ini_cbica:end_cbica]
        test_tcia = tcia[ini_tcia:end_tcia]
        test_tmc = tmc[ini_tmc:end_tmc]
        test_b2013 = b2013[ini_b2013:end_b2013]
        test_patients = test_cbica + test_tcia + test_tmc + test_b2013
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


def train_test_survival(net_name, n_folds, val_split=0.1):
    # Init
    c = color_codes()
    options = parse_inputs()
    epochs = options['epochs']
    patch_size = options['patch_size']
    batch_size = options['batch_size']
    patience = options['patience']
    depth = options['blocks']
    filters = options['filters']

    d_path = options['loo_dir']
    patients = get_dirs(d_path)
    survival_dict = get_survival_data()
    seg_patients = filter(lambda p: p not in survival_dict.keys(), patients)
    survival_patients = survival_dict.keys()
    survival_ages = map(lambda v: v['Age'], survival_dict.values())
    survivals = map(lambda v: v['Survival'], survival_dict.values())

    ''' Segmentation training'''
    # The goal here is to pretrain a unique segmentation network for all
    # the survival folds. We only do it once. Then we split the survival
    # patients accordingly.
    net = BratsSurvivalNet(depth_seg=depth, depth_pred=depth, filters=filters)

    print(
        'Training segmentation samples = %d' % (
            len(seg_patients)
        )
    )

    # Training itself
    model_name = '%s-init.mdl' % net_name
    try:
        net.base_model.load_model(os.path.join(d_path, model_name))
    except IOError:
        num_workers = 16

        n_params = sum(
            p.numel() for p in net.base_model.parameters() if p.requires_grad
        )
        print(
            '%sStarting segmentation training with a unet%s (%d parameters)' %
            (c['c'], c['nc'], n_params)
        )

        targets = get_labels(seg_patients)
        rois, data = get_images(seg_patients)

        print('< Training dataset >')
        seg_dataset = BBImageDataset(
            data, targets, rois, flip=True
        )

        print('Dataloader creation <with validation>')
        train_loader = DataLoader(
            seg_dataset, batch_size, True, num_workers=num_workers,
        )

        # Validation
        print('< Validation dataset >')
        val_dataset = BBImageDataset(
            data, targets, rois
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

        net.base_model.fit(train_loader, val_loader, epochs=epochs, patience=patience)

        net.base_model.save_model(os.path.join(d_path, model_name))

        ''' Survival training'''
        # After that, we can finally train the model to predict the survival.
        for i in range(n_folds):
            ini_i = len(survival_patients) * i / n_folds
            end_i = len(survival_patients) * (i + 1) / n_folds
            fold_i = survival_patients[:ini_i] + survival_patients[end_i:]
            survival_i = survivals[:ini_i] + survivals[end_i:]
            n_fold = len(fold_i)
            n_train = int(n_fold * (1 - val_split))

            # Data split (using numpy) for train and validation.
            # We also compute the number of batches for both training and
            # validation according to the batch size.
            # Training
            train_i = fold_i[:n_train]
            train_ages = survival_ages[:n_train]
            train_survival = survival_i[:n_train]

            print('< Training dataset >')
            train_data, train_rois = get_images(train_i)
            train_dataset = BBImageValueDataset(
                train_data, train_ages, train_survival, train_rois
            )

            print('Dataloader creation <train>')
            train_loader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers,
            )

            # Validation
            val_i = fold_i[n_train:]
            val_ages = survival_ages[n_train:]
            val_survival = survival_i[n_train:]

            print('< Validation dataset >')
            val_data, val_rois = get_images(val_i)
            val_dataset = BBImageValueDataset(
                val_data, val_ages, val_survival, val_rois
            )

            print('Dataloader creation <val>')
            val_loader = DataLoader(
                val_dataset, 1, num_workers=num_workers
            )

            print(
                'Training / validation samples = %d / %d' % (
                    len(train_dataset), len(val_dataset)
                )
            )

            net.fit(train_loader, val_loader, epochs=epochs, patience=patience)

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

    # Prepare the sufix that will be added to the results for the net and images
    n_folds = 5
    filters_s = '-fold%d' % filters
    patch_s = '-ps%d' % patch_size if patch_size is not None else ''
    depth_s = '-filt%d' % depth

    print(
        '%s[%s] %s<BRATS 2019 pipeline testing>%s' % (
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    ''' <Survival task> '''
    net_name = 'brats2019-survival%s%s%s' % (
        filters_s, depth_s, patch_s
    )

    print(
        '%s[%s] %sStarting cross-validation (survival) - %d folds%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )
    #train_test_survival(net_name, n_folds)

    ''' <Segmentation task> '''
    print(
        '%s[%s] %sStarting cross-validation (segmentation) - %d folds%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )

    net_name = 'brats2019-%s%s' % (
        filters_s, depth_s
    )

    train_test_seg(net_name, n_folds)


if __name__ == '__main__':
    main()
