from __future__ import print_function
import argparse
import os
import csv
import time
from time import strftime
import numpy as np
from numpy import logical_not as log_not
from models import BratsSegmentationNet, BratsSurvivalNet
from datasets import BratsDataset, BoundarySegmentationCroppingDataset
from datasets import BBImageDataset, BBImageValueDataset
from utils import color_codes, get_dirs, time_to_string
from utils import get_mask, get_normalised_image, remove_small_regions
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
        '-v', '--validation-directory',
        dest='val_dir', default='/home/mariano/DATA/Brats19ValidationData',
        help='Directory containing the data for only testing'
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


def get_survival_data(test=False):
    # Init
    options = parse_inputs()
    path = options['val_dir'] if test else options['loo_dir']

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
    if test:
        final_dict = dict(
            filter(
                lambda (k, v): v['ResectionStatus'] == 'GTR',
                survivaldict.items()
            )
        )
    else:
        final_dict = dict(
            filter(
                lambda (k, v): v['ResectionStatus'] == 'GTR' and isint(v['Survival']),
                survivaldict.items()
            )
        )

    return final_dict


def get_images(names, test=False):
    options = parse_inputs()
    images = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
    d_path = options['val_dir'] if test else options['loo_dir']
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


def train_seg(
        net, model_name, train_patients, val_patients, refine=False,
        dropout=0.99, lr=1.
):
    # Init
    c = color_codes()
    options = parse_inputs()
    epochs = options['epochs']
    patience = options['patience']
    batch_size = options['batch_size']
    patch_size = options['patch_size']
    d_path = options['loo_dir']

    try:
        net.load_model(os.path.join(d_path, model_name))
    except IOError:
        n_params = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )
        print(
            '%sStarting refinement with a unet%s (%d parameters)' %
            (c['c'], c['nc'], n_params)
        )

        num_workers = 8

        # Training
        targets = get_labels(train_patients)
        rois, data = get_images(train_patients)

        print('< Training dataset >')
        if patch_size is None:
            if refine:
                train_dataset = BBImageDataset(
                    data, targets, rois, flip=True, mode='min'
                )
                batch_size = batch_size * 2
            else:
                train_dataset = BBImageDataset(
                    data, targets, rois, flip=True
                )
        else:
            # train_dataset = BoundarySegmentationCroppingDataset(
            #     data, targets, rois, patch_size
            # )
            train_dataset = BratsDataset(
                data, targets, rois, patch_size
            )

        print('Dataloader creation <with validation>')
        train_loader = DataLoader(
            train_dataset, batch_size, True, num_workers=num_workers,
        )

        # Validation
        targets = get_labels(val_patients)
        rois, data = get_images(val_patients)

        print('< Validation dataset >')
        val_dataset = BBImageDataset(
            data, targets, rois
        )

        print('Dataloader creation <val>')
        val_loader = DataLoader(
            val_dataset, 1
        )

        print(
            'Training / validation samples = %d / %d' % (
                len(train_dataset), len(val_dataset)
            )
        )

        net.fit(
            train_loader, val_loader, initial_dropout=dropout, initial_lr=lr,
            epochs=epochs, patience=patience, refine=refine
        )

        net.save_model(os.path.join(d_path, model_name))


def train_test_seg(net_name, n_folds, val_split=0.1):
    # Init
    c = color_codes()
    options = parse_inputs()
    depth = options['blocks']
    filters = options['filters']

    d_path = options['loo_dir']
    unc_path = os.path.join(d_path, 'uncertainty')
    seg_path = os.path.join(d_path, 'segmentation')
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

        # Training
        train_cbica = fold_cbica[:n_cbica]
        train_tcia = fold_tcia[:n_tcia]
        train_tmc = fold_tmc[:n_tmc]
        train_b2013 = fold_b2013[:n_b2013]
        train_patients = train_cbica + train_tcia + train_tmc + train_b2013

        # Validation
        val_cbica = fold_cbica[n_cbica:]
        val_tcia = fold_tcia[n_tcia:]
        val_tmc = fold_tmc[n_tmc:]
        val_b2013 = fold_b2013[n_b2013:]
        val_patients = val_cbica + val_tcia + val_tmc + val_b2013

        model_name = '%s-f%d.mdl' % (net_name, i)
        net = BratsSegmentationNet(depth=depth, filters=filters)

        train_seg(net, model_name, train_patients, val_patients)

        model_name = '%s-f%d-R.mdl' % (net_name, i)
        train_seg(
            net, model_name, train_patients, val_patients,
            refine=True, dropout=0.5, lr=1e-2
        )

        # Testing data (with GT)
        test_cbica = cbica[ini_cbica:end_cbica]
        test_tcia = tcia[ini_tcia:end_tcia]
        test_tmc = tmc[ini_tmc:end_tmc]
        test_b2013 = b2013[ini_b2013:end_b2013]
        test_patients = test_cbica + test_tcia + test_tmc + test_b2013

        patient_paths = map(lambda p: os.path.join(d_path, p), test_patients)
        _, test_x = get_images(test_patients)

        print(
            'Testing patients (with GT) = %d' % (
                len(test_patients)
            )
        )

        # The sub-regions considered for evaluation are:
        #   1) the "enhancing tumor" (ET)
        #   2) the "tumor core" (TC)
        #   3) the "whole tumor" (WT)
        #
        # The provided segmentation labels have values of 1 for NCR & NET,
        # 2 for ED, 4 for ET, and 0 for everything else.
        # The participants are called to upload their segmentation labels
        # as a single multi-label file in nifti (.nii.gz) format.
        #
        # The participants are called to upload 4 nifti (.nii.gz) volumes
        # (3 uncertainty maps and 1 multi-class segmentation volume from
        # Task 1) onto CBICA's Image Processing Portal format. For example,
        # for each ID in the dataset, participants are expected to upload
        # following 4 volumes:
        # 1. {ID}.nii.gz (multi-class label map)
        # 2. {ID}_unc_whole.nii.gz (Uncertainty map associated with whole tumor)
        # 3. {ID}_unc_core.nii.gz (Uncertainty map associated with tumor core)
        # 4. {ID}_unc_enhance.nii.gz (Uncertainty map associated with enhancing tumor)
        for p, (path_i, p_i, test_i) in enumerate(zip(
                patient_paths, test_patients, test_x
        )):
            pred_i = net.segment([test_i])[0]
            unc_i = net.uncertainty([test_i], steps=25)[0]
            whole_i = np.sum(unc_i[1:])
            core_i = unc_i[1] + unc_i[-1]
            enhance_i = unc_i[-1]
            seg_i = np.argmax(pred_i, axis=0)
            seg_i[seg_i == 3] = 4
            seg_unc_i = np.argmax(unc_i, axis=0)
            seg_unc_i[seg_unc_i == 3] = 4

            tumor_mask = remove_small_regions(
                seg_i.astype(np.bool), min_size=30
            )

            seg_i[log_not(tumor_mask)] = 0
            seg_unc_i[log_not(tumor_mask)] = 0

            whole_i *= tumor_mask.astype(np.float32)
            core_i *= tumor_mask.astype(np.float32)
            enhance_i *= tumor_mask.astype(np.float32)

            niiname = os.path.join(path_i, p_i + '_seg.nii.gz')
            nii = load_nii(niiname)
            seg = nii.get_data()

            dsc = map(
                lambda label: dsc_seg(seg == label, seg_i == label),
                [1, 2, 4]
            )
            dsc_unc = map(
                lambda label: dsc_seg(seg == label, seg_unc_i == label),
                [1, 2, 4]
            )

            nii.get_data()[:] = seg_i
            save_nii(nii, os.path.join(seg_path, p_i + '.nii.gz'))
            nii.get_data()[:] = seg_unc_i
            save_nii(nii, os.path.join(unc_path, p_i + '.nii.gz'))

            niiname = os.path.join(d_path, p_i, p_i + '_flair.nii.gz')
            nii = load_nii(niiname)
            nii.get_data()[:] = whole_i
            save_nii(nii, os.path.join(unc_path, p_i + '_unc_whole.nii.gz'))
            nii.get_data()[:] = core_i
            save_nii(nii, os.path.join(unc_path, p_i + '_unc_core.nii.gz'))
            nii.get_data()[:] = enhance_i
            save_nii(nii, os.path.join(unc_path, p_i + '_unc_enhance.nii.gz'))

            print(
                'Segmentation - Patient %s (%d/%d): %s' % (
                    p_i, p, len(test_x), ' / '.join(map(str, dsc))
                )
            )
            print(
                'Uncertainty - Patient %s (%d/%d): %s' % (
                    p_i, p, len(test_x), ' / '.join(map(str, dsc_unc))
                )
            )


def train_test_survival(net_name, n_folds, val_split=0.1):
    # Init
    c = color_codes()
    options = parse_inputs()
    epochs = options['epochs']
    batch_size = options['batch_size']
    patch_size = options['patch_size']
    patience = options['patience']
    depth = options['blocks']
    filters = options['filters']

    d_path = options['loo_dir']
    patients = get_dirs(d_path)
    survival_dict = get_survival_data()
    seg_patients = filter(lambda p: p not in survival_dict.keys(), patients)
    survival_patients = survival_dict.keys()
    low_patients = filter(
        lambda p: int(survival_dict[p]['Survival']) < 300,
        survival_patients
    )
    mid_patients = filter(
        lambda p: (int(survival_dict[p]['Survival']) >= 300) &
                  (int(survival_dict[p]['Survival']) < 450),
        survival_patients
    )
    hi_patients = filter(
        lambda p: int(survival_dict[p]['Survival']) >= 450,
        survival_patients
    )

    t_survival_dict = get_survival_data(True)
    t_survival_patients = t_survival_dict.keys()
    test_survivals = np.zeros(len(t_survival_patients))

    t_survival_ages = map(lambda v: float(v['Age']), t_survival_dict.values())

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
    num_workers = 8
    try:
        net.base_model.load_model(os.path.join(d_path, model_name))
    except IOError:

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
        if patch_size is None:
            seg_dataset = BBImageDataset(
                data, targets, rois, flip=True
            )
        else:
            seg_dataset = BratsDataset(
                data, targets, rois, patch_size
            )

        print('Dataloader creation <with validation>')
        train_loader = DataLoader(
            seg_dataset, batch_size, True, num_workers=num_workers,
        )

        targets = get_labels(survival_patients)
        rois, data = get_images(survival_patients)

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

    tr_csv = os.path.join(options['loo_dir'], 'survival_results.csv')
    val_csv = os.path.join(options['val_dir'], 'survival_results.csv')
    with open(tr_csv, 'w') as csvfile, open(val_csv, 'w') as val_csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        val_csvwriter = csv.writer(val_csvfile, delimiter=',')
        # First we'll define the maximum bounding box, for training and testing
        masks, _ = get_images(survival_patients)
        indices = map(lambda mask: np.where(mask > 0), masks)
        min_bb = np.min(
            map(
                lambda idx: np.min(idx, axis=-1),
                indices
            ),
            axis=0
        )
        max_bb = np.max(
            map(
                lambda idx: np.max(idx, axis=-1),
                indices
            ),
            axis=0
        )
        bb = map(
            lambda (min_i, max_i): slice(min_i, max_i),
            zip(min_bb, max_bb)
        )

        # After that, we can finally train the model to predict the survival.
        for i in range(n_folds):
            model_name = '%s-init.mdl' % net_name
            net = BratsSurvivalNet(depth_seg=depth, depth_pred=depth, filters=filters)
            net.base_model.load_model(os.path.join(d_path, model_name))

            ''' Survival training'''
            model_name = '%s_f%d.mdl' % (net_name, i)
            print(
                '%s[%s] %sFold %s(%s%d%s%s/%d)%s' % (
                    c['c'], strftime("%H:%M:%S"), c['g'],
                    c['c'], c['b'], i + 1, c['nc'], c['c'], n_folds, c['nc']
                )
            )

            try:
                net.load_model(os.path.join(d_path, model_name))
            except IOError:
                # Train/Test patient split according to survival classes
                hi_ini = len(hi_patients) * i / n_folds
                mid_ini = len(mid_patients) * i / n_folds
                low_ini = len(low_patients) * i / n_folds
                hi_end = len(hi_patients) * (i + 1) / n_folds
                mid_end = len(mid_patients) * (i + 1) / n_folds
                low_end = len(low_patients) * (i + 1) / n_folds
                high_i = hi_patients[:hi_ini] + hi_patients[hi_end:]
                mid_i = mid_patients[:mid_ini] + mid_patients[mid_end:]
                low_i = low_patients[:low_ini] + low_patients[low_end:]

                # Split of the target survivals
                hi_survival = map(
                    lambda h_i: float(survival_dict[h_i]['Survival']), high_i
                )
                mid_survival = map(
                    lambda h_i: float(survival_dict[h_i]['Survival']), mid_i
                )
                low_survival = map(
                    lambda h_i: float(survival_dict[h_i]['Survival']), low_i
                )
                # Split of the age feature
                hi_ages = map(
                    lambda h_i: float(survival_dict[h_i]['Age']), high_i
                )
                mid_ages = map(
                    lambda h_i: float(survival_dict[h_i]['Age']), mid_i
                )
                low_ages = map(
                    lambda h_i: float(survival_dict[h_i]['Age']), low_i
                )

                # Data split (using numpy) for train and validation.
                # We also compute the number of batches for both training and
                # validation according to the batch size.
                # Train/Validation split
                n_hi = len(high_i)
                n_mid = len(mid_i)
                n_low = len(low_i)
                n_hitrain = int(n_hi * (1 - val_split))
                n_midtrain = int(n_mid * (1 - val_split))
                n_lowtrain = int(n_low * (1 - val_split))

                # Training
                hi_train = high_i[:n_hitrain]
                mid_train = mid_i[:n_midtrain]
                low_train = low_i[:n_lowtrain]
                train_i = hi_train + mid_train + low_train
                hi_train_ages = hi_ages[:n_hitrain]
                mid_train_ages = mid_ages[:n_midtrain]
                low_train_ages = low_ages[:n_lowtrain]
                train_ages = hi_train_ages + mid_train_ages + low_train_ages
                hi_train_surv = hi_survival[:n_hitrain]
                mid_train_surv = mid_survival[:n_midtrain]
                low_train_surv = low_survival[:n_lowtrain]
                train_surv = hi_train_surv + mid_train_surv + low_train_surv

                print('< Training dataset >')
                train_rois, train_data = get_images(train_i)
                train_dataset = BBImageValueDataset(
                    train_data, train_ages, train_surv, train_rois, bb=bb
                )

                print('Dataloader creation <train>')
                train_loader = DataLoader(
                    train_dataset, 8, True, num_workers=num_workers,
                )

                # Validation
                hi_train = high_i[n_hitrain:]
                mid_train = mid_i[n_midtrain:]
                low_train = low_i[n_lowtrain:]
                val_i = hi_train + mid_train + low_train
                hi_train_ages = hi_ages[n_hitrain:]
                mid_train_ages = mid_ages[n_midtrain:]
                low_train_ages = low_ages[n_lowtrain:]
                val_ages = hi_train_ages + mid_train_ages + low_train_ages
                hi_train_surv = hi_survival[n_hitrain:]
                mid_train_surv = mid_survival[n_midtrain:]
                low_train_surv = low_survival[n_lowtrain:]
                val_surv = hi_train_surv + mid_train_surv + low_train_surv

                print('< Validation dataset >')
                val_rois, val_data = get_images(val_i)
                val_dataset = BBImageValueDataset(
                    val_data, val_ages, val_surv, val_rois, bb=bb
                )

                print('Dataloader creation <val>')
                val_loader = DataLoader(
                    val_dataset, 1, num_workers=num_workers
                )

                n_train = len(train_dataset)
                n_val = len(val_dataset)
                n_test = len(survival_patients) - n_train - n_val
                print(
                    'Train / validation / test samples = %d / %d / %d' % (
                        n_train, n_val, n_test
                    )
                )

                net.fit(
                    train_loader, val_loader, epochs=epochs, patience=patience
                )

                net.save_model(os.path.join(d_path, model_name))

            ''' Survival testing '''
            # Testing data
            high_i = hi_patients[hi_ini:hi_end]
            mid_i = mid_patients[mid_ini:mid_end]
            low_i = low_patients[low_ini:low_end]
            test_patients = high_i + mid_i + low_i

            # Split of the target survivals
            hi_survival = map(
                lambda h_i: float(survival_dict[h_i]['Survival']), high_i
            )
            mid_survival = map(
                lambda h_i: float(survival_dict[h_i]['Survival']), mid_i
            )
            low_survival = map(
                lambda h_i: float(survival_dict[h_i]['Survival']), low_i
            )
            test_survival = hi_survival + mid_survival + low_survival
            # Split of the age feature
            hi_ages = map(
                lambda h_i: float(survival_dict[h_i]['Age']), high_i
            )
            mid_ages = map(
                lambda h_i: float(survival_dict[h_i]['Age']), mid_i
            )
            low_ages = map(
                lambda h_i: float(survival_dict[h_i]['Age']), low_i
            )
            test_ages = hi_ages + mid_ages + low_ages

            _, test_data = get_images(test_patients)

            print(
                'Testing patients (with GT) = %d' % (
                    len(test_patients)
                )
            )

            pred_y = net.predict(test_data, test_ages, bb)
            for p, survival_out, survival in zip(
                    test_patients, pred_y, test_survival
            ):
                print(
                    'Estimated survival = %f (%f)' % (survival_out, survival)
                )
                csvwriter.writerow([p, '%f' % float(survival_out)])

            _, test_data = get_images(t_survival_patients, True)

            print(
                'Testing patients = %d' % (
                    len(t_survival_patients)
                )
            )
            pred_y = net.predict(test_data, t_survival_ages, bb)
            test_survivals += np.array(pred_y)
            for p, survival_out, s in zip(
                    t_survival_patients, pred_y, test_survivals
            ):
                print(
                    'Estimated survival = %f (%f)' % (
                        survival_out, s / (i + 1)
                    )
                )

        test_survivals = test_survivals / n_folds
        for p, survival_out in zip(test_patients, test_survivals):
            print('Final estimated survival = %f' % survival_out)
            val_csvwriter.writerow([p, '%f' % float(survival_out)])


def test_seg_validation(net_name):
    # Init
    c = color_codes()
    options = parse_inputs()
    depth = options['blocks']
    filters = options['filters']
    d_path = options['loo_dir']
    v_path = options['val_dir']
    seg_path = os.path.join(v_path, 'segmentation')
    unc_path = os.path.join(v_path, 'uncertainty')
    p = get_dirs(d_path)[0]
    test_patients = filter(lambda p: 'BraTS19' in p, get_dirs(v_path))
    _, test_x = get_images(test_patients, True)

    print(
        'Testing patients = %d' % (
            len(test_patients)
        )
    )

    # The sub-regions considered for evaluation are:
    #   1) the "enhancing tumor" (ET)
    #   2) the "tumor core" (TC)
    #   3) the "whole tumor" (WT)
    #
    # The provided segmentation labels have values of 1 for NCR & NET,
    # 2 for ED, 4 for ET, and 0 for everything else.
    # The participants are called to upload their segmentation labels
    # as a single multi-label file in nifti (.nii.gz) format.
    #
    # The participants are called to upload 4 nifti (.nii.gz) volumes
    # (3 uncertainty maps and 1 multi-class segmentation volume from
    # Task 1) onto CBICA's Image Processing Portal format. For example,
    # for each ID in the dataset, participants are expected to upload
    # following 4 volumes:
    # 1. {ID}.nii.gz (multi-class label map)
    # 2. {ID}_unc_whole.nii.gz (Uncertainty map associated with whole tumor)
    # 3. {ID}_unc_core.nii.gz (Uncertainty map associated with tumor core)
    # 4. {ID}_unc_enhance.nii.gz (Uncertainty map associated with enhancing tumor)
    for i, (p_i, test_i) in enumerate(zip(test_patients, test_x)):
        t_in = time.time()
        unc_i = np.zeros((4,) + test_i.shape[1:])
        pred_i = np.zeros((4,) + test_i.shape[1:])
        for f in range(5):
            model_name = '%s-f%d.mdl' % (net_name, f)
            net = BratsSegmentationNet(depth=depth, filters=filters)
            net.load_model(os.path.join(d_path, model_name))

            unc_i += net.uncertainty([test_i], steps=10)[0] * 0.2
            pred_i += net.segment([test_i])[0]

        seg_i = np.argmax(pred_i, axis=0)
        seg_i[seg_i == 3] = 4
        seg_unc_i = np.argmax(unc_i, axis=0)
        seg_unc_i[seg_unc_i == 3] = 4

        tumor_mask = remove_small_regions(
            seg_i.astype(np.bool), min_size=30
        )

        seg_i[log_not(tumor_mask)] = 0
        seg_unc_i[log_not(tumor_mask)] = 0

        whole_i = np.sum(unc_i[1:]) * tumor_mask.astype(np.float32)
        core_i = unc_i[1] + unc_i[-1] * tumor_mask.astype(np.float32)
        enhance_i = unc_i[-1] * tumor_mask.astype(np.float32)

        seg_unc_i = np.argmax(unc_i, axis=0)
        seg_unc_i[seg_unc_i == 3] = 4

        niiname = os.path.join(d_path, p, p + '_seg.nii.gz')
        nii = load_nii(niiname)
        nii.get_data()[:] = seg_i
        save_nii(nii, os.path.join(seg_path, p_i + '.nii.gz'))
        nii.get_data()[:] = seg_i
        save_nii(nii, os.path.join(unc_path, p_i + '.nii.gz'))

        niiname = os.path.join(v_path, p_i, p_i + '_flair.nii.gz')
        nii = load_nii(niiname)
        nii.get_data()[:] = whole_i
        save_nii(
            nii, os.path.join(unc_path, p_i + '_unc_whole.nii.gz')
        )
        nii.get_data()[:] = core_i
        save_nii(
            nii, os.path.join(unc_path, p_i + '_unc_core.nii.gz')
        )
        nii.get_data()[:] = enhance_i
        save_nii(
            nii, os.path.join(unc_path, p_i + '_unc_enhance.nii.gz')
        )

        t_s = time_to_string(time.time() - t_in)

        print(
            'Finished patient %s (%d/%d) %s' % (p_i, i + 1, len(test_x), t_s)
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
    filters_s = '-filt%d' % filters
    patch_s = '-ps%d' % patch_size if patch_size is not None else ''
    depth_s = '-d%d' % depth

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
    train_test_survival(net_name, n_folds)

    ''' <Segmentation task> '''
    print(
        '%s[%s] %sStarting cross-validation (segmentation) - %d folds%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )

    net_name = 'brats2019-seg%s%s' % (
        filters_s, depth_s
    )

    # train_test_seg(net_name, n_folds)

    # test_seg_validation(net_name)


if __name__ == '__main__':
    main()
