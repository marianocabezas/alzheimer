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
from models import BratsSurvivalNet


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
        dest='loo_dir', default='/home/mariano/DATA/Brats18TrainingData',
        help='Option to use leave-one-out. The second parameter is the folder with all the patients.'
    )
    parser.add_argument(
        '-e', '--epochs',
        action='store', dest='epochs', type=int, default=30,
        help='Number of maximum epochs for training the segmentation task'
    )
    parser.add_argument(
        '-S', '--number-slices',
        dest='n_slices', type=int, default=30,
        help='Initial patch size'
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


def get_patient_survival_features(path, p, p_features, test=False):
    # Init
    options = parse_inputs()
    roi_sufix = '_seg.nii.gz' if not test else '.nii.gz'
    roi = load_nii(os.path.join(path, p, p + roi_sufix)).get_data()
    brain = load_nii(os.path.join(path, p, p + options['t1'])).get_data()
    brain_vol = np.count_nonzero(brain)
    vol_features = map(lambda l: np.count_nonzero(roi == l) / brain_vol, [1, 2, 4])
    age_features = [float(p_features['Age']) / 100]
    features = [age_features + vol_features]

    return features


def get_patient_roi_slice(path, p):
    options = parse_inputs()
    n_slices = options['n_slices']

    # roi_sufix = '_seg.nii.gz' if not test else '.nii.gz'
    roi_sufix = '.nii.gz'
    roi = load_nii(os.path.join(path, p, p + roi_sufix)).get_data()
    brain = load_nii(os.path.join(path, p, p + options['t1'])).get_data()
    bounding_box_min = np.min(np.nonzero(brain), axis=1)
    bounding_box_max = np.max(np.nonzero(brain), axis=1)
    center_of_masses = np.mean(np.nonzero(roi), axis=1, dtype=np.int)
    slices = [[
        slice(bounding_box_min[0], bounding_box_max[0] + 1),
        slice(bounding_box_min[1], bounding_box_max[1] + 1),
        slice(center_of_masses[-1] - n_slices / 2, center_of_masses[-1] + n_slices / 2)
    ]]

    return slices


def get_names(sufix, path):
    options = parse_inputs()
    if path is None:
        path = options['loo_dir']

    directories = filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])
    patients = sorted(directories)

    return map(lambda p: os.path.join(p, p.split('/')[-1] + sufix), patients)


def get_reshaped_data(
        image_names,
        slices,
        slice_shape,
        n_slices=20,
        datatype=np.float32,
        verbose=False
):
    if verbose:
        print('%s- Loading x' % ' '.join([''] * 12))

    def load_patient_data(names, slices, components=3):
        # Load the images first
        data = np.stack(
            map(lambda im: load_nii(im).get_data()[slices].astype(np.float32), names),
            axis=0,
        ).astype(dtype=datatype)

        # Prepare data for PCA and do it
        pca = decomposition.PCA(n_components=components)
        data_shape = (components,) + data.shape[1:]
        data_pca = pca.fit_transform(np.reshape(data, (len(data), -1)).T).T
        data_pca = np.reshape(data_pca, data_shape)
        data_pca = (data_pca - np.min(data_pca, axis=0)) / (np.max(data_pca, axis=0) - np.min(data_pca, axis=0))

        # Prepare data for the final slice shape
        final_shape = (3,) + slice_shape + (n_slices,)
        data_final = resize(data_pca, final_shape, mode='constant', anti_aliasing=True)
        return data_final

    x = map(lambda (i, s): load_patient_data(i, s), zip(image_names, slices))

    return x


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


def get_survival_data(options, recession=list(['GTR', 'STR', 'NA']), test=False):
    # Init
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

        flair_names = list()
        t1_names = list()
        t2_names = list()
        t1ce_names = list()
        survival = list()
        features = list()
        slices = list()
        names = list()
        for k, v in survivaldict.items():
            if v['ResectionStatus'] in recession:
                flair_names += [os.path.join(path, k, k + options['flair'])]
                t1_names += [os.path.join(path, k, k + options['t1'])]
                t1ce_names += [os.path.join(path, k, k + options['t1ce'])]
                t2_names += [os.path.join(path, k, k + options['t2'])]
                features += get_patient_survival_features(path, k, v, test)
                slices += get_patient_roi_slice(path, k)
                if not test:
                    survival += [float(v['Survival'])]
                else:
                    names += [k]

        image_names = np.stack([flair_names, t1_names, t1ce_names, t2_names], axis=1)

    features = np.array(features)

    if not test:
        survival = np.expand_dims(survival, axis=-1)
        packed_return = (image_names, survival, features, slices)
    else:
        packed_return = (names, image_names, features, slices)

    return packed_return


def preprocess_input(x):
    # We are assuming that the first two dimensions are number of samples and
    # number of channels. This could be hardcoded, since we know we are dealing
    # with RGB images with the VGG input shape, but we left it general, just in case.
    n_cases = x.shape[0]
    n_channels = x.shape[1]
    other_dims = len(x.shape[2:])
    normallizing_shape = (n_cases, n_channels) + (1,) * other_dims
    n_images = np.prod(n_cases * n_channels)
    vgg_mean = np.resize([0.485, 0.456, 0.406], (1, 3,) + (1,) * other_dims)
    vgg_std = np.resize([0.229, 0.224, 0.225], (1, 3,) + (1,) * other_dims)
    max_x = np.max(x.reshape(n_images, -1), axis=1).reshape(normallizing_shape)
    min_x = np.min(x.reshape(n_images, -1), axis=1).reshape(normallizing_shape)
    x_norm = x - min_x
    x_norm /= (max_x - min_x)
    x_norm -= vgg_mean
    x_norm /= vgg_std

    return x_norm


def train_net(net, net_name, image_names, survival, features, slices, experimental=False, verbose=False):
    # Init
    options = parse_inputs()
    c = color_codes()
    # Prepare the net hyperparameters
    epochs = options['epochs']
    n_slices = options['n_slices']

    ''' Training '''
    try:
        net.load_state_dict(torch.load(net_name))
        if verbose:
            print(
                '%s[%s] %sSurvival network weights %sloaded%s' % (
                    c['c'], strftime("%H:%M:%S"), c['g'],
                    c['b'], c['nc']
                )
            )
    except IOError:
        net = net.cuda()

        trainable_params = count_params(net)
        print(
            '%s[%s] %sTraining the survival network %s(%s%d %sparameters)' % (
                c['c'], strftime("%H:%M:%S"), c['g'], c['nc'],
                c['b'], trainable_params, c['nc']
            )
        )

        # Data preparation
        x_vol = get_reshaped_data(image_names, slices, (224, 224), n_slices=n_slices, verbose=verbose)

        if verbose:
            print('%s- Concatenating the data' % ' '.join([''] * 12))
        x_vol = np.stack(x_vol, axis=0)

        if verbose:
            print('%s-- X (volume) shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, x_vol.shape))))
            print('%s-- X (features) shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, features.shape))))
            print('%s-- Y shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, survival.shape))))

            print('%s- Randomising the training data' % ' '.join([''] * 12))

        idx = np.random.permutation(range(len(features)))

        x_vol = preprocess_input(x_vol[idx].astype(np.float32))
        x_feat = features[idx].astype(np.float32)
        x = [x_vol, x_feat]

        y = survival[idx].astype(np.float32)

        if verbose:
            print('%sStarting train loop' % ' '.join([''] * 12))

        if experimental:
            # net.fit_exp(x, y, epochs=epochs, batch_size=2, criterion='mse', val_split=0.25)
            net.fit_exp(x, y, epochs=epochs, batch_size=4, criterion='mse', verbose=verbose)
        else:
            # net.fit(x, y, epochs=epochs, batch_size=2, criterion='mse', val_split=0.25)
            net.fit(x, y, epochs=epochs, batch_size=4, criterion='mse', verbose=verbose, patience=10)
        torch.save(net.state_dict(), net_name)


def train_survival_function(image_names, survival, features, slices, save_path, sufix='', verbose=False):
    # Init
    options = parse_inputs()
    # Prepare the net hyperparameters
    n_slices = options['n_slices']

    # Old network
    net = BratsSurvivalNet(n_slices=n_slices, n_features=features.shape[-1])

    net_name = os.path.join(save_path, 'brats2018-pytorch-survival%s.hdf5' % sufix)

    train_net(net, net_name, image_names, survival, features, slices, verbose=verbose)

    return net


def test_survival_function(net, image_names, features, slices, n_slices, verbose=False):
    if verbose:
        print(
            '%s[%s] %sTesting the survival network %s' % (
                color_codes()['c'], strftime("%H:%M:%S"), color_codes()['g'], color_codes()['nc']
            )
        )

    x_vol = get_reshaped_data(image_names, slices, (224, 224), n_slices=n_slices, verbose=verbose)

    if verbose:
        print('%s- Concatenating the data' % ' '.join([''] * 12))

    x_vol = np.stack(x_vol, axis=0).astype(np.float32)

    if verbose:
        print('%s-- X (volume) shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, x_vol.shape))))
        print('%s-- X (features) shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, features.shape))))

    x_vol = preprocess_input(x_vol.astype(np.float32))
    x_feat = features.astype(np.float32)

    x = [x_vol, x_feat]

    if verbose:
        print('%sStarting test loop' % ' '.join([''] * 12))
    survival = net.predict(x, batch_size=4, verbose=verbose)

    return np.fabs(np.squeeze(survival))


def main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the sufix that will be added to the results for the net and images
    train_data, _ = get_names_from_path()

    print('%s[%s] %s<BRATS 2018 pipeline testing>%s' % (c['c'], strftime("%H:%M:%S"), c['y'], c['nc']))
    print('%s[%s] %sCenter computation%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
    # Block center computation

    ''' <Survival task> '''
    tst_simage_names, tst_survival, tst_features, tst_slices = get_survival_data(options, recession=['GTR'])
    max_survival = np.max(tst_survival)
    n_folds = len(tst_simage_names)
    print('%s[%s] %sStarting leave-one-out (survival)%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
    for i in range(n_folds):
        ''' Training '''
        ini_p = len(tst_simage_names) * i / n_folds
        end_p = len(tst_simage_names) * (i + 1) / n_folds
        # Validation data
        p = tst_simage_names[ini_p:end_p]
        p_features = np.asarray(tst_features[ini_p:end_p])
        p_slices = tst_slices[ini_p:end_p]
        p_survival = tst_survival[ini_p:end_p]
        # Training data
        train_images = np.concatenate([
            tst_simage_names[:ini_p, :],
            tst_simage_names[end_p:, :],
        ], axis=0)
        train_survival = np.asarray(
            tst_survival.tolist()[:ini_p] + tst_survival.tolist()[end_p:]
        )
        train_features = np.asarray(
            tst_features.tolist()[:ini_p] + tst_features.tolist()[end_p:]
        )
        train_slices = tst_slices[:ini_p] + tst_slices[end_p:]

        # Patient info
        p_name = map(lambda pi: pi[0].rsplit('/')[-2], p)

        # Data stuff
        print('%s[%s] %sFold %s(%s%d%s%s/%d)%s' % (
            c['c'], strftime("%H:%M:%S"),
            c['g'], c['c'], c['b'], i + 1, c['nc'], c['c'], n_folds, c['nc']
        ))

        print(
            '%s[%s] %sStarting training (%ssurvival%s)%s' % (
                c['c'], strftime("%H:%M:%S"),
                c['g'], c['b'], c['nc'] + c['g'], c['nc']
            )
        )

        survivals = list()

        for j in range(20):
            net = train_survival_function(
                train_images,
                train_survival / max_survival,
                train_features,
                train_slices,
                save_path=options['loo_dir'],
                sufix='_%02d-fold%02d' % (j, i),
                verbose=True
            )

            ''' Testing '''
            survival = test_survival_function(
                net,
                p,
                p_features,
                p_slices,
                options['n_slices'],
                verbose=True
            ) * max_survival

            survivals += [survival]

        print(
            '%s[%s] %sPatient %s%s%s predicted survival = %s [%smean = %f %s(%s%f%s)] %s(%f)%s' % (
                c['c'], strftime("%H:%M:%S"),
                c['g'], c['b'], p_name[0], c['nc'],
                ' / '.join(map(lambda s: '%s%f%s' % (c['g'], s, c['nc']), survivals)),
                c['g'], np.mean(survivals), c['nc'], c['g'], np.std(survivals), c['nc'], c['g'], p_survival, c['nc']
            )

        )


if __name__ == '__main__':
    main()
