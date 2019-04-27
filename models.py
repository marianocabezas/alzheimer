from __future__ import print_function
import time
import sys
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from layers import ScalingLayer, SpatialTransformer, SmoothingLayer
from criterions import normalised_xcor_loss, normalised_mi_loss, subtraction_loss
from criterions import df_modulo, df_loss, weighted_subtraction_loss
from criterions import histogram_loss, mahalanobis_loss
from criterions import dsc_bin_loss
from datasets import ImageListDataset, ImagePairListDataset
from datasets import ImagePairListCroppingDataset, ImageListCroppingDataset
from optimizers import AdaBound


def to_torch_var(
        np_array,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        requires_grad=False,
        dtype=torch.float32
):
    var = torch.tensor(
        np_array,
        requires_grad=requires_grad,
        device=device,
        dtype=dtype
    )
    return var


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

    def forward(self, *inputs):
        pass

    def mini_batch_loop(self, x, y, n_batches, batch_size, epoch, criterion_alg, optimizer_alg=None):
        losses = list()
        for batch_i in range(n_batches):
            batch_ini = batch_i * batch_size
            batch_end = (batch_i + 1) * batch_size
            # Mini batch loop
            # I'll try to support both multi-input and multi-output approaches. That
            # is why we have this "complicated" batch approach.
            if isinstance(x, list):
                batch_x = map(lambda b: to_torch_var(b[batch_ini:batch_end], requires_grad=True), x)
            else:
                batch_x = to_torch_var(x[batch_ini:batch_end], requires_grad=True)

            if isinstance(y, list):
                batch_y = map(lambda b: to_torch_var(b[batch_ini:batch_end]), y)
            else:
                batch_y = to_torch_var(y[batch_ini:batch_end])

            # We train the model and check the loss
            # torch.cuda.synchronize()
            y_pred = self(batch_x)
            # torch.cuda.synchronize()
            batch_loss = criterion_alg(y_pred, batch_y)

            loss_value = batch_loss.tolist()
            losses += [loss_value]

            if optimizer_alg is not None:
                percent = 20 * (batch_i + 1) / n_batches
                progress_s = ''.join(['.'] * percent)
                remainder_s = ''.join(['-'] * (20 - percent))
                whites = ' '.join([''] * 12)
                batch_s = '%sEpoch %03d (%02d/%02d) [%s>%s] loss %f (%f)' % (
                    whites, epoch,
                    batch_i, n_batches,
                    progress_s, remainder_s,
                    loss_value, np.asscalar(np.mean(losses))
                )
                print('\033[K', end='')
                print(batch_s, end='\r')
                sys.stdout.flush()

                # Backpropagation
                optimizer_alg.zero_grad()
                batch_loss.backward()
                optimizer_alg.step()

        return np.mean(losses)

    def mini_batch_exp_loop(
            self,
            train_x, train_y,
            val_x, val_y,
            n_t_batches, n_v_batches, batch_size,
            epoch, criterion_alg, optimizer_alg
    ):
        # Init
        # We need to keep the initial state to check which batch is better
        trained = [False] * n_t_batches
        best_loss = np.inf
        final_loss = np.inf
        for step in range(n_t_batches):
            # Step init
            t_in = time.time()

            losses = list()
            b_batch = 0
            base_state = deepcopy(self.state_dict())
            best_state = base_state
            base_optim = deepcopy(optimizer_alg.state_dict())
            best_optim = base_optim
            best_loss_batch = np.inf

            for batch_i in range(n_t_batches):
                # We haven't trained with that batch yet
                if not trained[batch_i]:
                    batch_ini = batch_i * batch_size
                    batch_end = (batch_i + 1) * batch_size
                    # Mini batch loop
                    # I'll try to support both multi-input and multi-output approaches. That
                    # is why we have this "complicated" batch approach.
                    if isinstance(train_x, list):
                        batch_x = map(
                            lambda b: to_torch_var(
                                b[batch_ini:batch_end],
                                requires_grad=True
                            ),
                            train_x
                        )
                    else:
                        batch_x = to_torch_var(
                            train_x[batch_ini:batch_end],
                            requires_grad=True
                        )

                    if isinstance(train_y, list):
                        batch_y = map(
                            lambda b: to_torch_var(
                                b[batch_ini:batch_end]
                            ),
                            train_y
                        )
                    else:
                        batch_y = to_torch_var(train_y[batch_ini:batch_end])

                    # We train the model and check the loss
                    # torch.cuda.synchronize()
                    y_pred = self(batch_x)
                    # torch.cuda.synchronize()
                    batch_loss = criterion_alg(y_pred, batch_y)

                    # Backpropagation
                    optimizer_alg.zero_grad()
                    batch_loss.backward()
                    optimizer_alg.step()

                    # Validation of that mini batch
                    with torch.no_grad():
                        loss_value = self.mini_batch_loop(
                            val_x, val_y,
                            n_v_batches, batch_size,
                            epoch,
                            criterion_alg
                        )

                    if loss_value < best_loss_batch:
                        best_state = deepcopy(self.state_dict())
                        best_optim = deepcopy(optimizer_alg.state_dict())
                        best_loss_batch = loss_value
                        b_batch = batch_i
                    losses += ['%5.2f' % loss_value]
                else:
                    loss_value = np.inf
                    losses += ['\033[30m  -  \033[0m']

                percent = 20 * (batch_i + 1) / n_t_batches
                progress_s = ''.join(['.'] * percent)
                remainder_s = ''.join(['-'] * (20 - percent))
                whites = ' '.join([''] * 12)
                batch_s = '%sStep %3d-%03d (%2d/%2d) [%s>%s] loss %f (best %f)'
                print('\033[K', end='')
                print(
                    batch_s % (
                        whites, epoch, step,
                        batch_i, n_t_batches,
                        progress_s, remainder_s,
                        loss_value, best_loss_batch
                    ),
                    end='\r'
                )
                sys.stdout.flush()

                # Reload the network to its initial state
                self.load_state_dict(base_state)
                optimizer_alg.load_state_dict(base_optim)

            # Prepare for the next step
            trained[b_batch] = True
            self.load_state_dict(best_state)
            optimizer_alg.load_state_dict(best_optim)

            t_out = time.time() - t_in

            if best_loss_batch < best_loss:
                best_loss = best_loss_batch
                color = '\033[32m'
            else:
                color = '\033[31m'

            n_s = ' %s |'
            b_s = ' %s%s\033[0m |'
            losses_s = map(
                lambda (i, l): n_s % l if i != b_batch else b_s % (color, l),
                enumerate(losses)
            )

            print('\033[K', end='')
            whites = ' '.join([''] * 12)
            if step == 0:
                hdr_dashes = ''.join(['--------'] * n_t_batches)
                bdy_dashes = ''.join(['-------|'] * n_t_batches)
                losses_h = ''.join(
                    map(
                        lambda i: ' b  %2d |' % i,
                        range(n_t_batches)
                    )
                )
                print('%s---------%s--------' % (whites, hdr_dashes))
                print('%sEpo-num |%s  time  ' % (whites, losses_h))
                print('%s--------|%s--------' % (whites, bdy_dashes))
            print('%s%3d-%03d |%s %.2fs' % (
                whites, epoch, step, ''.join(losses_s), t_out
            ))

            final_loss = best_loss_batch

        return final_loss

    def fit(
            self,
            data,
            target,
            val_split=0,
            criterion='xentr',
            optimizer='adam',
            epochs=100,
            patience=10,
            batch_size=32,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.train()

        best_e = 0
        e = 0
        best_loss_tr = np.inf
        best_loss_val = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        validation = val_split > 0

        criterion_dict = {
            'xentr': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
        }

        optimizer_dict = {
            'adam': torch.optim.Adam,
            'adabound': AdaBound,
        }

        model_params = filter(lambda p: p.requires_grad, self.parameters())

        criterion_alg = criterion_dict[criterion]() if isinstance(criterion, basestring) else criterion
        optimizer_alg = optimizer_dict[optimizer](model_params) if isinstance(optimizer, basestring) else optimizer

        t_start = time.time()

        # Data split (using numpy) for train and validation. We also compute the number of batches for both
        # training and validation according to the batch size.
        n_samples = len(data) if not isinstance(data, list) else len(data[0])

        n_t_samples = int(n_samples * (1 - val_split))
        n_t_batches = -(-n_t_samples / batch_size)

        n_v_samples = n_samples - n_t_samples
        n_v_batches = -(-n_v_samples / batch_size)

        d_train = data[:n_t_samples] if not isinstance(data, list) else map(lambda d: d[:n_t_samples], data)
        d_val = data[n_t_samples:] if not isinstance(data, list) else map(lambda d: d[n_t_samples:], data)

        t_train = target[:n_t_samples] if not isinstance(target, list) else map(lambda t: t[:n_t_samples], target)
        t_val = target[n_t_samples:] if not isinstance(target, list) else map(lambda t: t[n_t_samples:], target)

        if verbose:
            print('%sTraining / validation samples = %d / %d' % (' '.join([''] * 12), n_t_samples, n_v_samples))

        for e in range(epochs):
            # Main epoch loop
            t_in = time.time()
            loss_tr = self.mini_batch_loop(d_train, t_train, n_t_batches, batch_size, e, criterion_alg, optimizer_alg)
            # Patience check and validation/real-training loss and accuracy
            improvement = loss_tr < best_loss_tr
            if loss_tr < best_loss_tr:
                best_loss_tr = loss_tr
                loss_s = '\033[32m%0.5f\033[0m' % loss_tr
            else:
                loss_s = '%0.5f' % loss_tr

            if validation:
                with torch.no_grad():
                    loss_val = self.mini_batch_loop(d_val, t_val, n_v_batches, batch_size, e, criterion_alg)

                improvement = loss_val < best_loss_val
                if improvement:
                    best_loss_val = loss_val
                    loss_s += ' | \033[36m%0.5f\033[0m' % loss_val
                    best_e = e
                else:
                    loss_s += ' | %0.5f' % loss_val

            if improvement:
                best_e = e
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                no_improv_e += 1

            t_out = time.time() - t_in
            if verbose:
                print('\033[K', end='')
                if e == 0:
                    print(
                        '%sEpoch num | tr_loss%s |  time  ' % (
                            ' '.join([''] * 12),
                            ' | vl_loss' if validation else '')
                    )
                    print(
                        '%s----------|--------%s-|--------' % (
                            ' '.join([''] * 12),
                            '-|--------' if validation else '')
                    )
                print('%sEpoch %03d | %s | %.2fs' % (' '.join([''] * 12), e, loss_s, t_out))

            if no_improv_e == patience:
                self.load_state_dict(best_state)
                break

        t_end = time.time() - t_start
        if verbose:
            print(
                'Training finished in %d epochs (%fs) with minimum loss = %f (epoch %d)' % (
                    e + 1, t_end, best_loss_tr, best_e)
            )

    def fit_exp(
            self,
            data,
            target,
            val_split=0,
            criterion='xentr',
            optimizer='adam',
            epochs=100,
            patience=10,
            batch_size=32,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        # Init
        self.to(device)
        self.train()

        best_e = 0
        e = 0
        best_loss_tr = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        validation = val_split > 0

        criterion_dict = {
            'xentr': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
        }

        optimizer_dict = {
            'adam': torch.optim.Adam,
        }

        model_params = filter(lambda p: p.requires_grad, self.parameters())

        criterion_alg = criterion_dict[criterion]() if isinstance(criterion, basestring) else criterion
        optimizer_alg = optimizer_dict[optimizer](model_params) if isinstance(optimizer, basestring) else optimizer

        t_start = time.time()

        # Data split (using numpy) for train and validation. We also compute the number of batches for both
        # training and validation according to the batch size.
        n_samples = len(data) if not isinstance(data, list) else len(data[0])

        n_t_samples = int(n_samples * (1 - val_split))
        n_t_batches = -(-n_t_samples / batch_size)

        n_v_samples = n_samples - n_t_samples
        n_v_batches = -(-n_v_samples / batch_size)

        d_train = data[:n_t_samples] if not isinstance(data, list) else map(lambda d: d[:n_t_samples], data)
        d_val = data[n_t_samples:] if not isinstance(data, list) else map(lambda d: d[n_t_samples:], data)

        t_train = target[:n_t_samples] if not isinstance(target, list) else map(lambda t: t[:n_t_samples], target)
        t_val = target[n_t_samples:] if not isinstance(target, list) else map(lambda t: t[n_t_samples:], target)

        print('%sTraining / validation samples = %d / %d' % (' '.join([''] * 12), n_t_samples, n_v_samples))

        for e in range(epochs):
            # Main epoch loop
            t_in = time.time()
            if validation:
                loss_tr = self.mini_batch_exp_loop(
                    d_train, t_train,
                    d_val, t_val,
                    n_t_batches, n_v_batches, batch_size,
                    e, criterion_alg, optimizer_alg
                )
            else:
                loss_tr = self.mini_batch_exp_loop(
                    d_train, t_train,
                    deepcopy(d_train), deepcopy(t_train),
                    n_t_batches, n_t_batches, batch_size,
                    e, criterion_alg, optimizer_alg
                )

            if loss_tr < best_loss_tr:
                loss_s = '\033[32m%7.4f\033[0m' % loss_tr
                best_loss_tr = loss_tr
                best_e = e
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                loss_s = '%2.4f' % loss_tr
                no_improv_e += 1

            t_out = time.time() - t_in

            print('\033[K', end='')
            print('%sEpoch %3d | tr_loss = %s | %.2fs' % (' '.join([''] * 12), e, loss_s, t_out))

            if no_improv_e == patience:
                self.load_state_dict(best_state)
                break

        t_end = time.time() - t_start
        print(
            'Training finished in %d epochs (%fs) with minimum loss = %f (epoch %d)' % (
                e + 1, t_end, best_loss_tr, best_e
            )
        )

    def predict(
            self,
            data,
            batch_size=32,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            verbose=True
    ):
        # Init
        self.to(device)
        self.eval()

        n_batches = -(-len(data) / batch_size) if not isinstance(data, list) else -(-len(data[0]) / batch_size)

        y_pred = list()
        with torch.no_grad():
            for batch_i in range(n_batches):
                # Print stuff
                if verbose:
                    percent = 20 * (batch_i + 1) / n_batches
                    bar = '[' + ''.join(['.'] * percent) + '>' + ''.join(['-'] * (20 - percent)) + ']'
                    print(
                        '\033[K%sTesting batch (%02d/%02d) %s' % (' '.join([''] * 12), batch_i, n_batches, bar),
                        end='\r'
                    )

                # Testing stuff
                batch_ini = batch_i * batch_size
                batch_end = (batch_i + 1) * batch_size
                # Mini batch loop
                # I'll try to support both multi-input and multi-output approaches. That
                # is why we have this "complicated" batch approach.
                if isinstance(data, list):
                    batch_x = map(lambda b: to_torch_var(b[batch_ini:batch_end]), data)
                else:
                    batch_x = to_torch_var(data[batch_i * batch_size:(batch_i + 1) * batch_size])

                # We test the model with the current batch
                # torch.cuda.synchronize()
                y_pred += self(batch_x).tolist()
                # torch.cuda.synchronize()

        if verbose:
            print('\033[K%sTesting finished succesfully' % ' '.join([''] * 12))
            print('\033[K%sTesting finished succesfully' % ' '.join([''] * 12))

        return y_pred


class BratsSurvivalNet(CustomModel):
    def __init__(
            self,
            n_slices,
            n_features=4,
            dense_size=256,
            dropout=0.1,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):

        # Init
        super(BratsSurvivalNet, self).__init__()

        # VGG init
        base_model = nn.Sequential(*list(models.vgg16(pretrained=True).children())[:-1])
        base_model.to(device)
        for param in base_model.parameters():
            param.requires_grad = False

        self.base_model = base_model
        self.batchnorm = nn.BatchNorm2d(512)

        self.vgg_fcc1 = ScalingLayer((512, 7, 7))
        self.dropout1 = nn.Dropout2d(dropout)
        self.vgg_pool1 = nn.AvgPool2d(2)

        self.vgg_fcc2 = ScalingLayer((512, 3, 3))
        self.dropout2 = nn.Dropout2d(dropout)
        self.vgg_pool2 = nn.AvgPool2d(2)

        self.vgg_fcc3 = ScalingLayer((512, 1, 1))
        self.dropout3 = nn.Dropout2d(dropout)

        # Linear activation?
        self.vgg_dense = nn.Linear(512, dense_size)

        self.dense = nn.Linear((dense_size * n_slices) + n_features, 1)

    def forward(self, x):

        x_sliced = map(
            lambda xi: torch.squeeze(xi, dim=-1),
            torch.split(x[0], 1, dim=-1)
        )
        vgg_in = map(self.base_model, x_sliced)
        vgg_norm = map(self.batchnorm, vgg_in)

        vgg_fccout1 = map(self.vgg_fcc1, vgg_norm)
        vgg_relu1 = map(F.relu, vgg_fccout1)
        # vgg_dropout1 = map(self.dropout1, vgg_relu1)
        # vgg_poolout1 = map(self.vgg_pool1, vgg_dropout1)
        vgg_poolout1 = map(self.vgg_pool1, vgg_relu1)

        # vgg_fccout2 = map(self.dropout2, map(F.relu, map(self.vgg_fcc2, vgg_poolout1)))
        vgg_fccout2 = map(F.relu, map(self.vgg_fcc2, vgg_poolout1))
        vgg_poolout2 = map(self.vgg_pool2, vgg_fccout2)

        # vgg_fccout3 = map(self.dropout3, map(F.relu, map(self.vgg_fcc3, vgg_poolout2)))
        vgg_fccout3 = map(F.relu, map(self.vgg_fcc3, vgg_poolout2))

        vgg_out = torch.cat(
            map(
                self.vgg_dense,
                map(
                    lambda yi: yi.view(-1, yi.size()[1]),
                    vgg_fccout3
                )
            ),
            dim=-1
        )

        # Here we add the final layers to compute the survival value

        final_tensor = torch.cat([x[1], vgg_out], dim=-1)
        output = self.dense(final_tensor)
        return output


class MultiViewBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels=64,
            pool=2,
            kernels=[3, 5, 7, 9],
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        # Init
        super(MultiViewBlock, self).__init__()
        self.out_channels=out_channels
        self.kernels = kernels
        # We need to separate the output channels into pooled ones and normal
        # ones
        conv_channels = out_channels // 2
        pool_channels = out_channels - conv_channels

        # First we create the convolutional channels
        conv_filters = self._get_filters_list(conv_channels)
        self.convs = map(
            lambda (f_out, k): nn.Conv3d(in_channels, f_out, k, padding=k // 2),
            zip(conv_filters, kernels)
        )
        for c in self.convs:
            c.to(device)

        pool_filters = self._get_filters_list(pool_channels)
        self.pools = map(
            lambda (f_out, k): nn.Sequential(
                nn.Conv3d(in_channels, f_out, k, pool, k // 2),
                nn.ConvTranspose3d(in_channels, f_out, 1, pool),
            ),
            zip(pool_filters, kernels)
        )
        for c in self.pools:
            c.to(device)

    def _get_filters_list(self, channels):
        n_kernels = len(self.kernels)
        n_kernels_1 = n_kernels - 1
        filter_k = int(round(1.0 * channels / n_kernels))
        filters_k = (filter_k,) * n_kernels_1
        filters = filters_k + (channels - n_kernels_1 * filter_k,)

        return filters

    def forward(self, *inputs):
        conv_out = map(lambda c: c(inputs), self.convs)
        pool_out = map(lambda c: c(inputs), self.pools)

        return torch.cat(conv_out + pool_out, dim=1)


class MaskAtrophyNet(nn.Module):
    def __init__(
            self,
            conv_filters=list([32, 64, 64, 64]),
            deconv_filters=list([64, 64, 64, 64, 64, 32, 32]),
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            lambda_d=1,
            leakyness=0.2,
            loss_idx=list([0, 1, 6]),
            kernel_size=3,
            data_smooth=False,
            df_smooth=False,
            trainable_smooth=False,
    ):
        # Init
        final_filters = deconv_filters[-1]
        loss_names = list([
                ' subt ',
                'subt_l',
                ' xcor ',
                'xcor_l',
                ' mse  ',
                'mahal ',
                ' hist ',
                'deform',
                'modulo',
                ' n_mi ',
                'n_mi_l',
            ])
        super(MaskAtrophyNet, self).__init__()
        self.data_smooth = data_smooth
        self.df_smooth = df_smooth
        self.epoch = 0
        self.optimizer_alg = None
        self.lambda_d = lambda_d
        self.loss_names = map(lambda idx: loss_names[idx], loss_idx)
        self.device = device
        self.leakyness = leakyness
        # Down path of the unet
        conv_in = [2] + conv_filters[:-1]
        self.conv = map(
            lambda (f_in, f_out): nn.Conv3d(
                f_in, f_out, 3, padding=1, stride=2
            ),
            zip(conv_in, conv_filters)
        )
        unet_filters = len(conv_filters)
        for c in self.conv:
            c.to(device)
            nn.init.kaiming_normal_(c.weight)

        # Up path of the unet
        conv_out = conv_filters[-1]
        deconv_in = [conv_out] + map(
            sum, zip(deconv_filters[:unet_filters - 1], conv_in[::-1])
        )
        self.deconv_u = map(
            lambda (f_in, f_out): nn.ConvTranspose3d(
                f_in, f_out, 3, padding=1, stride=2
            ),
            zip(
                deconv_in,
                deconv_filters[:unet_filters]
            )
        )
        for d in self.deconv_u:
            d.to(device)
            nn.init.kaiming_normal_(d.weight)

        # Extra DF path
        deconv_out = 2 + deconv_filters[unet_filters - 1]
        zipped_f = zip(
            [deconv_out] + deconv_filters[unet_filters:-1],
            deconv_filters[unet_filters:]
        )
        if kernel_size is not None:
            pad = kernel_size // 2
            self.deconv = map(
                lambda (f_in, f_out): nn.Conv3d(
                    f_in, f_out, kernel_size, padding=pad
                ),
                zipped_f
            )
            for d in self.deconv:
                d.to(device)
                nn.init.kaiming_normal_(d.weight)
        else:
            self.deconv = map(
                lambda (f_in, f_out): MultiViewBlock(
                    f_in, f_out
                ),
                zipped_f
            )
            for d in self.deconv:
                d.to(device)

        # Final DF computation
        self.to_df = nn.Conv3d(final_filters, 3, 1)
        self.to_df.to(device)
        nn.init.normal_(self.to_df.weight, 0.0, 1e-5)

        self.smooth = SmoothingLayer(trainable=trainable_smooth)
        self.smooth.to(device)

        self.trans_im = SpatialTransformer()
        self.trans_im.to(device)
        self.trans_mask = SpatialTransformer('nearest')
        self.trans_mask.to(device)

    def forward(self, inputs):
        n_inputs = len(inputs)
        if n_inputs > 3:
            patch_source, target, mask, mesh, source = inputs
            data = torch.cat([patch_source, target], dim=1)
        else:
            source, target, mask = inputs
            data = torch.cat([source, target], dim=1)

        if self.data_smooth:
            data = self.smooth(data)

        down_inputs = list()
        for c in self.conv:
            down_inputs.append(data)
            data = F.leaky_relu(c(data), self.leakyness)

        for d, i in zip(self.deconv_u, down_inputs[::-1]):
            up = F.leaky_relu(d(data, output_size=i.size()), self.leakyness)
            data = torch.cat((up, i), dim=1)

        for d in self.deconv:
            data = F.leaky_relu(d(data), self.leakyness)

        df = self.to_df(data)

        if self.df_smooth:
            df = self.smooth(df)

        if n_inputs > 3:
            source_mov = self.trans_im(
                [source, df, mesh]
            )

            mask_mov = self.trans_mask(
                [mask, df, mesh]
            )
        else:
            source_mov = self.trans_im(
                [source, df]
            )

            mask_mov = self.trans_mask(
                [mask, df]
            )

        return source_mov, mask_mov, df

    def register(
            self,
            cases,
            masks,
            brain_masks,
            batch_size=1,
            val_batch_size=1,
            optimizer='adam',
            epochs=100,
            patience=10,
            num_workers=10,
            patch_based=False,
            patch_size=32,
            curriculum=False,
            verbose=True
    ):
        # Init
        self.train()
        max_step = max(map(len, cases)) - 1

        # Optimizer init
        optimizer_dict = {
            'adadelta': torch.optim.Adadelta,
            'adam': torch.optim.Adam,
            'adabound': AdaBound,
        }
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = optimizer_dict[optimizer](model_params)

        # Pre-loop init
        best_loss_tr = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        t_start = time.time()

        # This is actually a registration approach. It uses the nn framework
        # but it doesn't actually do any supervised training. Therefore, there
        # is no real validation.
        # Due to this, we modified the generic fit algorithm.
        if curriculum:
            curr_step = 1
            epochs = epochs * max_step
            overlap = patch_size * 3 // 4
        else:
            curr_step = None
            overlap = 8

        if patch_based:
            tr_dataset = ImageListCroppingDataset(
                cases, masks, masks,
                patch_size=patch_size, overlap=overlap, step=curr_step
            )
        else:
            tr_dataset = ImageListDataset(
                cases, masks, brain_masks, step=curr_step
            )
        val_dataset = ImageListDataset(
            cases, masks, brain_masks, limits_only=True
        )
        tr_dataloader = DataLoader(
            tr_dataset, batch_size, True, num_workers=num_workers
        )

        val_dataloader = DataLoader(
            val_dataset, val_batch_size, False, num_workers=num_workers
        )

        l_names = [' loss '] + self.loss_names
        best_losses = [np.inf] * (len(l_names) - 1)
        best_e = 0

        for self.epoch in range(epochs):
            # Main epoch loop
            t_in = time.time()
            with torch.autograd.set_detect_anomaly(True):
                self.step_train(tr_dataloader)

                loss_tr, mid_losses = self.step_validate(val_dataloader)

            losses_color = map(
                lambda (pl, l): '\033[36m%s\033[0m' if l < pl else '%s',
                zip(best_losses, mid_losses)
            )
            losses_s = map(
                lambda (c, l): c % '{:8.4f}'.format(l),
                zip(losses_color, mid_losses)
            )
            best_losses = map(
                lambda (pl, l): l if l < pl else pl,
                zip(best_losses, mid_losses)
            )

            # Patience check
            improvement = loss_tr < best_loss_tr
            loss_s = '{:8.4f}'.format(loss_tr)
            if improvement:
                best_loss_tr = loss_tr
                epoch_s = '\033[32mEpoch %03d\033[0m' % self.epoch
                loss_s = '\033[32m%s\033[0m' % loss_s
                best_e = self.epoch
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch %03d' % self.epoch
                no_improv_e += 1

            t_out = time.time() - t_in
            t_s = '%.2fs' % t_out

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                if self.epoch == 0:
                    l_bars = '--|--'.join(['-' * 6] * len(l_names))
                    l_hdr = '  |  '.join(l_names)
                    print('%sEpoch num |  %s  |' % (whites, l_hdr))
                    print('%s----------|--%s--|' % (whites, l_bars))
                final_s = whites + ' | '.join([epoch_s, loss_s] + losses_s + [t_s])
                print(final_s)

            if no_improv_e == patience:
                # If we are going to use curriculum learning, once we surpass
                # the patience value, we see if we can increase the difficulty.
                # That means changing the dataloader for a new one with a
                # bigger step between timepoints.
                if curriculum and curr_step < max_step:
                    # Print the end of an "era"
                    whites = ' '.join([''] * 12)
                    l_bars = '--|--'.join(['-' * 6] * len(l_names))
                    print('%s----------|--%s--|' % (whites, l_bars))

                    # Re-init
                    curr_step += 1
                    no_improv_e = 0
                    self.load_state_dict(best_state)

                    # New dataloaders
                    if patch_based:
                        tr_dataset = ImageListCroppingDataset(
                            cases, masks, masks,
                            overlap=overlap, step=curr_step
                        )
                    else:
                        tr_dataset = ImageListDataset(
                            cases, masks, brain_masks, step=curr_step
                        )
                    tr_dataloader = DataLoader(
                        tr_dataset, batch_size, True, num_workers=num_workers
                    )
                else:
                    break

        self.epoch = best_e
        self.load_state_dict(best_state)
        t_end = time.time() - t_start
        if verbose:
            print(
                'Registration finished in %d epochs (%fs) with minimum loss = %f (epoch %d)' % (
                    self.epoch + 1, t_end, best_loss_tr, best_e)
            )

    def step_train(self, dataloader, optimizer_alg=None):
        if optimizer_alg is not None:
            self.optimizer_alg = optimizer_alg
        n_batches = len(dataloader)
        loss_list = []
        for (batch_i, (inputs, output)) in enumerate(dataloader):
            b_losses = self.step(inputs, output)

            # Final loss value computation per batch
            b_loss_value = sum(b_losses).tolist()
            loss_list.append(b_loss_value)
            mean_loss = np.mean(loss_list)

            # Print the intermediate results
            self.print_progress(batch_i, n_batches, b_loss_value, mean_loss)

    def step_validate(self, dataloader):
        with torch.no_grad():
            losses_list = []
            n_batches = len(dataloader)
            for (batch_i, (inputs, output)) in enumerate(dataloader):
                b_losses = self.step(inputs, output, False)

                b_mid_losses = map(lambda l: l.tolist(), b_losses)
                losses_list.append(b_mid_losses)

                # Print the intermediate results
                self.print_progress(
                    batch_i, n_batches,
                    sum(b_mid_losses),
                    np.sum(np.mean(zip(*losses_list), axis=1)),
                    False
                )

        mid_losses = np.mean(zip(*losses_list), axis=1)
        loss_value = np.sum(mid_losses)

        return loss_value, mid_losses

    def step(
            self,
            inputs,
            output,
            train=True
    ):
        # Again. This is supposed to be a step on the registration process,
        # there's no need for splitting data. We just compute the deformation,
        # then compute the global loss (and intermediate ones to show) and do
        # back propagation.
        # We train the model and check the loss
        n_inputs = len(inputs)
        if n_inputs > 4:
            b_source, b_target, b_lesion, b_mask, b_mesh, b_im, b_m = inputs
            b_source = b_source.to(self.device)
            b_target = b_target.to(self.device)
            b_lesion = b_lesion.to(self.device)
            b_mask = b_mask.to(self.device)
            b_mesh = b_mesh.to(self.device)
            b_im = b_im.to(self.device)
            b_m = b_m.to(self.device)

            b_gt = output.to(self.device)

            b_inputs = (b_source, b_target, b_m, b_mesh, b_im)

        else:
            b_source, b_target, b_lesion, b_mask = inputs
            b_source = b_source.to(self.device)
            b_target = b_target.to(self.device)
            b_lesion = b_lesion.to(self.device)
            b_mask = b_mask.to(self.device)

            b_gt = output.to(self.device)

            b_inputs = (b_source, b_target, b_lesion)

        torch.cuda.synchronize()
        b_moved, b_moved_lesion, b_df = self(b_inputs)

        b_losses = self.longitudinal_loss(
            b_source,
            b_moved,
            b_lesion,
            b_moved_lesion,
            b_gt,
            b_df,
            b_mask,
            train
        )

        if train:
            self.optimizer_alg.zero_grad()
            sum(b_losses).to(output.device).backward()
            self.optimizer_alg.step()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return b_losses

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss, train=True):
        init_c = '\033[0m' if train else '\033[38;5;238m'
        whites = ' '.join([''] * 12)
        percent = 20 * (batch_i + 1) / n_batches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))
        loss_name = 'train_loss' if train else 'val_loss'
        batch_s = '%s%sEpoch %03d (%03d/%03d) [%s>%s] %s %f (%f)%s' % (
            init_c, whites, self.epoch, batch_i + 1, n_batches,
            progress_s, remainder_s,
            loss_name, b_loss, mean_loss, '\033[0m'
        )
        print('\033[K', end='')
        print(batch_s, end='\r')
        sys.stdout.flush()

    def transform(
            self,
            source,
            target,
            mask,
            verbose=True
    ):
        # Init
        self.eval()

        source_tensor = to_torch_var(source)
        target_tensor = to_torch_var(target)
        mask_tensor = to_torch_var(mask)

        with torch.no_grad():
            torch.cuda.synchronize()
            source_mov, mask_mov, df = self(
                (source_tensor, target_tensor, mask_tensor)
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if verbose:
            print(
                '\033[K%sTransformation finished' % ' '.join([''] * 12)
            )

        source_mov = map(np.squeeze, source_mov.cpu().numpy())
        mask_mov = map(np.squeeze, mask_mov.cpu().numpy())
        df = map(np.squeeze, df.cpu().numpy())

        return source_mov, mask_mov, df

    def longitudinal_loss(
            self,
            source,
            moved,
            mask,
            moved_mask,
            target,
            df,
            roi,
            train=True
    ):
        # Init
        moved_lesion = moved[moved_mask > 0]
        target_lesion = target[moved_mask > 0]
        source_lesion = source[mask > 0]
        moved_roi = moved[roi > 0]
        target_roi = target[roi > 0]

        float_mask = mask.type(torch.float32)
        # float_mask = moved_mask.type(torch.float32)
        mask_w = torch.sum(float_mask) / float_mask.numel()

        functions = {
            ' subt ': weighted_subtraction_loss,
            'subt_l': subtraction_loss,
            ' xcor ': normalised_xcor_loss,
            'xcor_l': normalised_xcor_loss,
            ' mse  ': torch.nn.MSELoss(),
            'mahal ': mahalanobis_loss,
            ' hist ': histogram_loss,
            'deform': df_loss,
            'modulo': df_modulo,
            ' mod_l': df_modulo,
            ' n_mi ': normalised_mi_loss,
            'n_mi_l': normalised_mi_loss,

        }

        inputs = {
            ' subt ': (moved, target, roi > 0),
            'subt_l': (moved, target, moved_mask > 0),
            ' xcor ': (moved_roi, target_roi),
            'xcor_l': (moved_lesion, target_lesion),
            ' mse  ': (moved_roi, target_roi),
            'mahal ': (moved_lesion, source_lesion),
            ' hist ': (moved_lesion, source_lesion),
            'deform': (df, roi),
            'modulo': (df, roi),
            ' mod_l': (df, mask),
            ' n_mi ': (moved_roi, target_roi),
            'n_mi_l': (moved_lesion, target_lesion),
        }

        weights = {
            ' subt ': 1.0,
            'subt_l': 10.0 * mask_w if train else 1.0,
            ' xcor ': 1.0,
            'xcor_l': 1.0,
            ' mse  ': 1.0,
            'mahal ': 1.0,
            ' hist ': 1.0,
            'deform': self.lambda_d,
            'modulo': 1.0,
            ' mod_l': 1.0,
            ' n_mi ': 1.0,
            'n_mi_l': 1.0,
        }

        losses = tuple(
            map(
                lambda l: weights[l] * functions[l](*inputs[l]),
                self.loss_names
            )
        )

        return losses

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class LongitudinalNet(nn.Module):
    def __init__(
            self,
            conv_filters_s=list([32, 64, 64, 64]),
            conv_filters_r=list([32, 64, 64, 64]),
            deconv_filters_r=list([64, 64, 64, 64, 64, 32, 32]),
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            lambda_d=1,
            leakyness=0.2,
            loss_idx=list([0, 1, 7]),
            data_smooth=False,
            df_smooth=False,
            trainable_smooth=False
    ):
        super(LongitudinalNet, self).__init__()
        # Init
        self.epoch = 0
        self.optimizer_alg = None
        self.atrophy = MaskAtrophyNet(
            conv_filters=conv_filters_r,
            deconv_filters=deconv_filters_r,
            device=device,
            lambda_d=lambda_d,
            leakyness=leakyness,
            loss_idx=loss_idx,
            data_smooth=data_smooth,
            df_smooth=df_smooth,
            trainable_smooth=trainable_smooth
        )
        self.device = device

        # Down path of the unet
        conv_in = [5] + conv_filters_s[:-1]
        self.down = map(
            lambda (f_in, f_out): nn.Conv3d(
                f_in, f_out, 3, stride=2
            ),
            zip(conv_in, conv_filters_s)
        )
        for c in self.down:
            c.to(device)

        # Up path of the unet
        deconv_in = [conv_filters_s[-1]] + map(
            sum, zip(conv_filters_s[-2::-1], conv_filters_s[:0:-1])
        )
        self.up = map(
            lambda (f_in, f_out): nn.ConvTranspose3d(
                f_in, f_out, 3, stride=2
            ),
            zip(
                deconv_in,
                conv_filters_s[::-1]
            )
        )
        for d in self.up:
            d.to(device)

        self.seg = nn.Conv3d(conv_filters_s[0] + 5, 1, 1)
        self.seg.to(device)

    def forward(self, inputs):
        # Init
        n_inputs = len(inputs)

        # 2 inputs means registration + segmentation
        # 4 inputs means registration
        if n_inputs == 2:
            source, target = inputs
            mask = None
        else:
            source, target, mask = inputs

        # This is exactly like the MaskAtrophy net
        input_r = torch.cat([source, target], dim=1)

        down_inputs = list()
        for c in self.atrophy.conv:
            down_inputs.append(input_r)
            input_r = F.leaky_relu(
                c(input_r),
                self.atrophy.leakyness
            )

        for d, i in zip(self.atrophy.deconv_u, down_inputs[::-1]):
            up = F.leaky_relu(
                d(input_r, output_size=i.size()),
                self.atrophy.leakyness
            )
            input_r = torch.cat((up, i), dim=1)

        for d in self.atrophy.deconv:
            input_r = F.leaky_relu(
                d(input_r),
                self.atrophy.leakyness
            )

        df = self.atrophy.to_df(input_r)

        source_mov = self.atrophy.trans_im(
            [source, df]
        )

        if n_inputs > 2:
            if mask is not None:
                # We just register the mask and we are done here!
                mask_mov = self.atrophy.trans_mask(
                    [mask, df]
                )
            else:
                mask_mov = None
            return source_mov, mask_mov, df
        else:
            # Now we actually need to give a segmentation result.
            input_s = torch.cat([source, target, df], dim=1)
            down_inputs = list()
            for c in self.down:
                down_inputs.append(input_s)
                input_s = F.relu(c(input_s))

            for d, i in zip(self.up, down_inputs[::-1]):
                up = F.relu(d(input_s, output_size=i.size()))
                input_s = torch.cat((up, i), dim=1)

            seg = torch.sigmoid(self.seg(input_s))

            return seg, source_mov

    def transform(
            self,
            source,
            target,
            mask=None,
            verbose=True
    ):
        # Init
        # Init
        self.eval()

        source_tensor = to_torch_var(source)
        target_tensor = to_torch_var(target)
        if mask is not None:
            mask_tensor = to_torch_var(mask, dtype=torch.int32)
        else:
            mask_tensor = None

        with torch.no_grad():
            torch.cuda.synchronize()
            source_mov, mask_mov, df = self(
                (source_tensor, target_tensor, mask_tensor)
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if verbose:
            print(
                '\033[K%sTransformation finished' % ' '.join([''] * 12)
            )

        source_mov = map(np.squeeze, source_mov.cpu().numpy())
        df = map(np.squeeze, df.cpu().numpy())
        if mask_mov is not None:
            mask_mov = map(np.squeeze, mask_mov.cpu().numpy())
            return source_mov, mask_mov, df
        else:
            return source_mov, df

    def new_lesions(
            self,
            source,
            target,
            verbose=True
    ):
        # Init
        # Init
        self.eval()

        source_tensor = to_torch_var(source)
        target_tensor = to_torch_var(target)

        with torch.no_grad():
            torch.cuda.synchronize()
            seg, _ = self((source_tensor, target_tensor))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if verbose:
            print(
                '\033[K%sTransformation finished' % ' '.join([''] * 12)
            )

        return map(np.squeeze, seg.cpu().numpy())

    def register(
            self,
            source,
            target,
            new_lesion,
            cases,
            masks,
            brain_masks,
            seg_batch_size=32,
            reg_batch_size=1,
            optimizer='adam',
            epochs=100,
            patience=10,
            num_workers=10,
            verbose=True
    ):
        # Init
        self.train()

        # Optimizer init
        optimizer_dict = {
            'adam': torch.optim.Adam,
            'adabound': AdaBound,
        }
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = optimizer_dict[optimizer](model_params)

        # Pre-loop init
        best_loss_tr = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        t_start = time.time()

        # This is actually a registration approach. It uses the nn framework
        # but it doesn't actually do any supervised training. Therefore, there
        # is no real validation.
        # Due to this, we modified the generic fit algorithm.
        seg_dataset = ImagePairListCroppingDataset(
            source, target, new_lesion, overlap=24
        )
        seg_dataloader = DataLoader(
            seg_dataset, seg_batch_size, True, num_workers=num_workers
        )

        reg_dataset = ImageListCroppingDataset(
            cases, masks, brain_masks
        )
        reg_dataloader = DataLoader(
            reg_dataset, reg_batch_size, True, num_workers=num_workers
        )

        l_names = [' loss '] + self.atrophy.loss_names + ['  dsc ']
        best_losses = [np.inf] * (len(l_names))
        best_e = 0
        e = 0

        for self.epoch in range(epochs):
            self.atrophy.epoch = self.epoch
            # Main epoch loop
            t_in = time.time()
            self.step_train(
                seg_dataloader,
                reg_dataloader
            )
            loss_seg, loss_reg, mid_reg_losses = self.step_validate(
                seg_dataloader,
                reg_dataloader
            )

            if loss_reg is not None and mid_reg_losses is not None:
                losses_color = map(
                    lambda (pl, l): '\033[36m%s\033[0m' if l < pl else '%s',
                    zip(best_losses, mid_reg_losses)
                )
                losses_s = map(
                    lambda (c, l): c % '{:8.4f}'.format(l),
                    zip(losses_color, mid_reg_losses)
                )
                best_losses = map(
                    lambda (pl, l): l if l < pl else pl,
                    zip(best_losses, mid_reg_losses)
                )
            else:
                loss_reg = []
                losses_s = []

            # Patience check
            loss_value = loss_seg + loss_reg
            improvement = loss_value < best_loss_tr
            loss_s = '{:8.4f}'.format(loss_value)
            if improvement:
                best_loss_tr = loss_value
                epoch_s = '\033[32mEpoch %03d\033[0m' % self.epoch
                loss_s = '\033[32m%s\033[0m' % loss_s
                best_e = self.epoch
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch %03d' % self.epoch
                no_improv_e += 1

            t_out = time.time() - t_in
            t_s = '%.2fs' % t_out

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                if self.epoch == 0:
                    l_bars = '--|--'.join(['-' * 6] * len(l_names))
                    l_hdr = '  |  '.join(l_names)
                    print('%sEpoch num |  %s  |' % (whites, l_hdr))
                    print('%s----------|--%s--|' % (whites, l_bars))
                final_s = whites + ' | '.join([epoch_s, loss_s] + losses_s + [t_s])
                print(final_s)

            if no_improv_e == patience:
                break

        self.epoch = best_e
        self.load_state_dict(best_state)
        t_end = time.time() - t_start
        if verbose:
            print(
                'Registration finished in %d epochs (%fs) with minimum loss = %f (epoch %d)' % (
                    e + 1, t_end, best_loss_tr, best_e)
            )

    def step_train(
            self,
            dataloader_seg,
            dataloader_reg=None,
    ):
        # This step should combine both registration and segmentation.
        # The goal is to affect the deformation with two datasets and different
        # goals and loss functions.
        with torch.autograd.set_detect_anomaly(True):
            if dataloader_reg is not None:
                # Registration update
                self.atrophy.step_train(
                    dataloader_reg,
                    optimizer_alg=self.optimizer_alg
                )

            # Segmentation update
            n_batches = len(dataloader_seg)
            loss_list = []
            for batch_i, (inputs, output) in enumerate(dataloader_seg):
                b_seg_loss = self.step(inputs, output)

                b_loss_value = b_seg_loss.tolist()
                loss_list.append(b_loss_value)

                mean_loss = np.mean(loss_list)

                # Print the intermediate results
                self.print_progress(batch_i, n_batches, b_loss_value, mean_loss)

    def step_validate(
            self,
            dataloader_seg,
            dataloader_reg=None,
    ):

        if dataloader_reg is not None:
            loss_reg_value, mid_reg_losses = self.atrophy.step_validate(
                dataloader_reg
            )
        else:
            loss_reg_value, mid_reg_losses = None, None

        with torch.no_grad():
            n_batches = len(dataloader_seg)
            losses_list = []
            for batch_i, (inputs, output) in enumerate(dataloader_seg):
                b_loss = self.step(inputs, output, False)

                losses_list.append(b_loss.tolist())

                self.print_progress(
                    batch_i, n_batches,
                    b_loss.tolist(),
                    np.mean(losses_list),
                    False
                )

        loss_seg_value = np.mean(losses_list)
        if mid_reg_losses is not None:
            mid_reg_losses = mid_reg_losses.tolist()
            mid_reg_losses.append(loss_seg_value)

        return loss_seg_value, loss_reg_value, mid_reg_losses

    def step(
            self,
            inputs,
            output,
            train=True
    ):
        # We train the model and check the loss
        b_source = inputs[0].to(self.device)
        b_target = inputs[1].to(self.device)
        b_lesion = output.to(self.device)

        torch.cuda.synchronize()
        b_pred_lesion, b_moved = self(
            (b_source, b_target)
        )
        b_loss = dsc_bin_loss(b_pred_lesion, b_lesion)

        if train:
            self.optimizer_alg.zero_grad()
            b_loss.backward()
            self.optimizer_alg.step()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # return b_loss
        return b_loss

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss, train=True):
        init_c = '\033[0m' if train else '\033[38;5;238m'
        whites = ' '.join([''] * 12)
        percent = 20 * (batch_i + 1) / n_batches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))
        loss_name = 'train_loss' if train else 'val_loss'
        batch_s = '%s%sEpoch %03d (%03d/%03d) [%s][%s>%s] %s %f (%f)%s' % (
            init_c, whites, self.epoch, batch_i + 1, n_batches,
            ''.join(['-'] * 21), progress_s, remainder_s,
            loss_name, b_loss, mean_loss, '\033[0m'
        )
        print('\033[K', end='')
        print(batch_s, end='\r')
        sys.stdout.flush()

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class NewLesionsNet(nn.Module):
    def __init__(
            self,
            conv_filters_s=list([32, 64, 64, 64]),
            conv_filters_r=list([32, 64, 64, 64]),
            deconv_filters_r=list([64, 64, 64, 64, 64, 32, 32]),
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            lambda_d=1,
            leakyness=0.2,
            loss_idx=list([0, 1, 7]),
            data_smooth=False,
            df_smooth=False,
            trainable_smooth=False,
            hybrid=False,
    ):
        super(NewLesionsNet, self).__init__()
        # Init
        self.epoch = 0
        self.optimizer_alg = None
        self.atrophy = MaskAtrophyNet(
            conv_filters=conv_filters_r,
            deconv_filters=deconv_filters_r,
            device=device,
            lambda_d=lambda_d,
            leakyness=leakyness,
            loss_idx=loss_idx,
            data_smooth=data_smooth,
            df_smooth=df_smooth,
            trainable_smooth=trainable_smooth
        )
        self.lambda_d = lambda_d
        self.device = device
        self.hybrid = hybrid

        loss_names = list([
                ' subt ',
                ' xcor ',
                ' mse  ',
                'deform',
                'modulo',
                ' n_mi ',
            ])
        self.loss_names = map(lambda idx: loss_names[idx], loss_idx)

        # Down path of the unet
        conv_in = [5] + conv_filters_s[:-1]
        self.down = map(
            lambda (f_in, f_out): nn.Conv3d(
                f_in, f_out, 3, stride=2
            ),
            zip(conv_in, conv_filters_s)
        )
        for c in self.down:
            c.to(device)

        # Up path of the unet
        deconv_in = [conv_filters_s[-1]] + map(
            sum, zip(conv_filters_s[-2::-1], conv_filters_s[:0:-1])
        )
        self.up = map(
            lambda (f_in, f_out): nn.ConvTranspose3d(
                f_in, f_out, 3, stride=2
            ),
            zip(
                deconv_in,
                conv_filters_s[::-1]
            )
        )
        for d in self.up:
            d.to(device)

        self.seg = nn.Conv3d(conv_filters_s[0] + 5, 2, 1)
        self.seg.to(device)

    def forward(self, inputs):
        # Init
        source, target = inputs

        input_r = torch.cat([source, target], dim=1)

        # This is exactly like the MaskAtrophy net
        down_inputs = list()
        for c in self.atrophy.conv:
            down_inputs.append(input_r)
            input_r = F.leaky_relu(
                c(input_r),
                self.atrophy.leakyness
            )

        for d, i in zip(self.atrophy.deconv_u, down_inputs[::-1]):
            up = F.leaky_relu(
                d(input_r, output_size=i.size()),
                self.atrophy.leakyness
            )
            input_r = torch.cat((up, i), dim=1)

        for d in self.atrophy.deconv:
            input_r = F.leaky_relu(
                d(input_r),
                self.atrophy.leakyness
            )

        df = self.atrophy.to_df(input_r)

        source_mov = self.atrophy.trans_im(
            [source, df]
        )

        # Now we actually need to give a segmentation result.
        input_s = torch.cat([source, target, df], dim=1)
        down_inputs = list()
        for c in self.down:
            down_inputs.append(input_s)
            input_s = F.relu(c(input_s))

        for d, i in zip(self.up, down_inputs[::-1]):
            up = F.relu(d(input_s, output_size=i.size()))
            input_s = torch.cat((up, i), dim=1)

        seg = torch.split(torch.softmax(self.seg(input_s), dim=1), 1, dim=1)[1]

        return seg, source_mov, df

    def new_lesions(
            self,
            source,
            target,
            verbose=True
    ):
        # Init
        self.eval()

        source_tensor = to_torch_var(source)
        target_tensor = to_torch_var(target)

        with torch.no_grad():
            torch.cuda.synchronize()
            seg, source_mov, df = self(
                (source_tensor, target_tensor)
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if verbose:
            print(
                '\033[K%sTransformation finished' % ' '.join([''] * 12)
            )

        source_mov = map(np.squeeze, source_mov.cpu().numpy())
        df = map(np.squeeze, df.cpu().numpy())
        seg = map(np.squeeze, seg.cpu().numpy())
        return seg, source_mov, df

    def fit(
            self,
            source,
            target,
            new_lesion,
            masks,
            reg_batch_size=1,
            seg_batch_size=64,
            optimizer='adam',
            epochs=100,
            patience=10,
            num_workers=10,
            verbose=True
    ):
        # Init
        self.train()

        # Optimizer init
        optimizer_dict = {
            'adam': torch.optim.Adam,
            'adabound': AdaBound,
        }
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = optimizer_dict[optimizer](model_params)

        # Pre-loop init
        best_loss_tr = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        t_start = time.time()

        # We have a mix of registration and segmentation here. We still don't
        # use a validation set, but instead we train first and then recheck
        # the loss. Since part of the training is unsupervised... Having both
        # validation and training might be not that interesting.
        reg_dataloader = None
        if self.hybrid:
            # Image based dataloader
            cases = map(list, zip(source, target))
            reg_dataset = ImageListDataset(
                cases, new_lesion, masks
            )
            reg_dataloader = DataLoader(
                reg_dataset, reg_batch_size, True, num_workers=num_workers
            )

            # Patch based data loader
            seg_dataset = ImagePairListCroppingDataset(
                source, target, new_lesion, masks, overlap=28
            )
            seg_dataloader = DataLoader(
                seg_dataset, seg_batch_size, True, num_workers=num_workers
            )

            # Validation dataloader
            val_dataset = ImagePairListDataset(
                source, target, new_lesion, masks
            )
            val_dataloader = DataLoader(
                val_dataset, reg_batch_size, True, num_workers=num_workers
            )
        else:
            seg_dataset = ImagePairListDataset(
                source, target, new_lesion, masks
            )
            seg_dataloader = DataLoader(
                seg_dataset, reg_batch_size, True, num_workers=num_workers
            )

        l_names = [' loss '] if self.hybrid else\
            [' loss '] + self.loss_names + ['  dsc ']
        best_losses = [np.inf] * (len(l_names))
        best_e = 0
        e = 0

        for self.epoch in range(epochs):
            self.atrophy.epoch = self.epoch
            # Main epoch loop
            t_in = time.time()
            self.step_train(
                dataloader_seg=seg_dataloader,
                dataloader_reg=reg_dataloader
            )

            if self.hybrid:
                loss_value = self.step_validate(
                    val_dataloader
                )
                losses_s = []
            else:
                loss_value, mid_losses = self.step_validate(
                    seg_dataloader
                )

                losses_color = map(
                    lambda (pl, l): '\033[36m%s\033[0m' if l < pl else '%s',
                    zip(best_losses, mid_losses)
                )
                losses_s = map(
                    lambda (c, l): c % '{:8.4f}'.format(l),
                    zip(losses_color, mid_losses)
                )
                best_losses = map(
                    lambda (pl, l): l if l < pl else pl,
                    zip(best_losses, mid_losses)
                )

            # Patience check
            improvement = loss_value < best_loss_tr
            loss_s = '{:8.4f}'.format(loss_value)
            if improvement:
                best_loss_tr = loss_value
                epoch_s = '\033[32mEpoch %03d\033[0m' % self.epoch
                loss_s = '\033[32m%s\033[0m' % loss_s
                best_e = self.epoch
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch %03d' % self.epoch
                no_improv_e += 1

            t_out = time.time() - t_in
            t_s = '%.2fs' % t_out

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                if self.epoch == 0:
                    l_bars = '--|--'.join(['-' * 6] * len(l_names))
                    l_hdr = '  |  '.join(l_names)
                    print('%sEpoch num |  %s  |' % (whites, l_hdr))
                    print('%s----------|--%s--|' % (whites, l_bars))
                final_s = whites + ' | '.join([epoch_s, loss_s] + losses_s + [t_s])
                print(final_s)

            if no_improv_e == patience:
                break

        self.epoch = best_e
        self.load_state_dict(best_state)
        t_end = time.time() - t_start
        if verbose:
            print(
                'Registration finished in %d epochs (%fs) with minimum loss = %f (epoch %d)' % (
                    e + 1, t_end, best_loss_tr, best_e)
            )

    def step_train(
            self,
            dataloader_seg,
            dataloader_reg=None,
    ):
        # This step should combine both registration and segmentation.
        # The goal is to affect the deformation with two datasets and different
        # goals and loss functions.
        with torch.autograd.set_detect_anomaly(True):
            if dataloader_reg is not None:
                # Registration update
                self.atrophy.step_train(
                    dataloader_reg,
                    optimizer_alg=self.optimizer_alg
                )

            # Segmentation update
            n_batches = len(dataloader_seg)
            loss_list = []
            for batch_i, (inputs, output) in enumerate(dataloader_seg):
                b_seg_loss = self.batch_step(inputs, output)

                if self.hybrid:
                    b_loss_value = b_seg_loss.tolist()
                else:
                    b_loss_value = b_seg_loss[0].tolist()

                loss_list.append(b_loss_value)

                # Print the intermediate results
                self.print_progress(
                    batch_i, n_batches,
                    b_loss_value, np.mean(loss_list)
                )

    def step_validate(
            self,
            dataloader_seg,
    ):

        with torch.no_grad():
            n_batches = len(dataloader_seg)
            losses_list = []
            loss_list = []
            for batch_i, (inputs, output) in enumerate(dataloader_seg):
                if self.hybrid:
                    b_loss = self.batch_step(inputs, output, False)
                    b_loss_value = b_loss.tolist()
                else:
                    b_loss, b_losses = self.batch_step(inputs, output, False)

                    losses_list.append(map(lambda l: l.tolist(), b_losses))
                    b_loss_value = b_loss.tolist()

                loss_list.append(b_loss_value)

                self.print_progress(
                    batch_i, n_batches,
                    b_loss.tolist(), np.mean(loss_list),
                    False
                )

            if self.hybrid:
                loss_value = np.mean(loss_list)
                return loss_value
            else:
                mid_losses = np.mean(zip(*losses_list), axis=1)
                loss_value = np.sum(mid_losses)
                return loss_value, mid_losses

    def batch_step(
            self,
            inputs,
            output,
            train=True
    ):
        # We train the model and check the loss
        b_source = inputs[0].to(self.device)
        b_target = inputs[1].to(self.device)
        b_roi = inputs[2].to(self.device)
        b_target_gt = output[0].to(self.device)
        b_lesion = output[1].to(self.device)

        torch.cuda.synchronize()
        b_pred_lesion, b_moved, b_df = self(
            (b_source, b_target)
        )

        if self.hybrid:
            b_loss = dsc_bin_loss(b_pred_lesion, b_lesion)

            return_loss = b_loss

        else:
            b_dsc_loss = dsc_bin_loss(b_pred_lesion, b_lesion)
            b_reg_losses = self.longitudinal_loss(b_moved, b_target_gt, b_df, b_roi)
            b_losses = b_reg_losses + (b_dsc_loss,)
            b_loss = sum(b_losses)

            return_loss = b_loss, b_losses

        if train:
            self.optimizer_alg.zero_grad()
            b_loss.backward()
            self.optimizer_alg.step()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return return_loss

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss, train=True):
        init_c = '\033[0m' if train else '\033[38;5;238m'
        whites = ' '.join([''] * 12)
        percent = 20 * (batch_i + 1) / n_batches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))
        loss_name = 'train_loss' if train else 'val_loss'
        if self.hybrid and train:
            batch_s = '%s%sEpoch %03d (%03d/%03d) [%s][%s>%s] %s %f (%f)%s' % (
                init_c, whites, self.epoch, batch_i + 1, n_batches,
                ''.join(['-'] * 21), progress_s, remainder_s,
                loss_name, b_loss, mean_loss, '\033[0m'
            )
        else:
            batch_s = '%s%sEpoch %03d (%03d/%03d) [%s>%s] %s %f (%f)%s' % (
                init_c, whites, self.epoch, batch_i + 1, n_batches,
                progress_s, remainder_s,
                loss_name, b_loss, mean_loss, '\033[0m'
            )
        print('\033[K', end='')
        print(batch_s, end='\r')
        sys.stdout.flush()

    def longitudinal_loss(
            self,
            moved,
            target,
            df,
            roi
    ):
        # Init
        moved_roi = moved[roi > 0]
        target_roi = target[roi > 0]

        functions = {
            ' subt ': subtraction_loss,
            ' xcor ': normalised_xcor_loss,
            ' mse  ': torch.nn.MSELoss(),
            'deform': df_loss,
            'modulo': df_modulo,
            ' n_mi ': normalised_mi_loss,
        }

        inputs = {
            ' subt ': (moved, target, roi > 0),
            ' xcor ': (moved_roi, target_roi),
            ' mse  ': (moved_roi, target_roi),
            'deform': (df, roi),
            'modulo': (df, roi),
            ' n_mi ': (moved_roi, target_roi),
        }

        weights = {
            ' subt ': 1.0,
            ' xcor ': 1.0,
            ' mse  ': 1.0,
            'deform': self.lambda_d,
            'modulo': 1.0,
            ' n_mi ': 1.0,
        }

        losses = tuple(
            map(
                lambda l: weights[l] * functions[l](*inputs[l]),
                self.loss_names
            )
        )

        return losses

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))
