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
from layers import ScalingLayer, SpatialTransformer
from criterions import normalized_xcor_loss, subtraction_loss
from criterions import df_modulo, df_gradient_mean
from criterions import dice_loss, histogram_loss, mahalanobis_loss
from datasets import ImageListDataset


def to_torch_var(
        np_array,
        device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        requires_grad=False
):
    var = torch.autograd.Variable(torch.from_numpy(np_array), requires_grad=requires_grad)
    return var.to(device)


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
            y_pred = self(batch_x)
            batch_loss = criterion_alg(y_pred, batch_y)

            loss_value = batch_loss.tolist()
            losses += [loss_value]

            if optimizer_alg is not None:
                percent = 20 * batch_i / n_batches
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
                    y_pred = self(batch_x)
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

                percent = 20 * batch_i / n_t_batches
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
                "cuda:1" if torch.cuda.is_available() else "cpu"
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
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
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
                    percent = 20 * batch_i / n_batches
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
                y_pred += self(batch_x).tolist()

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
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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


class MaskAtrophyNet(nn.Module):
    def __init__(
            self,
            conv_filters=list([32, 64, 64, 64]),
            deconv_filters=list([64, 64, 64, 64, 64, 32, 32]),
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
            lambda_d=1,
            loss_names=list([
                # ' subt ',
                ' xcor ',
                'xcor_l',
                # ' mse  ',
                # 'mask d',
                # 'mahal ',
                # ' hist ',
                'deform',
                # 'modulo'
            ])
    ):
        # Init
        super(MaskAtrophyNet, self).__init__()
        self.lambda_d = lambda_d
        self.loss_names = loss_names
        self.device = device
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
        self.deconv = map(
            lambda (f_in, f_out): nn.ConvTranspose3d(
                f_in, f_out, 3, padding=1
            ),
            zip(
                [deconv_out] + deconv_filters[unet_filters:-1],
                deconv_filters[unet_filters:]
            )
        )
        for d in self.deconv:
            d.to(device)
            nn.init.kaiming_normal_(d.weight)

        # Final DF computation
        self.to_df = nn.Conv3d(deconv_filters[-1], 3, 3, padding=1)
        self.to_df.to(device)
        nn.init.normal_(self.to_df.weight, 0.0, 1e-5)

        self.trans_im = SpatialTransformer()
        self.trans_im.to(device)
        self.trans_mask = SpatialTransformer('nearest')
        self.trans_mask.to(device)

    def forward(self, inputs):

        source, target, mask = inputs
        input_s = torch.cat([source, target], dim=1)

        down_inputs = list()
        for c in self.conv:
            down_inputs.append(input_s)
            input_s = F.leaky_relu(c(input_s), 0.2)

        for d, i in zip(self.deconv_u, down_inputs[::-1]):
            up = F.leaky_relu(d(input_s, output_size=i.size()), 0.2)
            input_s = torch.cat((up, i), dim=1)

        for d in self.deconv:
            input_s = F.leaky_relu(d(input_s), 0.2)

        df = self.to_df(input_s)

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
            val_ratio =4,
            optimizer='adam',
            epochs=100,
            patience=10,
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
            num_workers=10,
            verbose=True
    ):
        # Init
        self.to(device)
        self.train()

        # Optimizer init
        optimizer_dict = {
            'adam': torch.optim.Adam,
        }
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer_alg = optimizer_dict[optimizer](model_params)

        # Pre-loop init
        best_loss_tr = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        t_start = time.time()

        # This is actually a registration approach. It uses the nn framework
        # but it doesn't actually do any supervised training. Therefore, there
        # is no real validation.
        # Due to this, we modified the generic fit algorithm.

        dataset = ImageListDataset(
            cases, masks, brain_masks
        )
        dataloader = DataLoader(
            dataset, batch_size, True, num_workers=num_workers
        )

        l_names = [' loss '] + self.loss_names
        best_losses = [np.inf] * (len(l_names) - 1)
        best_e = 0
        e = 0

        for e in range(epochs):
            # Main epoch loop
            t_in = time.time()
            loss_tr, mid_losses = self.step(
                optimizer_alg,
                e,
                dataloader
            )

            losses_color = map(
                lambda (pl, l): '\033[36m%.4f\033[0m' if l < pl else '%.4f',
                zip(best_losses, mid_losses)
            )
            losses_s = map(lambda (c, l): c % l, zip(losses_color, mid_losses))
            best_losses = map(
                lambda (pl, l): l if l < pl else pl,
                zip(best_losses, mid_losses)
            )

            # Patience check
            improvement = loss_tr < best_loss_tr
            if improvement:
                best_loss_tr = loss_tr
                epoch_s = '\033[32mEpoch %03d\033[0m' % e
                loss_s = '\033[32m%.4f\033[0m' % loss_tr
                best_e = e
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch %03d' % e
                loss_s = '%.4f' % loss_tr
                no_improv_e += 1

            t_out = time.time() - t_in
            t_s = '%.2fs' % t_out

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                if e == 0:
                    l_bars = '-|-'.join(['-' * 6] * len(l_names))
                    l_hdr = ' | '.join(l_names)
                    print('%sEpoch num | %s |' % (whites, l_hdr))
                    print('%s----------|-%s-|' % (whites, l_bars))
                final_s = whites + ' | '.join([epoch_s, loss_s] + losses_s + [t_s])
                print(final_s)

            if no_improv_e == patience:
                break

        self.load_state_dict(best_state)
        t_end = time.time() - t_start
        if verbose:
            print(
                'Registration finished in %d epochs (%fs) with minimum loss = %f (epoch %d)' % (
                    e + 1, t_end, best_loss_tr, best_e)
            )

    def step(
            self,
            optimizer_alg,
            epoch,
            dataloader,
    ):
        # Again. This is supposed to be a step on the registration process,
        # there's no need for splitting data. We just compute the deformation,
        # then compute the global loss (and intermidiate ones to show) and do
        # back propagation.
        with torch.autograd.set_detect_anomaly(True):
            n_data = len(dataloader.dataset)
            batch_size = dataloader.batch_size
            n_batches = int(np.round(1.0 * n_data / batch_size))
            loss_list = []
            for (
                    batch_i,
                    ((b_source, b_target, b_lesion, b_mask), b_gt)
            ) in enumerate(dataloader):
                # We train the model and check the loss
                b_source = b_source.to(self.device)
                b_target = b_target.to(self.device)
                b_lesion = b_lesion.to(self.device)
                b_mask = b_mask.to(self.device)

                b_gt = b_gt.to(self.device)

                b_moved, b_moved_lesion, b_df = self(
                    (b_source, b_target, b_lesion)
                )

                b_losses = self.longitudinal_loss(
                    b_source,
                    b_moved,
                    b_lesion,
                    b_moved_lesion,
                    b_gt,
                    b_df,
                    b_mask
                )

                # Final loss value computation per batch
                batch_loss = sum(b_losses).to(b_target.device)
                b_loss_value = batch_loss.tolist()
                loss_list.append(b_loss_value)
                mean_loss = np.mean(loss_list)

                # Print the intermediate results
                whites = ' '.join([''] * 12)
                percent = 20 * batch_i / n_batches
                progress_s = ''.join(['-'] * percent)
                remainder_s = ''.join([' '] * (20 - percent))
                batch_s = '%sEpoch %03d (%02d/%02d) [%s>%s] loss %f (%f)' % (
                    whites, epoch, batch_i, n_batches,
                    progress_s, remainder_s,
                    b_loss_value, mean_loss
                )
                print('\033[K', end='')
                print(batch_s, end='\r')
                sys.stdout.flush()

                # Backpropagation
                optimizer_alg.zero_grad()
                batch_loss.backward()
                optimizer_alg.step()

            with torch.no_grad():
                losses_list = []
                for (source, target, lesion, mask), gt in dataloader:
                    # We train the model and check the loss
                    source = source.to(self.device)
                    target = target.to(self.device)
                    lesion = lesion.to(self.device)
                    mask = mask.to(self.device)

                    gt = gt.to(self.device)

                    moved, moved_lesion, df = self(
                        (source, target, lesion)
                    )

                    b_losses = self.longitudinal_loss(
                        source,
                        moved,
                        lesion,
                        moved_lesion,
                        gt,
                        df,
                        mask
                    )

                    b_mid_losses = map(lambda l: l.tolist(), b_losses)
                    losses_list.append(b_mid_losses)

            mid_losses = np.mean(zip(*losses_list), axis=1)
            loss_value = np.sum(mid_losses)

        return loss_value, mid_losses

    def transform(
            self,
            source,
            target,
            mask,
            device=torch.device(
                "cuda:1" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.eval()

        source_tensor = to_torch_var(source)
        target_tensor = to_torch_var(target)
        mask_tensor = to_torch_var(mask)

        with torch.no_grad():
            source_mov, mask_mov, df = self(
                (source_tensor, target_tensor, mask_tensor)
            )
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
    ):
        # Init
        moved_lesion = moved[moved_mask > 0]
        target_lesion = target[moved_mask > 0]
        source_lesion = source[mask > 0]
        moved_roi = moved[roi > 0]
        target_roi = target[roi > 0]

        losses_dict = {
            ' subt ': lambda: subtraction_loss(moved_roi, target_roi),
            ' xcor ': lambda: normalized_xcor_loss(moved_roi, target_roi),
            'xcor_l': lambda: normalized_xcor_loss(moved_lesion, target_lesion),
            ' mse  ': lambda: torch.nn.MSELoss()(moved_roi, target_roi),
            'mask d': lambda: dice_loss(moved_mask, mask),
            'mahal ': lambda: mahalanobis_loss(moved_lesion, source_lesion),
            ' hist ': lambda: histogram_loss(moved_lesion, source_lesion),
            'deform': lambda: self.lambda_d * df_gradient_mean(df, roi),
            'modulo': lambda: df_modulo(df, roi),

        }

        functions = {
            ' subt ': subtraction_loss,
            ' xcor ': normalized_xcor_loss,
            'xcor_l': normalized_xcor_loss,
            ' mse  ': torch.nn.MSELoss(),
            'mask d': dice_loss,
            'mahal ': mahalanobis_loss,
            ' hist ': histogram_loss,
            'deform': df_gradient_mean,
            'modulo': df_modulo,

        }

        inputs = {
            ' subt ': (moved_roi, target_roi),
            ' xcor ': (moved_roi, target_roi),
            'xcor_l': (moved_lesion, target_lesion),
            ' mse  ': (moved_roi, target_roi),
            'mask d': (moved_mask, mask),
            'mahal ': (moved_lesion, source_lesion),
            ' hist ': (moved_lesion, source_lesion),
            'deform': (df, roi),
            'modulo': (df, roi),
        }

        weights = {
            ' subt ': 1.0,
            ' xcor ': 1.0,
            'xcor_l': 1.0,
            ' mse  ': 1.0,
            'mask d': 1.0,
            'mahal ': 1.0,
            ' hist ': 1.0,
            'deform': self.lambda_d,
            'modulo': 1.0,
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
