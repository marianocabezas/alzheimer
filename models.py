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
from criterions import GenericLossLayer, multidsc_loss, normalised_xcor_loss
from datasets import ImageListDataset
from datasets import LongitudinalCroppingDataset, ImageListCroppingDataset
from datasets import GenericSegmentationCroppingDataset, get_image
from datasets import WeightedSubsetRandomSampler
from optimizers import AdaBound
from utils import time_to_string


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
    def __init__(
            self,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
    ):
        super(CustomModel, self).__init__()
        self.criterion_alg = None
        self.optimizer_alg = None
        self.sampler = None
        self.epoch = 0
        self.losses = None
        self.device = device

    def forward(self, *inputs):
        pass

    def step(
            self,
            data,
            train=True
    ):
        # We train the model and check the loss
        torch.cuda.synchronize()
        if self.sampler:
            x, y, indices = data
            pred_labels = self(x.to(self.device))
            b_losses = torch.stack(
                map(
                    lambda (y_predi, y_i): self.criterion_alg(
                        torch.unsqueeze(y_predi, 0),
                        torch.unsqueeze(y_i.to(self.device), 0)
                    ),
                    zip(pred_labels, y)
                )
            )
            self.losses[indices] = b_losses.clone().detach()
            b_loss = torch.mean(b_losses)
        else:
            x, y = data
            pred_labels = self(x.to(self.device))
            b_loss = self.criterion_alg(pred_labels, y.to(self.device))

        if train:
            self.optimizer_alg.zero_grad()
            b_loss.backward()
            self.optimizer_alg.step()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # return b_loss
        return b_loss

    def mini_batch_loop(
            self, training, train=True
    ):
        losses = list()
        n_batches = len(training)
        for batch_i, batch_data in enumerate(training):
            # We train the model and check the loss
            batch_loss = self.step(batch_data, train=train)

            loss_value = batch_loss.tolist()
            losses.append(loss_value)

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses), train
            )

        if self.sampler:
            self.sampler.update(self.losses)

        return np.mean(losses)

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss, train=True):
        init_c = '\033[0m' if train else '\033[38;5;238m'
        whites = ' '.join([''] * 12)
        percent = 20 * (batch_i + 1) / n_batches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))
        loss_name = 'train_loss' if train else 'val_loss'

        if train:
            t_out = time.time() - self.t_train
        else:
            t_out = time.time() - self.t_val
        time_s = time_to_string(t_out)

        t_eta = (t_out / (batch_i + 1)) * (n_batches - (batch_i + 1))
        eta_s = time_to_string(t_eta)

        batch_s = '%s%sEpoch %03d (%03d/%03d) [%s>%s] %s %f (%f) %s / ETA %s%s' % (
            init_c, whites, self.epoch, batch_i + 1, n_batches,
            progress_s, remainder_s,
            loss_name, b_loss, mean_loss, time_s, eta_s, '\033[0m'
        )
        print('\033[K', end='')
        print(batch_s, end='\r')
        sys.stdout.flush()

    def fit(
            self,
            data,
            target,
            val_split=0,
            criterion='xentr',
            optimizer='adadelta',
            patch_size=32,
            epochs=100,
            patience=10,
            batch_size=32,
            neg_ratio=1,
            num_workers=32,
            weighted=False,
            sample_rate=5,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.train()

        self.t_train = 0
        self.t_val = 0
        best_e = 0
        self.epoch = 0
        best_loss_tr = np.inf
        best_loss_val = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        validation = val_split > 0

        criterion_dict = {
            'xentr': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            # This is just a hacky way of adding my custom made criterions
            # as "pytorch" modules.
            'dsc': lambda: GenericLossLayer(multidsc_loss)
        }

        optimizer_dict = {
            'adam': torch.optim.Adam,
            'adadelta': torch.optim.Adadelta,
            'adabound': AdaBound,
        }

        model_params = filter(lambda p: p.requires_grad, self.parameters())

        is_string = isinstance(criterion, basestring)

        self.criterion_alg = criterion_dict[criterion]() if is_string else criterion

        is_string = isinstance(optimizer, basestring)

        self.optimizer_alg = optimizer_dict[optimizer](model_params) if is_string\
            else optimizer

        t_start = time.time()

        # Data split (using numpy) for train and validation.
        # We also compute the number of batches for both training and
        # validation according to the batch size.
        if validation:
            n_samples = len(data)

            n_t_samples = int(n_samples * (1 - val_split))
            n_v_samples = n_samples - n_t_samples

            d_train = data[:n_t_samples]
            d_val = data[n_t_samples:]

            t_train = target[:n_t_samples]
            t_val = target[n_t_samples:]

            train_dataset = GenericSegmentationCroppingDataset(
                d_train, t_train, patch_size=patch_size, neg_ratio=neg_ratio,
                preload=True,
            )
            if weighted:
                self.sampler = WeightedSubsetRandomSampler(
                    len(train_dataset), sample_rate
                )
                self.losses = self.sampler.weights.clone().detach()
                train_loader = DataLoader(
                    train_dataset, batch_size, num_workers=num_workers,
                    sampler=self.sampler
                )
            else:
                train_loader = DataLoader(
                    train_dataset, batch_size, True, num_workers=num_workers
                )
            val_dataset = GenericSegmentationCroppingDataset(
                d_val, t_val, patch_size=patch_size, neg_ratio=neg_ratio
            )
            val_loader = DataLoader(
                val_dataset, batch_size, True, num_workers=num_workers
            )
        else:
            train_dataset = GenericSegmentationCroppingDataset(
                data, target, patch_size=patch_size, neg_ratio=neg_ratio
            )
            train_loader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers
            )
            val_loader = None

        if verbose:
            print(
                '%sTraining / validation samples = %d / %d' % (
                    ' '.join([''] * 12), n_t_samples, n_v_samples
                )
            )

        for self.epoch in range(epochs):
            # Main epoch loop
            t_in = time.time()
            self.t_train = time.time()
            loss_tr = self.mini_batch_loop(train_loader)
            # Patience check and validation/real-training loss and accuracy
            improvement = loss_tr < best_loss_tr
            if loss_tr < best_loss_tr:
                best_loss_tr = loss_tr
                loss_s = '\033[32m%0.5f\033[0m' % loss_tr
            else:
                loss_s = '%0.5f' % loss_tr

            if validation:
                with torch.no_grad():
                    self.t_val = time.time()
                    loss_val = self.mini_batch_loop(val_loader, False)

                improvement = loss_val < best_loss_val
                if improvement:
                    best_loss_val = loss_val
                    loss_s += ' | \033[36m%0.5f\033[0m' % loss_val
                    best_e = self.epoch
                else:
                    loss_s += ' | %0.5f' % loss_val

            if improvement:
                best_e = self.epoch
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                no_improv_e += 1

            t_out = time.time() - t_in

            t_out_s = time_to_string(t_out)

            if verbose:
                print('\033[K', end='')
                if self.epoch == 0:
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
                print(
                    '%sEpoch %03d | %s | %s' % (
                        ' '.join([''] * 12), self.epoch, loss_s, t_out_s
                    )
                )

            if no_improv_e == patience:
                self.load_state_dict(best_state)
                break

        t_end = time.time() - t_start
        t_end_s = time_to_string(t_end)
        if verbose:
            print(
                'Training finished in %d epochs (%s) '
                'with minimum loss = %f (epoch %d)' % (
                    self.epoch + 1, t_end_s, best_loss_tr, best_e)
            )

    def predict(
            self,
            data,
            masks,
            patch_size=32,
            batch_size=32,
            num_workers=32,
            overlap=0,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.eval()
        whites = ' '.join([''] * 12)
        y_pred = map(lambda d: np.zeros_like(get_image(d)[0, ...]), data)

        test_set = GenericSegmentationCroppingDataset(
            data, masks=masks, patch_size=patch_size, overlap=overlap
        )
        test_loader = DataLoader(
            test_set, batch_size, num_workers=num_workers
        )

        n_batches = len(test_loader)

        with torch.no_grad():
            for batch_i, (batch_x, cases, slices) in enumerate(test_loader):
                # Print stuff
                if verbose:
                    percent = 20 * (batch_i + 1) / n_batches
                    progress_s = ''.join(['-'] * percent)
                    remainder_s = ''.join([' '] * (20 - percent))
                    print(
                        '\033[K%sTesting batch (%02d/%02d) [%s>%s]' % (
                            whites, batch_i, n_batches, progress_s, remainder_s
                        ),
                        end='\r'
                    )

                # We test the model with the current batch
                torch.cuda.synchronize()
                pred = self(batch_x).tolist()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                for c, slice_i, pred_i in zip(cases, slices, pred):
                    slice_i = tuple(
                        map(
                            lambda (p_ini, p_end): slice(p_ini, p_end), slice_i
                        )
                    )
                    y_pred[c][slice_i] += pred_i.tolist()

        if verbose:
            print('\033[K%sTesting finished succesfully' % whites)

        return y_pred


class BratsSegmentationNet(CustomModel):
    """
    This class is based on the nnUnet (No-New Unet).
    """
    def __init__(
            self,
            filters=16,
            kernel_size=3,
            pool_size=2,
            depth=4,
            n_images=4,
    ):
        super(BratsSegmentationNet, self).__init__()
        # Init
        padding = kernel_size // 2

        # Down path
        self.pooling = pool_size
        filters_list = map(lambda i: filters * 2 ** i, range(depth))
        groups_list = map(
            lambda i: n_images * 2 ** i, range(depth)
        )
        self.convlist = map(
            lambda (ini, out, g): nn.Sequential(
                nn.Conv3d(
                    ini, out, kernel_size,
                    padding=padding,
                    groups=g
                ),
                nn.InstanceNorm3d(out),
                nn.LeakyReLU(),
                nn.Conv3d(
                    out, out, kernel_size,
                    padding=padding,
                    groups=2 * g
                ),
                nn.InstanceNorm3d(out),
                nn.LeakyReLU(),
            ),
            zip([n_images] + filters_list[:-1], filters_list, groups_list)
        )
        for c in self.convlist:
            c.to(self.device)

        self.midconv = nn.Sequential(
            nn.Conv3d(
                filters * (2 ** (depth - 1)),
                filters * (2 ** depth), kernel_size,
                padding=padding
            ),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(),
            nn.Conv3d(
                filters * (2 ** depth),
                filters * (2 ** (depth - 1)), kernel_size,
                padding=padding
            ),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(),
        )
        self.midconv.to(self.device)

        self.deconvlist = map(
            lambda (ini, out, g): nn.Sequential(
                nn.ConvTranspose3d(
                  2 * ini, 2 * ini, 1,
                ),
                nn.ConvTranspose3d(
                    2 * ini, ini, kernel_size,
                    padding=padding,
                    groups=g
                ),
                nn.InstanceNorm3d(filters),
                nn.LeakyReLU(),
                nn.ConvTranspose3d(
                    ini, out, kernel_size,
                    padding=padding,
                    groups=g
                ),
                nn.InstanceNorm3d(filters),
                nn.LeakyReLU(),
            ),
            zip(
                filters_list[::-1], filters_list[-2::-1] + [filters],
                groups_list[::-1]
            )
        )
        for d in self.deconvlist:
            d.to(self.device)

        # Segmentation
        self.out = nn.Sequential(
            nn.Conv3d(filters, 5, 1),
            nn.Softmax(dim=1)
        )
        self.out.to(self.device)

    def forward(self, x):
        down_list = []
        for c in self.convlist:
            down = c(x)
            down_list.append(down)
            x = F.max_pool3d(down, self.pooling)

        x = self.midconv(x)

        for d, prev in zip(self.deconvlist, down_list[::-1]):
            interp = F.interpolate(x, size=prev.shape[2:])
            x = d(torch.cat((prev, interp), dim=1))

        output = self.out(x)
        return output


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


class MultiViewBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels=64,
            pool=2,
            kernels=[3, 5],
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        # Init
        super(MultiViewBlock3D, self).__init__()
        self.out_channels = out_channels
        self.kernels = kernels
        # We need to separate the output channels into pooled ones and normal
        # ones
        conv_channels = out_channels // 2
        pool_channels = out_channels - conv_channels

        # First we create the convolutional channels
        self.conv_filters = self._get_filters_list(conv_channels)
        self.convs = map(
            lambda (f_out, k): nn.Conv3d(in_channels, f_out, k, padding=k // 2),
            zip(self.conv_filters, kernels)
        )
        for c in self.convs:
            c.to(device)

        self.pool_filters = self._get_filters_list(pool_channels)
        self.pool_in = map(
            lambda (f_out, k): nn.Conv3d(in_channels, f_out, k, pool, k // 2),
            zip(self.pool_filters, kernels)
        )
        self.pool_out = map(
            lambda (f_out, k): nn.ConvTranspose3d(f_out, f_out, 1, pool),
            zip(self.pool_filters, kernels)
        )
        for c in self.pool_in:
            c.to(device)
        for d in self.pool_out:
            d.to(device)

    def _get_filters_list(self, channels):
        n_kernels = len(self.kernels)
        n_kernels_1 = n_kernels - 1
        filter_k = int(round(1.0 * channels / n_kernels))
        filters_k = (filter_k,) * n_kernels_1
        filters = filters_k + (channels - n_kernels_1 * filter_k,)

        return filters

    def forward(self, inputs):
        conv_out = map(lambda c: c(inputs), self.convs)
        pool_out = map(
            lambda (c_in, c_out, f): c_out(
                c_in(inputs), output_size=(len(inputs), f) + inputs.shape[2:]
            ),
            zip(self.pool_in, self.pool_out, self.pool_filters)
        )

        return torch.cat(tuple(conv_out + pool_out), dim=1)


class MaskAtrophyNet(nn.Module):
    def __init__(
            self,
            conv_filters=list([32, 64, 64, 64]),
            deconv_filters=list([64, 64, 64, 64, 64, 64]),
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            leakyness=0.2,
            n_images=1,
            kernel_size=3,
            data_smooth=False,
            df_smooth=False,
            trainable_smooth=False,
    ):
        # Init
        super(MaskAtrophyNet, self).__init__()
        self.data_smooth = data_smooth
        self.df_smooth = df_smooth
        self.epoch = 0
        self.optimizer_alg = None
        self.device = device
        self.leakyness = leakyness
        # Down path of the unet
        conv_in = [n_images * 2] + conv_filters[:-2]
        self.conv_u = map(
            lambda (f_in, f_out): nn.Conv3d(
                f_in, f_out, 3, padding=1, groups=n_images
            ),
            zip(conv_in, conv_filters[:-1])
        )
        unet_filters = len(conv_filters) - 1
        for c in self.conv_u:
            c.to(device)
            nn.init.kaiming_normal_(c.weight)

        self.u = nn.Conv3d(conv_filters[-2], conv_filters[-1], 3, padding=1)
        self.u.to(device)
        nn.init.kaiming_normal_(self.u.weight)

        # Up path of the unet
        down_out = conv_filters[-2::-1]
        up_out = [conv_filters[-1]] + deconv_filters[:unet_filters - 1]
        deconv_in = map(sum, zip(down_out, up_out))
        deconv_unet = deconv_filters[:unet_filters]
        self.deconv_u = map(
            lambda (f_in, f_out): nn.ConvTranspose3d(
                f_in, f_out, 3, padding=1,
            ),
            zip(deconv_in, deconv_unet)
        )
        for d in self.deconv_u:
            d.to(device)
            nn.init.kaiming_normal_(d.weight)

        # Extra DF path
        deconv_out = deconv_unet[-1]
        extra_filters = deconv_filters[unet_filters:]
        final_filters = deconv_filters[-1]
        pad = kernel_size // 2
        self.conv = map(
            lambda (f_in, f_out): nn.Conv3d(
                f_in, f_out, kernel_size, padding=pad,
            ),
            zip(
                [deconv_out] + extra_filters[:-1],
                extra_filters
            )
        )
        for c in self.conv:
            c.to(device)
            nn.init.kaiming_normal_(c.weight)

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

    def forward(self, patch_source, target, mask=None, mesh=None, source=None):
        data = torch.stack(
            map(
                lambda (s, t): torch.cat([s, t]),
                zip(patch_source, target)
            )
        )

        if self.data_smooth:
            data = self.smooth(data)

        down_inputs = list()
        for c in self.conv_u:
            data = F.leaky_relu(c(data), self.leakyness)
            down_inputs.append(data)
            data = F.max_pool3d(data, 2)

        data = F.leaky_relu(self.u(data), self.leakyness)

        for d, i in zip(self.deconv_u, down_inputs[::-1]):
            data = torch.cat(
                [F.interpolate(data, size=i.size()[2:]), i], dim=1
            )
            data = F.leaky_relu(
                d(data), self.leakyness
            )

        for c in self.conv:
            data = F.leaky_relu(c(data), self.leakyness)

        df = F.leaky_relu(self.to_df(data), self.leakyness)

        if self.df_smooth:
            df = self.smooth(df)

        if source is not None and mesh is not None:
            source_mov = self.trans_im(
                [source, df, mesh]
            )

        else:
            source_mov = self.trans_im(
                [patch_source, df]
            )

        if mask is not None:
            if mesh is not None:
                mask_mov = self.trans_mask(
                    [mask, df, mesh]
                )
            else:
                mask_mov = self.trans_mask(
                    [mask, df]
                )
            return source_mov, mask_mov, df
        else:
            return source_mov, df

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
            max_epochs = epochs * max_step
            overlap = patch_size * 3 // 4
        else:
            max_epochs = epochs
            curr_step = None
            overlap = 8

        if patch_based:
            tr_dataset = ImageListCroppingDataset(
                cases, masks, brain_masks,
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

        for self.epoch in range(max_epochs):
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

            step_done = curriculum and (self.epoch == (curr_step * epochs - 1))
            if no_improv_e == patience or step_done:
                # If we are going to use curriculum learning, once we surpass
                # the patience value, we'll see if we can increase the
                # difficulty.
                # That means changing the dataloader for a new one with a
                # bigger step between timepoints.
                if curriculum and (curr_step < max_step):
                    # Print the end of a step ("era")
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
                            cases, masks, brain_masks,
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

            b_moved, b_moved_lesion, b_df = self(
                b_source, b_target, b_m, b_mesh, b_im
            )

        else:
            b_source, b_target, b_lesion, b_mask = inputs
            b_source = b_source.to(self.device)
            b_target = b_target.to(self.device)
            b_lesion = b_lesion.to(self.device)
            b_mask = b_mask.to(self.device)

            b_gt = output.to(self.device)

            b_moved, b_moved_lesion, b_df = self(b_source, b_target, b_lesion)

        torch.cuda.synchronize()

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

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class NewLesionsUNet(nn.Module):
    def __init__(
            self,
            conv_filters=list([32, 64, 64, 64, 64]),
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=1,
    ):
        super(NewLesionsUNet, self).__init__()
        # Init
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.optimizer_alg = None
        self.device = device

        # Down path of the unet
        conv_in = [n_images * 2] + conv_filters[:-2]

        self.down = map(
            lambda (f_in, f_out): nn.Conv3d(
                f_in, f_out, 3, padding=1,
            ),
            zip(conv_in, conv_filters[:-1])
        )
        for c in self.down:
            c.to(device)

        self.u = nn.Conv3d(
            conv_filters[-2], conv_filters[-1], 3, padding=1
        )
        self.u.to(self.device)

        # Up path of the unet
        down_out = conv_filters[-2::-1]
        up_out = conv_filters[:0:-1]
        deconv_in = map(sum, zip(down_out, up_out))
        self.up = map(
            lambda (f_in, f_out): nn.ConvTranspose3d(
                f_in, f_out, 3, padding=1
            ),
            zip(
                deconv_in,
                down_out
            )
        )
        for d in self.up:
            d.to(device)

        self.seg = nn.Sequential(
            nn.Conv3d(conv_filters[0], conv_filters[0], 1),
            nn.Conv3d(conv_filters[0], 2, 1)
        )
        self.seg.to(device)

    def forward(self, source, target):
        # input_s = torch.cat([source, target], dim=1)
        input_s = torch.stack(
            map(
                lambda (s, t): torch.cat([s, t]),
                zip(source, target)
            )
        )
        down_inputs = [input_s]
        for c in self.down:
            input_s = F.relu(c(input_s))
            down_inputs.append(input_s)
            input_s = F.max_pool3d(input_s, 2)

        input_s = F.relu(self.u(input_s))

        for d, i in zip(self.up, down_inputs[::-1]):
            input_s = torch.cat(
                (F.interpolate(input_s, size=i.size()[2:]), i), dim=1
            )
            input_s = F.relu(d(input_s))

        multi_seg = torch.softmax(self.seg(input_s), dim=1)

        return multi_seg

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
            seg = map(
                lambda (s, t): self(
                    torch.unsqueeze(s, dim=0), torch.unsqueeze(t, dim=0)
                ),
                zip(source_tensor, target_tensor)
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if verbose:
            print(
                '\033[K%sTransformation finished' % ' '.join([''] * 12)
            )

        seg = map(lambda s: np.squeeze(s.cpu().numpy()), seg)
        return seg

    def fit(
            self,
            source,
            target,
            new_lesion,
            masks=None,
            patch_size=32,
            val_split=0,
            batch_size=32,
            optimizer='adadelta',
            epochs=100,
            patience=10,
            num_workers=32,
            verbose=True
    ):
        # Init
        self.train()

        # Optimizer init
        optimizer_dict = {
            'adam': lambda param: torch.optim.Adam(param, lr=1e-2),
            'adabound': AdaBound,
            'adadelta': torch.optim.Adadelta
        }
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = optimizer_dict[optimizer](model_params)

        # Pre-loop init
        best_loss_tr = np.inf
        best_loss_val = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        t_start = time.time()

        validation = val_split > 0

        if validation:
            n_samples = len(source)

            n_t_samples = int(n_samples * (1 - val_split))

            s_train = source[:n_t_samples]
            s_val = source[n_t_samples:]

            t_train = target[:n_t_samples]
            t_val = target[n_t_samples:]

            l_train = new_lesion[:n_t_samples]
            l_val = new_lesion[n_t_samples:]

            if masks is None:
                m_train = None
                m_val = None
            else:
                m_train = masks[:n_t_samples]
                m_val = masks[n_t_samples:]

            train_dataset = LongitudinalCroppingDataset(
                s_train, t_train, l_train, m_train, patch_size=patch_size,
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers
            )

            val_dataset = LongitudinalCroppingDataset(
                s_val, t_val, l_val, m_val, patch_size=patch_size,
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size, num_workers=num_workers
            )
        else:
            train_dataset = LongitudinalCroppingDataset(
                source, target, new_lesion, masks, patch_size=patch_size,
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers
            )

            val_dataset = LongitudinalCroppingDataset(
                source, target, new_lesion, masks, patch_size=patch_size
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size, num_workers=num_workers
            )

        l_names = ['train', ' val ', '  bck ', '  les ']
        best_losses = [np.inf] * (len(l_names))
        best_e = 0
        e = 0

        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            tr_loss_value = self.step_train(train_dataloader)
            loss_s = '{:7.3f}'.format(tr_loss_value)
            if tr_loss_value < best_loss_tr:
                best_loss_tr = tr_loss_value
                tr_loss_s = '\033[32;1m%s\033[0m' % loss_s
            else:
                tr_loss_s = '%s' % loss_s

            self.t_val = time.time()
            loss_value, mid_losses = self.step_validate(val_dataloader)

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
            improvement = loss_value < best_loss_val
            loss_s = '{:7.3f}'.format(loss_value)
            if improvement:
                best_loss_val = loss_value
                epoch_s = '\033[32mEpoch %03d\033[0m' % self.epoch
                loss_s = '\033[32m%s\033[0m' % loss_s
                best_e = self.epoch
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch %03d' % self.epoch
                no_improv_e += 1

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                if self.epoch == 0:
                    l_bars = '--|--'.join(
                        ['-' * 5] * 2 + ['-' * 6] * len(l_names[2:])
                    )
                    l_hdr = '  |  '.join(l_names)
                    print('%sEpoch num |  %s  |' % (whites, l_hdr))
                    print('%s----------|--%s--|' % (whites, l_bars))
                final_s = whites + ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] + losses_s + [t_s]
                )
                print(final_s)

            if no_improv_e == patience:
                break

        self.epoch = best_e
        self.load_state_dict(best_state)
        t_end = time.time() - t_start
        if verbose:
            out_s = 'Segmentation finished in %d epochs (%fs) ' \
                    'with minimum loss = %f (epoch %d)'
            print(
                out_s % (
                    self.e + 1, t_end, best_loss_val, best_e
                )
            )

    def step_train(
            self,
            dataloader_seg,
    ):
        # This step should combine both registration and segmentation.
        # The goal is to affect the deformation with two datasets and different
        # goals and loss functions.
        with torch.autograd.set_detect_anomaly(True):
            # Segmentation update
            n_batches = len(dataloader_seg)
            loss_list = []
            for batch_i, (inputs, output) in enumerate(dataloader_seg):
                b_seg_loss, _ = self.batch_step(inputs, output)

                b_loss_value = b_seg_loss.tolist()

                loss_list.append(b_loss_value)

                # Print the intermediate results
                self.print_progress(
                    batch_i, n_batches,
                    b_loss_value, np.mean(loss_list)
                )

        return np.mean(loss_list)

    def step_validate(
            self,
            dataloader_seg,
    ):

        with torch.no_grad():
            n_batches = len(dataloader_seg)
            losses_list = []
            loss_list = []
            for batch_i, (inputs, output) in enumerate(dataloader_seg):
                b_loss, b_losses = self.batch_step(inputs, output, False)

                losses_list.append(map(lambda l: l.tolist(), b_losses))
                b_loss_value = b_loss.tolist()

                loss_list.append(b_loss_value)

                self.print_progress(
                    batch_i, n_batches,
                    b_loss.tolist(), np.mean(loss_list),
                    False
                )

            mid_losses = np.mean(zip(*losses_list), axis=1)
            loss_value = np.mean(loss_list)
            return loss_value, mid_losses

    def batch_step(
            self,
            inputs,
            outputs,
            train=True,
    ):
        # We train the model and check the loss
        b_inputs = map(lambda b_i: b_i.to(self.device), inputs)[:2]
        b_lesion = outputs[0].to(self.device)

        torch.cuda.synchronize()
        b_pred_lesion = self(*b_inputs)

        b_dsc_losses = multidsc_loss(
            b_pred_lesion, b_lesion, averaged=False
        )
        if train:
            sum_class = map(lambda c: torch.sum(b_lesion == c), range(2))
            b_loss = sum(
                map(
                    lambda (loss, s): loss * s / sum(sum_class),
                    zip(b_dsc_losses, sum_class[::-1])
                )
            )
            self.optimizer_alg.zero_grad()
            b_loss.backward()
            self.optimizer_alg.step()
        else:
            b_loss = torch.mean(b_dsc_losses)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return b_loss, b_dsc_losses

    def print_progress(
            self, batch_i, n_batches, b_loss, mean_loss, train=True
    ):
        init_c = '\033[0m' if train else '\033[38;5;238m'
        whites = ' '.join([''] * 12)
        percent = 20 * (batch_i + 1) / n_batches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))
        loss_name = 'train_loss' if train else 'val_loss'
        if train:
            t_out = time.time() - self.t_train
        else:
            t_out = time.time() - self.t_val
        time_s = time_to_string(t_out)

        t_eta = (t_out / (batch_i + 1)) * (n_batches - (batch_i + 1))
        eta_s = time_to_string(t_eta)

        b_s = '%s%sEpoch %03d (%03d/%03d) [%s>%s] %s %f (%f) %s / ETA: %s %s'

        batch_s = b_s % (
            init_c, whites, self.epoch, batch_i + 1, n_batches,
            progress_s, remainder_s,
            loss_name, b_loss, mean_loss, time_s, eta_s, '\033[0m'
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
            conv_filters_s=list([32, 64, 64, 64, 64]),
            conv_filters_r=list([32, 64, 64, 64]),
            deconv_filters_r=list([64, 64, 64, 32, 32]),
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            leakyness=0.2,
            n_images=1,
            data_smooth=False,
            df_smooth=False,
            trainable_smooth=False,
    ):
        super(NewLesionsNet, self).__init__()
        # Init
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.optimizer_alg = None
        self.atrophy = MaskAtrophyNet(
            conv_filters=conv_filters_r,
            deconv_filters=deconv_filters_r,
            device=device,
            leakyness=leakyness,
            n_images=n_images,
            data_smooth=data_smooth,
            df_smooth=df_smooth,
            trainable_smooth=trainable_smooth
        )
        self.device = device

        # Down path of the unet
        conv_in = [conv_filters_s[0] * 2] + conv_filters_s[1:-2]
        init_out = conv_filters_s[0]
        self.init_df = nn.Conv3d(3, init_out, 3, padding=1)
        self.init_df.to(device)
        self.init_im = nn.Conv3d(
            n_images * 2, init_out, 3, padding=1, groups=n_images
        )
        self.init_im.to(device)

        self.down = map(
            lambda (f_in, f_out): nn.Conv3d(
                f_in, f_out, 3, padding=1, groups=2
            ),
            zip(conv_in, conv_filters_s[1:-1])
        )
        for c in self.down:
            c.to(device)

        self.u = nn.Conv3d(
            conv_filters_s[-2], conv_filters_s[-1], 3, padding=1,
        )
        self.u.to(self.device)

        # Up path of the unet
        down_out = conv_filters_s[-2:0:-1] + [conv_filters_s[0] * 2]
        up_out = conv_filters_s[:0:-1]
        deconv_in = map(sum, zip(down_out, up_out))
        self.up = map(
            lambda (f_in, f_out): nn.ConvTranspose3d(
                f_in, f_out, 3, padding=1, groups=2
            ),
            zip(
                deconv_in,
                conv_filters_s[-2::-1]
            )
        )
        for d in self.up:
            d.to(device)

        self.seg = nn.Sequential(
            nn.Conv3d(conv_filters_s[0], conv_filters_s[0], 1),
            nn.Conv3d(conv_filters_s[0], 2, 1)
        )
        self.seg.to(device)

    def forward(self, patch_source, target, mesh=None, source=None):
        # Atrophy network
        source_mov, df = self.atrophy(
            patch_source, target, mesh=mesh, source=source
        )

        data = torch.stack(
            map(
                lambda (s, t): torch.cat([s, t]),
                zip(patch_source, target)
            )
        )
        # Now we actually need to give a segmentation result.
        input_df = F.relu(self.init_df(df))
        input_im = F.relu(self.init_im(data))
        input_s = torch.cat([input_im, input_df], dim=1)
        down_inputs = [input_s]
        for c in self.down:
            input_s = F.relu(c(input_s))
            down_inputs.append(input_s)
            input_s = F.max_pool3d(input_s, 2)

        input_s = F.relu(self.u(input_s))

        for d, i in zip(self.up, down_inputs[::-1]):
            input_s = torch.cat(
                (F.interpolate(input_s, size=i.size()[2:]), i), dim=1
            )
            input_s = F.relu(d(input_s))

        multi_seg = torch.softmax(self.seg(input_s), dim=1)

        return multi_seg, source_mov, df

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
                source_tensor, target_tensor
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
            masks=None,
            patch_size=32,
            val_split=0,
            batch_size=32,
            optimizer='adadelta',
            epochs=100,
            patience=10,
            num_workers=32,
            verbose=True
    ):
        # Init
        self.train()

        # Optimizer init
        optimizer_dict = {
            'adam': lambda param: torch.optim.Adam(param, lr=1e-2),
            'adabound': AdaBound,
            'adadelta': torch.optim.Adadelta
        }
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = optimizer_dict[optimizer](model_params)

        # Pre-loop init
        best_loss_tr = np.inf
        best_loss_val = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        t_start = time.time()

        validation = val_split > 0

        if validation:
            n_samples = len(source)

            n_t_samples = int(n_samples * (1 - val_split))

            s_train = source[:n_t_samples]
            s_val = source[n_t_samples:]

            t_train = target[:n_t_samples]
            t_val = target[n_t_samples:]

            l_train = new_lesion[:n_t_samples]
            l_val = new_lesion[n_t_samples:]

            if masks is None:
                m_train = None
                m_val = None
            else:
                m_train = masks[:n_t_samples]
                m_val = masks[n_t_samples:]

            train_dataset = LongitudinalCroppingDataset(
                s_train, t_train, l_train, m_train, patch_size=patch_size,
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers
            )

            val_dataset = LongitudinalCroppingDataset(
                s_val, t_val, l_val, m_val, patch_size=patch_size,
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size, num_workers=num_workers
            )
        else:
            train_dataset = LongitudinalCroppingDataset(
                source, target, new_lesion, masks, patch_size=patch_size,
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers
            )

            val_dataset = LongitudinalCroppingDataset(
                source, target, new_lesion, masks, patch_size=patch_size
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size, num_workers=num_workers
            )

        l_names = ['train', ' val ', ' xcor ', '  bck ', '  les ']
        # l_names = [' train', ' loss ', '  mse ', '  dsc ']
        # l_names = [' train', ' loss ', ' subt ', '  dsc ']
        best_losses = [np.inf] * (len(l_names))
        best_e = 0
        e = 0

        for self.epoch in range(epochs):
            self.atrophy.epoch = self.epoch
            # Main epoch loop
            self.t_train = time.time()
            tr_loss_value = self.step_train(dataloader=train_dataloader)
            loss_s = '{:7.3f}'.format(tr_loss_value)
            if tr_loss_value < best_loss_tr:
                best_loss_tr = tr_loss_value
                tr_loss_s = '\033[32;1m%s\033[0m' % loss_s
            else:
                tr_loss_s = '%s' % loss_s

            self.t_val = time.time()
            loss_value, mid_losses = self.step_validate(val_dataloader)

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
            improvement = loss_value < best_loss_val
            loss_s = '{:7.3f}'.format(loss_value)
            if improvement:
                best_loss_val = loss_value
                epoch_s = '\033[32mEpoch %03d\033[0m' % self.epoch
                loss_s = '\033[32m%s\033[0m' % loss_s
                best_e = self.epoch
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch %03d' % self.epoch
                no_improv_e += 1

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                if self.epoch == 0:
                    l_bars = '--|--'.join(
                        ['-' * 5] * 2 + ['-' * 6] * len(l_names[2:])
                    )
                    l_hdr = '  |  '.join(l_names)
                    print('%sEpoch num |  %s  |' % (whites, l_hdr))
                    print('%s----------|--%s--|' % (whites, l_bars))
                final_s = whites + ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] + losses_s + [t_s]
                )
                print(final_s)

            if no_improv_e == patience:
                break

        self.epoch = best_e
        self.load_state_dict(best_state)
        t_end = time.time() - t_start
        if verbose:
            out_s = 'Registration finished in %d epochs (%fs) ' \
                    'with minimum loss = %f (epoch %d)'
            print(
                out_s % (
                    self.e + 1, t_end, best_loss_val, best_e)
            )

    def step_train(
            self,
            dataloader,
    ):
        # This step should combine both registration and segmentation.
        # The goal is to affect the deformation with two datasets and different
        # goals and loss functions.
        with torch.autograd.set_detect_anomaly(True):
            # Segmentation update
            n_batches = len(dataloader)
            loss_list = []
            for batch_i, (inputs, output) in enumerate(dataloader):
                b_loss, _ = self.batch_step(inputs, output)

                b_loss_value = b_loss.tolist()

                loss_list.append(b_loss_value)

                # Print the intermediate results
                self.print_progress(
                    batch_i, n_batches,
                    b_loss_value, np.mean(loss_list)
                )
        return np.mean(loss_list)

    def step_validate(
            self,
            dataloader_seg,
    ):

        with torch.no_grad():
            n_batches = len(dataloader_seg)
            losses_list = []
            loss_list = []
            for batch_i, (inputs, output) in enumerate(dataloader_seg):
                b_loss, b_losses = self.batch_step(inputs, output, False)

                losses_list.append(map(lambda l: l.tolist(), b_losses))
                b_loss_value = b_loss.tolist()

                loss_list.append(b_loss_value)

                self.print_progress(
                    batch_i, n_batches,
                    b_loss.tolist(), np.mean(loss_list),
                    False
                )

            mid_losses = np.mean(zip(*losses_list), axis=1)
            loss_value = np.mean(loss_list)
            return loss_value, mid_losses

    def batch_step(
            self,
            inputs,
            outputs,
            train=True
    ):
        # We train the model and check the loss
        b_inputs = map(lambda b_i: b_i.to(self.device), inputs)
        b_lesion = outputs[0].to(self.device)
        b_target = outputs[1].to(self.device)

        torch.cuda.synchronize()
        b_pred_lesion, b_moved, b_df = self(*b_inputs)

        b_dsc_losses = multidsc_loss(
            b_pred_lesion, b_lesion, averaged=False
        )
        b_reg_loss = normalised_xcor_loss(b_moved, b_target)
        # b_reg_loss = subtraction_loss(b_moved, b_target, roi)
        if train:
            sum_class = map(lambda c: torch.sum(b_lesion == c), range(2))
            b_dsc_loss = sum(
                map(
                    lambda (loss, s): loss * s / sum(sum_class),
                    zip(b_dsc_losses, sum_class[::-1])
                )
            )
            b_losses = (b_reg_loss, b_dsc_loss)
            b_loss = sum(b_losses)
            self.optimizer_alg.zero_grad()
            b_loss.backward()
            self.optimizer_alg.step()
        else:
            b_losses = (b_reg_loss,) + tuple(map(lambda l: l, b_dsc_losses))
            b_loss = b_reg_loss + torch.mean(b_dsc_losses)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return b_loss, b_losses

    def print_progress(
            self, batch_i, n_batches, b_loss, mean_loss, train=True
    ):
        init_c = '\033[0m' if train else '\033[38;5;238m'
        whites = ' '.join([''] * 12)
        percent = 20 * (batch_i + 1) / n_batches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))
        loss_name = 'train_loss' if train else 'val_loss'
        if train:
            t_out = time.time() - self.t_train
        else:
            t_out = time.time() - self.t_val
        time_s = time_to_string(t_out)

        t_eta = (t_out / (batch_i + 1)) * (n_batches - (batch_i + 1))
        eta_s = time_to_string(t_eta)

        b_s = '%s%sEpoch %03d (%03d/%03d) [%s>%s] %s %f (%f) %s / ETA: %s %s'

        batch_s = b_s % (
            init_c, whites, self.epoch, batch_i + 1, n_batches,
            progress_s, remainder_s,
            loss_name, b_loss, mean_loss, time_s, eta_s, '\033[0m'
        )
        print('\033[K', end='')
        print(batch_s, end='\r')
        sys.stdout.flush()

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))
