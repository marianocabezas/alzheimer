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
from layers import ScalingLayer
from criterions import GenericLossLayer, multidsc_loss
from datasets import WeightedSubsetRandomSampler
from datasets import GenericSegmentationCroppingDataset
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


class BratsSegmentationNet(nn.Module):
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
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
    ):
        super(BratsSegmentationNet, self).__init__()
        # Init
        self.sampler = None
        self.t_train = 0
        self.t_val = 0
        self.epoch = 0
        self.optimizer_alg = None
        self.device = device
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
            nn.Conv3d(filters, 4, 1),
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

    def step(
            self,
            data,
            train=True
    ):
        # We train the model and check the loss
        torch.cuda.synchronize()
        if self.sampler is not None and train:
            x, y, idx = data
            pred_labels = self(x.to(self.device))
            b_losses = torch.stack(
                map(
                    lambda (ypred_i, y_i): multidsc_loss(
                        torch.unsqueeze(ypred_i, 0),
                        torch.unsqueeze(y_i, 0),
                        averaged=train
                    ),
                    zip(pred_labels, y.to(self.device))
                )
            )
            b_loss = torch.mean(b_losses)
            self.sampler.update_weights(b_losses.clone().detach().cpu(), idx)
        else:
            x, y = data
            pred_labels = self(x.to(self.device))
            b_loss = multidsc_loss(
                pred_labels, y.to(self.device), averaged=train
            )

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
        mid_losses = list()
        n_batches = len(training)
        for batch_i, batch_data in enumerate(training):
            # We train the model and check the loss
            batch_loss = self.step(batch_data, train=train)

            if train:
                loss_value = batch_loss.tolist()
            else:
                loss_value = torch.mean(batch_loss).tolist()
                mid_losses.append(batch_loss.tolist())
            losses.append(loss_value)

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses), train
            )

        if self.sampler is not None and train:
            self.sampler.update()

        if train:
            return np.mean(losses)
        else:
            return np.mean(losses), np.mean(zip(*mid_losses), axis=1)

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
            optimizer='adadelta',
            patch_size=32,
            epochs=100,
            patience=10,
            batch_size=32,
            neg_ratio=1,
            num_workers=32,
            sample_rate=1,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.train()

        best_loss_tr = np.inf
        best_loss_val = np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        validation = val_split > 0
        optimizer_dict = {
            'adam': torch.optim.Adam,
            'adadelta': torch.optim.Adadelta,
            'adabound': AdaBound,
        }

        model_params = filter(lambda p: p.requires_grad, self.parameters())

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

            if verbose:
                print(
                    'Training / validation samples = %d / %d' % (
                        n_t_samples, n_v_samples
                    )
                )

            d_train = data[:n_t_samples]
            d_val = data[n_t_samples:]

            t_train = target[:n_t_samples]
            t_val = target[n_t_samples:]

            # Training
            print('Dataset creation')
            use_sampler = sample_rate > 1
            train_dataset = GenericSegmentationCroppingDataset(
                d_train, t_train, patch_size=patch_size,
                neg_ratio=neg_ratio, sampler=use_sampler,
            )
            if use_sampler:
                print('Sampler creation')
                self.sampler = WeightedSubsetRandomSampler(
                    len(train_dataset), sample_rate
                )
                print('Dataloader creation with sampler')
                train_loader = DataLoader(
                    train_dataset, batch_size, num_workers=num_workers,
                    sampler=self.sampler
                )
            else:
                print('Dataloader creation')
                train_loader = DataLoader(
                    train_dataset, batch_size, True, num_workers=num_workers,
                )

            # Validation
            val_dataset = GenericSegmentationCroppingDataset(
                d_val, t_val, patch_size=patch_size, neg_ratio=neg_ratio,
                balanced=False
            )
            val_loader = DataLoader(
                val_dataset, 2 * batch_size, num_workers=num_workers
            )
        else:
            train_dataset = GenericSegmentationCroppingDataset(
                data, target, patch_size=patch_size, neg_ratio=neg_ratio
            )
            self.sampler = WeightedSubsetRandomSampler(
                len(train_dataset), sample_rate
            )
            train_loader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers,
                sampler=self.sampler
            )
            val_loader = DataLoader(
                train_dataset, batch_size, num_workers=num_workers,
            )

        l_names = ['train', ' val ', '  BCK ', '  NET ', '  ED  ', '  ET  ']
        best_losses = [np.inf] * (len(l_names))
        best_e = 0

        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            loss_tr = self.mini_batch_loop(train_loader)
            if loss_tr < best_loss_tr:
                best_loss_tr = loss_tr
                tr_loss_s = '\033[32m%0.5f\033[0m' % loss_tr
            else:
                tr_loss_s = '%0.5f' % loss_tr

            with torch.no_grad():
                self.t_val = time.time()
                loss_val, mid_losses = self.mini_batch_loop(val_loader, False)

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
            improvement = loss_val < best_loss_val
            loss_s = '{:7.3f}'.format(loss_val)
            if improvement:
                best_loss_val = loss_val
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
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.eval()
        whites = ' '.join([''] * 12)
        y_pred = map(lambda d: np.zeros_like(d[0, ...]), data)

        test_set = GenericSegmentationCroppingDataset(
            data, masks=masks, patch_size=patch_size
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

    def segment(
            self,
            data,
            masks,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.eval()
        whites = ' '.join([''] * 12)
        results = []

        with torch.no_grad():
            cases = len(data)
            for i, (data_i, m_i) in enumerate(zip(data, masks)):
                # Print stuff
                if verbose:
                    percent = 20 * (i + 1) / cases
                    progress_s = ''.join(['-'] * percent)
                    remainder_s = ''.join([' '] * (20 - percent))
                    print(
                        '\033[K%sTesting case (%02d/%02d) [%s>%s]' % (
                            whites, i, cases, progress_s, remainder_s
                        ),
                        end='\r'
                    )

                # We test the model with the current batch
                torch.cuda.synchronize()
                pred = self(data_i).tolist()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                results.appent(pred.tolist() * m_i)

        if verbose:
            print('\033[K%sTesting finished succesfully' % whites)

        return results

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class BratsSurvivalNet(nn.Module):
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