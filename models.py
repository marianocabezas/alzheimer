from __future__ import print_function
import time
import sys
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from criterions import multidsc_loss
from datasets import WeightedSubsetRandomSampler
from datasets import GenericSegmentationCroppingDataset
from datasets import BoundarySegmentationCroppingDataset
from datasets import BBImageDataset, BBImageTupleDataset, BBImageValueDataset
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
            filters=32,
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
        filter_list = map(lambda i: filters * 2 ** i, range(depth))
        groups_list = map(
            lambda i: n_images * 2 ** i, range(depth)
        )
        self.convlist = map(
            lambda (ini, out, g): nn.Sequential(
                nn.Conv3d(
                    ini, out, kernel_size,
                    padding=padding,
                    # groups=g
                ),
                nn.LeakyReLU(),
                # nn.ReLU(),
                nn.InstanceNorm3d(out),
                # nn.BatchNorm3d(out),
                nn.Conv3d(
                    out, out, kernel_size,
                    padding=padding,
                    # groups=2 * g
                ),
                nn.LeakyReLU(),
                # nn.ReLU(),
                nn.InstanceNorm3d(out),
                # nn.BatchNorm3d(out),
            ),
            zip([n_images] + filter_list[:-1], filter_list, groups_list)
        )
        # self.pooling = map(
        #     lambda f: nn.Conv3d(f, f, pool_size, stride=pool_size, groups=f),
        #     filter_list
        # )
        self.pooling = [nn.AvgPool3d(pool_size)] * len(filter_list)
        for c, p in zip(self.convlist, self.pooling):
            c.to(self.device)
            p.to(self.device)

        self.midconv = nn.Sequential(
            nn.Conv3d(
                filters * (2 ** (depth - 1)),
                filters * (2 ** depth), kernel_size,
                padding=padding
            ),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.InstanceNorm3d(filters * (2 ** depth)),
            # nn.BatchNorm3d(filters * (2 ** depth)),
            nn.Conv3d(
                filters * (2 ** depth),
                filters * (2 ** (depth - 1)), kernel_size,
                padding=padding
            ),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.InstanceNorm3d(filters * (2 ** (depth - 1))),
            # nn.BatchNorm3d(filters * (2 ** (depth - 1))),
        )
        self.midconv.to(self.device)

        self.deconvlist = map(
            lambda (ini, out, g): nn.Sequential(
                nn.ConvTranspose3d(
                    2 * ini, ini, kernel_size,
                    padding=padding,
                    # groups=g
                ),
                nn.LeakyReLU(),
                # nn.ReLU(),
                nn.InstanceNorm3d(ini),
                # nn.BatchNorm3d(ini),
                nn.ConvTranspose3d(
                    ini, out, kernel_size,
                    padding=padding,
                    # groups=g
                ),
                nn.LeakyReLU(),
                # nn.ReLU(),
                nn.InstanceNorm3d(out),
                # nn.BatchNorm3d(out),
            ),
            zip(
                filter_list[::-1], filter_list[-2::-1] + [filters],
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

    def forward(self, x, dropout=0):
        down_list = []
        for c, p in zip(self.convlist, self.pooling):
            down = c(x)
            if dropout > 0:
                down = nn.functional.dropout3d(down, dropout)
            down_list.append(down)
            # x = F.max_pool3d(down, self.pooling)
            x = p(down)

        x = self.midconv(x)

        if dropout > 0:
            x = nn.functional.dropout3d(x, dropout)

        for d, prev in zip(self.deconvlist, down_list[::-1]):
            interp = F.interpolate(x, size=prev.shape[2:])
            x = d(torch.cat((prev, interp), dim=1))
            if dropout > 0:
                x = nn.functional.dropout3d(x, dropout)

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
        for batch_i, (x, y) in enumerate(training):
            # We train the model and check the loss
            torch.cuda.synchronize()
            pred_labels = self(x.to(self.device))
            batch_loss = multidsc_loss(
                pred_labels, y.to(self.device), averaged=train
            )
            if train:
                self.optimizer_alg.zero_grad()
                batch_loss.backward()
                self.optimizer_alg.step()
                loss_value = batch_loss.tolist()
            else:
                loss_value = torch.mean(batch_loss).tolist()
                dsc = 1 - batch_loss
                mid_losses.append(dsc.tolist())

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            losses.append(loss_value)

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses), train
            )

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
            rois=None,
            val_split=0,
            optimizer='adadelta',
            patch_size=None,
            epochs=100,
            patience=10,
            batch_size=1,
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

        if val_split <= 0:
            val_split = 0.1
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
        use_sampler = sample_rate > 1
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

        if rois is not None:
            r_train = rois[:n_t_samples]
            r_val = rois[n_t_samples:]
        else:
            r_train = None
            r_val = None

        # Training
        if patch_size is None:
            # Full image one
            print('Dataset creation images <with validation>')
            train_dataset = BBImageDataset(
                d_train, t_train, r_train, sampler=use_sampler
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

        l_names = ['train', ' val ', '  BCK ', '  NET ', '  ED  ', '  ET  ']
        best_losses = [-np.inf] * (len(l_names))
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
                lambda (pl, l): '\033[36m%s\033[0m' if l > pl else '%s',
                zip(best_losses, mid_losses)
            )
            losses_s = map(
                lambda (c, l): c % '{:8.4f}'.format(l),
                zip(losses_color, mid_losses)
            )
            best_losses = map(
                lambda (pl, l): l if l > pl else pl,
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
                if self.epoch > sample_rate:
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

    def segment(
            self,
            data,
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
            t_in = time.time()
            for i, data_i in enumerate(data):

                # We test the model with the current batch
                input_i = torch.unsqueeze(
                    to_torch_var(data_i, self.device), 0
                )
                torch.cuda.synchronize()
                pred = self(input_i).squeeze().tolist()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                results.append(pred)

                t_out = time.time() - t_in
                t_s = time_to_string(t_out)
                # Print stuff
                if verbose:
                    percent = 20 * (i + 1) / cases
                    progress_s = ''.join(['-'] * percent)
                    remainder_s = ''.join([' '] * (20 - percent))
                    t_eta = (t_out / (i + 1)) * (cases - (i + 1))
                    eta_s = time_to_string(t_eta)
                    print(
                        '\033[K%sTested case (%02d/%02d) [%s>%s]'
                        ' %s / ETA: %s' % (
                            whites, i, cases, progress_s, remainder_s,
                            t_s, eta_s
                        ),
                        end='\r'
                    )
                    sys.stdout.flush()

        if verbose:
            print('\033[K%sTesting finished succesfully' % whites)

        return results

    def uncertainty(
            self,
            data,
            dropout=0.5,
            steps=100,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.eval()
        whites = ' '.join([''] * 12)
        entropy_results = []
        seg_results = []

        with torch.no_grad():
            cases = len(data)
            t_in = time.time()
            for i, data_i in enumerate(data):
                outputs = []
                for e in range(steps):
                    # We test the model with the current batch
                    input_i = torch.unsqueeze(
                        to_torch_var(data_i, self.device), 0
                    )
                    torch.cuda.synchronize()
                    pred = self(input_i, dropout).squeeze().tolist()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    outputs.append(pred)

                    # Print stuff
                    if verbose:
                        percent_i = 20 * (i + 1) / cases
                        percent_e = 20 * (e + 1) / steps
                        progress_is = ''.join(['-'] * percent_i)
                        progress_es = ''.join(['-'] * percent_e)
                        remainder_is = ''.join([' '] * (20 - percent_i))
                        remainder_es = ''.join([' '] * (20 - percent_e))
                        remaining_e = steps - (e + 1)
                        remaining_i = steps * (cases - (i + 1))
                        completed = steps * cases - (remaining_i + remaining_e)
                        t_out = time.time() - t_in
                        t_out_e = t_out / completed
                        t_s = time_to_string(t_out)
                        t_eta = (remaining_e + remaining_i) * t_out_e
                        eta_s = time_to_string(t_eta)
                        print(
                            '\033[K%sTested case (%02d/%02d - %02d/%02d) '
                            '[%s>%s][%s>%s] %s / ETA: %s' % (
                                whites, e, steps, i, cases,
                                progress_es, remainder_es,
                                progress_is, remainder_is,
                                t_s, eta_s
                            ),
                            end='\r'
                        )
                        sys.stdout.flush()

                mean_output = np.mean(outputs, axis=0)
                entropy = - np.sum(mean_output * np.log(mean_output), axis=0)

                entropy_results.append(entropy)
                seg_results.append(mean_output)

        if verbose:
            print('\033[K%sTesting finished succesfully' % whites)

        return entropy_results, seg_results

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class BratsSegmentationHybridNet(nn.Module):
    """
    This class is based on the nnUnet (No-New Unet).
    """

    def __init__(
            self,
            filters=24,
            kernel_size=3,
            pool_size=2,
            depth=4,
            n_images=4,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
    ):
        super(BratsSegmentationHybridNet, self).__init__()
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
        filter_list = map(lambda i: filters * 2 ** i, range(depth))
        self.convlist = map(
            lambda (ini, out): nn.Sequential(
                nn.Conv3d(
                    ini, out, kernel_size,
                    padding=padding,
                    # groups=n_images
                ),
                nn.ReLU(),
                nn.Dropout3d(),
                nn.InstanceNorm3d(out),
                nn.Conv3d(
                    out, out, 1,
                    padding=padding,
                    groups=out
                ),
                nn.ReLU(),
                nn.Conv3d(
                    out, out, kernel_size,
                    padding=padding,
                    # groups=n_images
                ),
                nn.ReLU(),
                nn.Dropout3d(),
                nn.InstanceNorm3d(out),
            ),
            zip([n_images] + filter_list[:-1], filter_list)
        )
        self.pooling = map(
            lambda f: nn.Conv3d(f, f, pool_size, stride=pool_size, groups=f),
            filter_list
        )
        for c, p in zip(self.convlist, self.pooling):
            c.to(self.device)
            p.to(self.device)

        self.midconv = nn.Sequential(
            nn.Conv3d(
                filters * (2 ** (depth - 1)),
                filters * (2 ** depth), kernel_size,
                padding=padding,
                # groups=n_images
            ),
            nn.ReLU(),
            nn.Dropout3d(),
            nn.InstanceNorm3d(filters * (2 ** depth)),
            nn.Conv3d(
                filters * (2 ** depth),
                filters * (2 ** depth), 1,
                padding=padding,
                groups=filters * (2 ** depth)
            ),
            nn.ReLU(),
            nn.InstanceNorm3d(filters * (2 ** depth)),
            nn.Conv3d(
                filters * (2 ** depth),
                filters * (2 ** (depth - 1)), kernel_size,
                padding=padding,
                # groups=n_images
            ),
            nn.ReLU(),
            nn.Dropout3d(),
            nn.InstanceNorm3d(filters * (2 ** (depth - 1))),
        )
        self.midconv.to(self.device)

        self.deconvlist_roi = map(
            lambda (ini, out): nn.Sequential(
                nn.ConvTranspose3d(
                    2 * ini, ini, kernel_size,
                    padding=padding,
                ),
                nn.ReLU(),
                nn.Dropout3d(),
                nn.InstanceNorm3d(ini),
                nn.ConvTranspose3d(
                    ini, ini, 1,
                    padding=padding,
                    groups=ini
                ),
                nn.ReLU(),
                nn.InstanceNorm3d(ini),
                nn.ConvTranspose3d(
                    ini, out, kernel_size,
                    padding=padding,
                ),
                nn.ReLU(),
                nn.Dropout3d(),
                nn.InstanceNorm3d(out),
            ),
            zip(
                filter_list[::-1], filter_list[-2::-1] + [filters]
            )
        )
        for d in self.deconvlist_roi:
            d.to(self.device)

        self.deconvlist_tumor = map(
            lambda (ini, out): nn.Sequential(
                nn.ConvTranspose3d(
                    2 * ini, ini, kernel_size,
                    padding=padding,
                ),
                nn.ReLU(),
                nn.Dropout3d(),
                nn.InstanceNorm3d(ini),
                nn.ConvTranspose3d(
                    ini, ini, 1,
                    padding=padding,
                    groups=ini
                ),
                nn.ReLU(),
                nn.InstanceNorm3d(ini),
                nn.ConvTranspose3d(
                    ini, out, kernel_size,
                    padding=padding,
                ),
                nn.ReLU(),
                nn.Dropout3d(),
                nn.InstanceNorm3d(out),
            ),
            zip(
                filter_list[::-1], filter_list[-2::-1] + [filters]
            )
        )
        for d in self.deconvlist_tumor:
            d.to(self.device)

        # Segmentation
        self.out_tumor = nn.Sequential(
            nn.Conv3d(filters, 4, 1),
            nn.Softmax(dim=1)
        )
        self.out_tumor.to(self.device)

        self.out_roi = nn.Sequential(
            nn.Conv3d(filters, 3, 1),
            nn.Softmax(dim=1)
        )
        self.out_roi.to(self.device)

    def forward(self, x, dropout=0):
        down_list = []
        for c, p in zip(self.convlist, self.pooling):
            down = c(x)
            if dropout > 0:
                down = nn.functional.dropout3d(down, dropout)
            down_list.append(down)
            # x = F.max_pool3d(down, self.pooling)
            x = p(down)

        xr = xt = self.midconv(x)

        if dropout > 0:
            xr = nn.functional.dropout3d(xr, dropout)
            xt = nn.functional.dropout3d(xt, dropout)

        for dr, dt, prev in zip(
                self.deconvlist_roi, self.deconvlist_tumor, down_list[::-1]
        ):
            interpr = F.interpolate(xr, size=prev.shape[2:])
            xr = dr(torch.cat((prev, interpr), dim=1))
            if dropout > 0:
                xr = nn.functional.dropout3d(xr, dropout)

            interpt = F.interpolate(xt, size=prev.shape[2:])
            xt = dt(torch.cat((prev, interpt), dim=1))
            if dropout > 0:
                xt = nn.functional.dropout3d(xt, dropout)

        output_roi = self.out_roi(xr)
        output_tumor = self.out_tumor(xt)
        return output_roi, output_tumor

    def step(
            self,
            data,
            train=True
    ):
        # We train the model and check the loss
        torch.cuda.synchronize()
        x, (yt, yr) = data
        predr, predt = self(x.to(self.device))
        b_lossr = multidsc_loss(
            predr, yr.to(self.device), averaged=train
        )
        b_losst = multidsc_loss(
            predt, yt.to(self.device), averaged=train
        )

        pred_bckr = torch.sum(predr[:, :-1, ...], dim=1)
        pred_tmrr = predr[:, -1, ...]
        predr_mix = torch.stack((pred_bckr, pred_tmrr), dim=1)

        pred_bckt = predt[:, 0, ...]
        pred_tmrt = torch.sum(predt[:, 1:, ...], dim=1)
        predt_mix = torch.stack((pred_bckt, pred_tmrt), dim=1)

        b_loss_mix = multidsc_loss(predr_mix, predt_mix, averaged=train)

        if train:
            b_loss = b_lossr + b_losst + b_loss_mix
            self.optimizer_alg.zero_grad()
            b_loss.backward()
            self.optimizer_alg.step()
        else:
            b_mean_r = torch.mean(b_lossr)
            b_mean_t = torch.mean(b_losst)
            b_mean_mix = torch.mean(b_loss_mix)
            b_loss = b_mean_r + b_mean_t + b_mean_mix

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return b_loss.tolist(), 1 - b_lossr, 1 - b_losst, 1 - b_loss_mix

    def mini_batch_loop(
            self, training, train=True
    ):
        losses = list()
        mid_losses = list()
        n_batches = len(training)
        for batch_i, batch_data in enumerate(training):
            # We train the model and check the loss
            loss_value, b_lossr, b_losst, b_lossmix = self.step(
                batch_data, train=train
            )

            mid_losses.append(
                b_lossr.tolist() + b_losst.tolist() + b_lossmix.tolist()
            )
            losses.append(loss_value)

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses), train
            )

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
            rois,
            val_split=0.1,
            optimizer='adadelta',
            epochs=50,
            patience=5,
            num_workers=16,
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

        optimizer_dict = {
            'adam': lambda param: torch.optim.Adam(param, lr=1e-2),
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

        if rois is not None:
            r_train = rois[:n_t_samples]
            r_val = rois[n_t_samples:]
        else:
            r_train = None
            r_val = None

        # < Training >
        # Full image one
        print('Dataset creation')
        train_dataset = BBImageTupleDataset(
            d_train, t_train, r_train
        )
        print('Dataloader creation')
        train_loader = DataLoader(
            train_dataset, 1, num_workers=num_workers,
        )

        # < Validation >
        val_dataset = BBImageTupleDataset(
            d_val, t_val, r_val
        )
        val_loader = DataLoader(
            val_dataset, 1, num_workers=num_workers
        )

        l_names = [
            'train', ' val ',
            ' BCKr ', ' BRAIN', ' TUMOR',
            ' BCKt ', '  NET ', '  ED  ', '  ET  ',
            ' BCKo ', ' TMRo '
        ]
        best_losses = [-np.inf] * (len(l_names))
        best_e = 0

        for self.epoch in range(epochs):
            # Main epoch loop
            # < Training >
            self.t_train = time.time()
            loss_tr = self.mini_batch_loop(train_loader)
            tr_loss_s = '{:7.5f}'.format(loss_tr)
            if loss_tr < best_loss_tr:
                best_loss_tr = loss_tr
                tr_loss_s = '\033[32m%s\033[0m' % tr_loss_s

            # < Validation >
            with torch.no_grad():
                self.t_val = time.time()
                loss_val, mid_losses = self.mini_batch_loop(val_loader, False)

            losses_color = map(
                lambda (pl, l): '\033[36m%s\033[0m' if l > pl else '%s',
                zip(best_losses, mid_losses)
            )
            losses_s = map(
                lambda (c, l): c % '{:8.4f}'.format(l),
                zip(losses_color, mid_losses)
            )
            best_losses = map(
                lambda (pl, l): l if l > pl else pl,
                zip(best_losses, mid_losses)
            )

            # Patience check
            improvement = loss_val < best_loss_val
            loss_s = '{:7.5f}'.format(loss_val)
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

    def segment(
            self,
            data,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.eval()
        whites = ' '.join([''] * 12)
        results_roi = []
        results_tumor = []

        with torch.no_grad():
            cases = len(data)
            t_in = time.time()
            for i, data_i in enumerate(data):

                # We test the model with the current batch
                input_i = torch.unsqueeze(
                    to_torch_var(data_i, self.device), 0
                )
                torch.cuda.synchronize()
                pred_roi, pred_tumor = self(input_i).squeeze().tolist()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                results_roi.append(pred_roi)
                results_tumor.append(pred_tumor)

                t_out = time.time() - t_in
                t_s = time_to_string(t_out)
                # Print stuff
                if verbose:
                    percent = 20 * (i + 1) / cases
                    progress_s = ''.join(['-'] * percent)
                    remainder_s = ''.join([' '] * (20 - percent))
                    t_eta = (t_out / (i + 1)) * (cases - (i + 1))
                    eta_s = time_to_string(t_eta)
                    print(
                        '\033[K%sTested case (%02d/%02d) [%s>%s]'
                        ' %s / ETA: %s' % (
                            whites, i, cases, progress_s, remainder_s,
                            t_s, eta_s
                        ),
                        end='\r'
                    )
                    sys.stdout.flush()

        if verbose:
            print('\033[K%sTesting finished succesfully' % whites)

        return results_roi, results_tumor

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class BratsSurvivalNet(nn.Module):
    def __init__(
            self,
            filters=32,
            kernel_size=3,
            pool_seg=2,
            depth_seg=4,
            pool_pred=2,
            depth_pred=4,
            n_images=4,
            n_features=4,
            dense_size=256,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):

        # Init
        super(BratsSurvivalNet, self).__init__()
        self.device = device
        self.optimizer_alg = None
        self.t_train = 0
        self.t_val = 0
        self.epoch = 0

        self.base_model = BratsSegmentationNet(
            filters=filters, kernel_size=kernel_size, pool_size=pool_seg,
            depth=depth_seg, n_images=n_images
        )

        init_features = filters * (2 ** (depth_pred - 1))
        end_features = filters * (2 ** (depth_pred))

        self.pooling = map(
            lambda d: nn.Sequential(
                nn.Conv3d(
                    init_features * (2 ** d),
                    init_features * (2 ** (d + 1)),
                    kernel_size,
                ),
                nn.ReLU(),
                nn.InstanceNorm3d(init_features * (2 ** (d + 1))),
                nn.Conv3d(
                    init_features * (2 ** d),
                    init_features * (2 ** (d + 1)),
                    pool_pred, stride=pool_pred
                )
            ),
            range(depth_pred)
        )
        for p in self.pooling:
            p.to(self.device)

        self.global_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.global_pooling.to(self.device)

        self.linear = nn.Sequential(
            nn.Linear(end_features + n_features, dense_size),
            nn.ReLU(),
            nn.InstanceNorm1d(dense_size)
        )
        self.linear.to(self.device)

        self.out = nn.Linear(dense_size, 1)
        self.out.to(self.device)

    def forward(self, im, features):
        for c, p in zip(self.base_model.convlist, self.base_model.pooling):
            down = c(im)
            im = p(down)

        im = self.midconv(im)

        for p in self.pooling:
            im = p(im)

        im = self.global_pooling(im).view(im.shape[:2])

        x = torch.cat((im, features), dim=1)

        x = self.linear(x)
        output = self.out(x)

        return output

    def mini_batch_loop(
            self, training, train=True
    ):
        losses = list()
        n_batches = len(training)
        for batch_i, (x, y) in enumerate(training):
            torch.cuda.synchronize()
            # We train the model and check the loss
            pred_y = self(x.to(self.device))
            batch_loss = nn.MSELoss()(pred_y, y.to(self.device))

            if train:
                self.optimizer_alg.zero_grad()
                batch_loss.backward()
                self.optimizer_alg.step()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            loss_value = batch_loss.tolist()

            losses.append(loss_value)

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses), train
            )

        if self.sampler is not None and train:
            self.sampler.update()

        return np.mean(losses)

    def fit(
            self,
            data_seg,
            target_seg,
            rois_seg,
            data_pred,
            target_pred,
            rois_pred,
            val_split=0.1,
            optimizer='adadelta',
            epochs=50,
            patience=5,
            num_workers=16,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=True
    ):
        # Init
        self.to(device)
        self.train()

        # We first fit the segmentation as best as we can with the cases that
        # are not used for prediction (no GTR, no age, etc.).
        self.base_model.fit(
            data_seg, target_seg, rois_seg, val_split=val_split,
            optimizer=optimizer, epochs=epochs, patience=patience,
            num_workers=num_workers, device=device, verbose=verbose
        )
        self.base_model.eval()

        # Now we actually train the prediction network.
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
            n_samples = len(data_pred)

            n_t_samples = int(n_samples * (1 - val_split))
            n_v_samples = n_samples - n_t_samples

            if verbose:
                print(
                    'Training / validation samples = %d / %d' % (
                        n_t_samples, n_v_samples
                    )
                )

            d_train = data_pred[:n_t_samples]
            d_val = data_pred[n_t_samples:]

            t_train = target_pred[:n_t_samples]
            t_val = target_pred[n_t_samples:]

            if rois_pred is not None:
                r_train = rois_pred[:n_t_samples]
                r_val = rois_pred[n_t_samples:]
            else:
                r_train = None
                r_val = None

            # Training
            # Full image one
            print('Dataset creation images <with validation>')
            train_dataset = BBImageValueDataset(
                d_train, t_train, r_train
            )
            train_loader = DataLoader(
                train_dataset, 1, shuffle=True, num_workers=num_workers
            )
            
            # Validation
            val_dataset = BBImageValueDataset(
                d_val, t_val, r_val
            )
            val_loader = DataLoader(
                val_dataset, 1, num_workers=num_workers
            )
        else:
            # Training
            # Full image one
            print('Dataset creation images <with validation>')
            dataset = BBImageValueDataset(
                data_pred, target_pred, rois_pred
            )
            train_loader = DataLoader(
                dataset, 1, shuffle=True, num_workers=num_workers
            )

            # Validation
            val_loader = DataLoader(
                dataset, 1, num_workers=num_workers
            )

        l_names = ['train', ' val ']
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
                loss_val = self.mini_batch_loop(val_loader, False)

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
                    [epoch_s, tr_loss_s, loss_s] + [t_s]
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
