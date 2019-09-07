from __future__ import print_function
import time
import sys
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from criterions import multidsc_loss
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
            final_dropout=0.0,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
    ):
        super(BratsSegmentationNet, self).__init__()
        # Init
        self.to(device)
        self.drop = False
        self.dropout = final_dropout
        self.final_dropout = final_dropout
        self.sampler = None
        self.t_train = 0
        self.t_val = 0
        self.epoch = 0
        self.optimizer_alg = None
        self.device = device
        padding = kernel_size // 2

        # Down path
        filter_list = map(lambda i: filters * 2 ** i, range(depth))
        self.convlist = nn.ModuleList(map(
            lambda (ini, out): nn.Sequential(
                nn.Conv3d(
                    ini, out, kernel_size,
                    padding=padding,
                ),
                nn.ReLU(),
                nn.InstanceNorm3d(out),
                # nn.BatchNorm3d(out),
                nn.Conv3d(
                    out, out, kernel_size,
                    padding=padding,
                    # groups=out
                ),
                nn.ReLU(),
                # nn.InstanceNorm3d(out),
                # nn.BatchNorm3d(out),
            ),
            zip([n_images] + filter_list[:-1], filter_list)
        ))
        # self.pooling = map(
        #     lambda f: nn.Conv3d(f, f, pool_size, stride=pool_size, groups=f),
        #     filter_list
        # )
        self.pooling = nn.ModuleList(
            [nn.AvgPool3d(pool_size)] * len(filter_list)
        )

        self.midconv = nn.Sequential(
            nn.Conv3d(
                filters * (2 ** (depth - 1)),
                filters * (2 ** depth), kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.InstanceNorm3d(filters * (2 ** depth)),
            # nn.BatchNorm3d(filters * (2 ** depth)),
            nn.Conv3d(
                filters * (2 ** depth),
                filters * (2 ** (depth - 1)), kernel_size,
                padding=padding,
                # groups=filters * (2 ** (depth - 1)),
            ),
            nn.ReLU(),
            # nn.InstanceNorm3d(filters * (2 ** (depth - 1))),
            # nn.BatchNorm3d(filters * (2 ** (depth - 1))),
        )
        self.midconv.to(self.device)

        self.deconvlist = nn.ModuleList(map(
            lambda (ini, out): nn.Sequential(
                nn.ConvTranspose3d(
                    2 * ini, ini, kernel_size,
                    padding=padding,
                ),
                nn.ReLU(),
                nn.InstanceNorm3d(ini),
                # nn.BatchNorm3d(ini),
                nn.ConvTranspose3d(
                    ini, out, kernel_size,
                    padding=padding,
                    # groups=out
                ),
                nn.ReLU(),
                # nn.InstanceNorm3d(out),
                # nn.BatchNorm3d(out),
            ),
            zip(filter_list[::-1], filter_list[-2::-1] + [filters])
        ))

        # Segmentation
        self.out = nn.Sequential(
            nn.Conv3d(filters, 4, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        down_list = []
        for c, p in zip(self.convlist, self.pooling):
            c.to(self.device)
            down = c(x)
            drop = F.dropout3d(down, p=self.dropout, training=self.drop)
            down_list.append(drop)
            p.to(self.device)
            x = p(drop)

        x = self.midconv(x)
        x = F.dropout3d(x, p=self.dropout, training=self.drop)

        for d, prev in zip(self.deconvlist, down_list[::-1]):
            interp = F.interpolate(x, size=prev.shape[2:])
            d.to(self.device)
            x = d(torch.cat((prev, interp), dim=1))
            x = F.dropout3d(x, p=self.dropout, training=self.drop)

        self.out.to(self.device)
        output = self.out(x)
        return output

    def mini_batch_loop(
            self, training, train=True, refine=False
    ):
        self.drop = train
        losses = list()
        mid_losses = list()
        n_batches = len(training)
        for batch_i, (x, y) in enumerate(training):
            # We train the model and check the loss
            if train:
                self.optimizer_alg.zero_grad()

            torch.cuda.synchronize()
            pred_labels = self(x.to(self.device))

            # Regular class loss
            batch_loss_c = multidsc_loss(
                pred_labels, y.to(self.device), averaged=False
            )

            # Evaluated BraTS losses
            # WT = all tumor labels
            pred_wt = torch.unsqueeze(
                torch.sum(pred_labels[:, 1:, ...], dim=1), dim=1
            )
            y_wt = torch.unsqueeze((y > 0).type_as(y), dim=1)
            batch_loss_wt = multidsc_loss(
                pred_wt, y_wt.to(self.device),
            )

            # TC = label 3 -ET- (4 in GT) and 1 -NET+NCR-
            pred_tc = torch.unsqueeze(
                pred_labels[:, 1, ...] + pred_labels[:, 3, ...], dim=1
            )
            y_tc = torch.unsqueeze(
                (y == 1).type_as(y) + (y == 3).type_as(y), dim=1
            )
            batch_loss_tc = multidsc_loss(
                pred_tc, y_tc.to(self.device),
            )

            # Final loss from BraTS
            if refine:
                class_loss = torch.sum(batch_loss_c)
                batch_loss = batch_loss_wt + batch_loss_tc + class_loss
            else:
                batch_loss = torch.sum(batch_loss_c)

            if train:
                # batch_loss = multidsc_loss(
                #     pred_labels, y.to(self.device), averaged=train
                # )
                batch_loss.backward()
                self.optimizer_alg.step()
            else:
                # roi_value = torch.mean(batch_loss_r).tolist()
                # tumor_value = torch.mean(batch_loss_t).tolist()
                # loss_value = roi_value + tumor_value
                dsc = (1 - batch_loss_c).tolist()
                dsc.append((1 - batch_loss_wt).tolist())
                dsc.append((1 - batch_loss_tc).tolist())
                mid_losses.append(dsc)

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            loss_value = batch_loss.tolist()
            losses.append(loss_value)

            # Curriculum dropout
            # (1 - rho) * exp(- gamma * t) + rho, gamma > 0

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
            train_loader,
            val_loader,
            epochs=100,
            patience=10,
            initial_dropout=0.99,
            ann_rate=1e-2,
            initial_lr=1,
            refine=False,
            verbose=True
    ):
        # Init
        l_names = [
            'train', ' val ', '  BCK ', '  NET ', '  ED  ', '  ET  ',
            '  WT  ', '  TC  ', 'p_drop'
        ]
        self.dropout = initial_dropout
        model_params = filter(lambda p: p.requires_grad, self.parameters())

        # If we are refining, the best train and validation losses are the ones
        # we already have. That is the point of refining.
        if refine:
            self.optimizer_alg = torch.optim.SGD(
                model_params, lr=initial_lr, weight_decay=1e-1
            )
            with torch.no_grad():
                self.eval()
                self.t_val = time.time()
                loss_tr, _ = self.mini_batch_loop(
                    train_loader, train=False, refine=refine
                )
                self.t_val = time.time()
                best_loss_val, best_losses = self.mini_batch_loop(
                    val_loader, train=False, refine=refine
                )
                best_e = None

                if verbose:
                    tr_loss_s = '\033[32m%0.5f\033[0m' % loss_tr
                    losses_color = map(
                        lambda l: '\033[36m%s\033[0m', best_losses
                    )
                    losses_s = map(
                        lambda (c, l): c % '{:8.4f}'.format(l),
                        zip(losses_color, best_losses)
                    )
                    epoch_s = '\033[32mEpoch ini\033[0m'
                    loss_s = '\033[32m%0.5f\033[0m' % best_loss_val
                    print('\033[K', end='')
                    whites = ' '.join([''] * 12)
                    l_bars = '--|--'.join(
                        ['-' * 5] * 2 + ['-' * 6] * len(l_names[2:])
                    )
                    l_hdr = '  |  '.join(l_names)
                    print('%sEpoch num |  %s  |' % (whites, l_hdr))
                    print('%s----------|--%s--|' % (whites, l_bars))
                    final_s = whites + ' | '.join(
                        [epoch_s, tr_loss_s, loss_s] + losses_s + [' ' * 8, '']
                    )
                    print(final_s)
        else:
            self.optimizer_alg = torch.optim.SGD(
                model_params, lr=initial_lr
            )

        no_improv_e = 0
        best_state = deepcopy(self.state_dict())
        best_opt = deepcopy(self.optimizer_alg.state_dict())

        t_start = time.time()
        best_loss_tr = np.inf

        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            self.train()
            loss_tr = self.mini_batch_loop(
                train_loader, refine=refine
            )
            improvement_tr = loss_tr < best_loss_tr
            if improvement_tr:
                best_loss_tr = loss_tr
                tr_loss_s = '\033[32m%0.5f\033[0m' % loss_tr
            else:
                # Learning rate update
                tr_loss_s = '%0.5f' % loss_tr

            with torch.no_grad():
                self.eval()
                self.t_val = time.time()
                loss_val, mid_losses = self.mini_batch_loop(
                    val_loader, train=False, refine=refine
                )

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
                best_opt = deepcopy(self.optimizer_alg.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch %03d' % self.epoch
                no_improv_e += 1

            if not (improvement_tr or improvement):
                self.optimizer_alg.load_state_dict(best_opt)
                if self.dropout <= 0.5:
                    self.load_state_dict(best_state)
                    for param_group in self.optimizer_alg.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)

            drop_s = '{:8.5f}'.format(self.dropout)
            if self.final_dropout <= self.dropout:
                self.dropout = max(
                    self.final_dropout, self.dropout - ann_rate
                )

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                if self.epoch == 0 and not refine:
                    l_bars = '--|--'.join(
                        ['-' * 5] * 2 + ['-' * 6] * len(l_names[2:])
                    )
                    l_hdr = '  |  '.join(l_names)
                    print('%sEpoch num |  %s  |' % (whites, l_hdr))
                    print('%s----------|--%s--|' % (whites, l_bars))
                final_s = whites + ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] + losses_s + [drop_s, t_s]
                )
                print(final_s)

            if no_improv_e == int(patience / (1 - self.dropout)):
                break

        self.epoch = best_e
        self.load_state_dict(best_state)
        t_end = time.time() - t_start
        t_end_s = time_to_string(t_end)
        if verbose:
            print(
                'Training finished in %d epochs (%s) '
                'with minimum loss = %f (epoch %d)' % (
                    self.epoch + 1, t_end_s, best_loss_val, best_e)
            )

    def segment(
            self,
            data,
            verbose=True
    ):
        # Init
        self.drop = False
        self.dropout = 0
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
                pred = np.array(self(input_i).squeeze().tolist())
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
            verbose=True
    ):
        # Init
        self.drop = True
        self.dropout = dropout
        self.eval()
        whites = ' '.join([''] * 12)
        seg_results = []

        with torch.no_grad():
            cases = len(data)
            t_in = time.time()
            for i, data_i in enumerate(data):
                outputs = np.zeros((4,) + data_i.shape[1:])
                for e in range(steps):
                    # We test the model with the current batch
                    input_i = torch.unsqueeze(
                        to_torch_var(data_i, self.device), 0
                    )

                    # Testing itself
                    torch.cuda.synchronize()
                    pred = np.array(self(input_i).squeeze().tolist())
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    outputs += pred

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

                mean_output = outputs / steps
                seg_results.append(mean_output)

        if verbose:
            print('\033[K%sTesting finished succesfully' % whites)

        return seg_results

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
            depth_pred=5,
            n_images=4,
            n_features=1,
            dense_size=256,
            dropout=0.99,
            ann_rate=1e-2,
            final_dropout=0,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):

        # Init
        super(BratsSurvivalNet, self).__init__()
        self.loss = nn.SmoothL1Loss()
        self.to(device)
        self.device = device
        self.optimizer_alg = None
        self.t_train = 0
        self.t_val = 0
        self.epoch = 0
        # Annealed dropout
        self.drop = False
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.ann_rate = ann_rate

        self.base_model = BratsSegmentationNet(
            filters=filters, kernel_size=kernel_size, pool_size=pool_seg,
            depth=depth_seg, n_images=n_images
        )

        self.global_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        init_feat = [filters * (2 ** (depth_seg - 1))]
        middle_feat = [dense_size] * (depth_pred - 1)
        channels_in = init_feat + middle_feat
        self.linear = nn.ModuleList(
            map(
                lambda ch_in: nn.Sequential(
                    nn.InstanceNorm1d(ch_in),
                    nn.Linear(ch_in, dense_size),
                    nn.SELU()
                ),
                channels_in
            )
        )

        self.out = nn.Linear(dense_size + n_features, 1)

    def forward(self, im, features):
        for c, p in zip(self.base_model.convlist, self.base_model.pooling):
            c.to(self.device)
            p.to(self.device)
            im = p(c(im))

        self.base_model.dropout = self.dropout
        im = self.base_model.midconv(im)
        drop = F.dropout3d(im, p=self.dropout, training=self.drop)

        self.global_pooling.to(self.device)
        x = self.global_pooling(drop).view(im.shape[:2])

        for l in self.linear:
            l.to(self.device)
            x = l(x)
            x = F.dropout(x, p=self.dropout, training=self.drop)

        x = torch.cat((x, features.type_as(x)), dim=1)
        self.out.to(self.device)
        output = self.out(x)
        if self.dropout <= 0.5:
            output = F.relu(output)
        else:
            output = F.leaky_relu(output)
        return output

    def mini_batch_loop(
            self, training, train=True
    ):
        self.drop = train
        losses = list()
        losses_cat = list()
        losses_abs = list()
        n_batches = len(training)
        for batch_i, (im, feat, y) in enumerate(training):
            torch.cuda.synchronize()
            if train:
                self.train()
                self.base_model.eval()
                self.optimizer_alg.zero_grad()
            else:
                self.eval()
            # We train the model and check the loss
            pred_y = self(im.to(self.device), feat.to(self.device))
            target_y = y.to(self.device).type_as(pred_y)

            target_short = (target_y < 300).type_as(pred_y)
            target_mid = (target_y >= 300).type_as(pred_y) *\
                         (target_y < 450).type_as(pred_y)
            target_long = (target_y >= 450).type_as(pred_y)

            pred_short = (pred_y < 300).type_as(pred_y)
            pred_mid = (pred_y >= 300).type_as(pred_y) *\
                       (pred_y < 450).type_as(pred_y)
            pred_long = (pred_y >= 450).type_as(pred_y)
            pred_cat = torch.stack(
                (pred_short, pred_mid, pred_long), dim=1
            )
            target_cat = torch.stack(
                (target_short, target_mid, target_long), dim=1
            )
            batch_loss_cat = torch.sum(1 - pred_cat * target_cat)
            batch_loss_abs = torch.abs(target_y - pred_y)
            batch_loss_sumabs = 1e-1 * torch.sum(batch_loss_abs)
            batch_loss = batch_loss_cat + batch_loss_sumabs

            loss_value = torch.squeeze(batch_loss).tolist()

            if train:
                batch_loss.backward()
                self.optimizer_alg.step()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            losses.append(loss_value)
            losses_cat.append(batch_loss_cat.tolist())
            losses_abs.append(torch.sum(batch_loss_abs).tolist())

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses), train
            )

        return np.mean(losses), np.mean(losses_cat), np.mean(losses_abs)

    def fit(
            self,
            train_loader,
            val_loader,
            epochs=50,
            patience=5,
            initial_lr=1e-1,
            verbose=True
    ):
        # Init
        self.train()
        self.base_model.eval()

        # Now we actually train the prediction network.
        best_loss_tr = np.inf
        best_loss_val = np.inf
        best_loss_abs = np.inf
        best_acc_cat = -np.inf
        no_improv_e = 0
        best_state = deepcopy(self.state_dict())

        for p in self.base_model.parameters():
            p.requires_grad = False

        model_params = filter(lambda p: p.requires_grad, self.parameters())

        t_start = time.time()

        l_names = ['train', ' val ', ' cat ', ' abs ', 'pdrop']

        best_e = 0
        # SGD for L1
        self.optimizer_alg = torch.optim.SGD(
            model_params, lr=initial_lr, #weight_decay=1e-1
        )
        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            loss_tr, _, _ = self.mini_batch_loop(train_loader)
            if loss_tr < best_loss_tr:
                best_loss_tr = loss_tr
                tr_loss_s = '\033[32m{:7.3f}\033[0m'.format(loss_tr)
            else:
                tr_loss_s = '{:7.3f}'.format(loss_tr)

            with torch.no_grad():
                self.t_val = time.time()
                loss_val, acc_cat, loss_abs = self.mini_batch_loop(
                    val_loader, False
                )

            # Mid losses check
            if best_acc_cat < acc_cat:
                best_acc_cat = acc_cat
                cat_s = '\033[36m{:8.4f}\033[0m'.format(acc_cat)
            else:
                cat_s = '{:7.3f}'.format(acc_cat)
            if best_loss_abs > loss_abs:
                best_loss_abs = loss_abs
                abs_s = '\033[36m{:7.3f}\033[0m'.format(loss_abs)
            else:
                abs_s = '{:8.4f}'.format(loss_abs)

            # Patience check
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                epoch_s = '\033[32mEpoch %03d\033[0m' % self.epoch
                loss_s = '\033[32m{:7.3f}\033[0m'.format(loss_val)
                best_e = self.epoch
                best_state = deepcopy(self.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch %03d' % self.epoch
                loss_s = '{:7.3f}'.format(loss_val)
                no_improv_e += 1

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)
            drop_s = '{:8.5f}'.format(self.dropout)
            self.dropout = max(
                self.final_dropout, self.dropout - self.ann_rate
            )

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
                    [epoch_s, tr_loss_s, loss_s, cat_s, abs_s, drop_s] + [t_s]
                )
                print(final_s)

            if no_improv_e == int(patience / (1 - self.dropout)):
                break

        self.epoch = best_e
        self.load_state_dict(best_state)
        t_end = time.time() - t_start
        t_end_s = time_to_string(t_end)
        if verbose:
            print(
                'Training finished in %d epochs (%s) '
                'with minimum loss = %f (epoch %d)' % (
                    self.epoch + 1, t_end_s, best_loss_val, best_e)
            )

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

    def predict(
            self,
            data,
            features,
            bb,
            verbose=True
    ):
        # Init
        self.eval()
        self.drop = False
        self.dropout = 0
        whites = ' '.join([''] * 12)
        results = []

        with torch.no_grad():
            cases = len(data)
            t_in = time.time()
            for i, (data_i, feat_i) in enumerate(zip(data, features)):

                # We test the model with the current batch
                inputd_i = to_torch_var(
                        [data_i[tuple([slice(None)] + bb)]], self.device
                )
                inputf_i = to_torch_var([[feat_i]], self.device)
                torch.cuda.synchronize()
                pred = F.relu(self(inputd_i, inputf_i)).squeeze().tolist()
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

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))

