from __future__ import print_function
from operator import and_
import itertools
import time
from copy import deepcopy
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from layers import SpatialTransformer
from torch.utils.data.dataset import Dataset


class ImageListDataset(Dataset):
    def __init__(self, sources, targets, masks):
        # Init
        # Image and mask should be numpy arrays
        self.sources = sources
        self.targets = targets
        self.masks = masks

    def __getitem__(self, index):
        inputs = (
            self.sources[index],
            self.targets[index],
            self.masks[index]
        )
        target = self.targets[index],
        return inputs, target

    def __len__(self):
        return len(self.sources)


class ImagesListCroppingDataset(Dataset):
    def __init__(self, cases, lesions, masks, patch_size=32, overlap=32):
        # Init
        # Image and mask should be numpy arrays
        shape_comparisons = map(
            lambda case: map(
                lambda (x, y): x.shape == y.shape,
                zip(case[:-1], case[1:])
            ),
            cases
        )
        case_comparisons = map(
            lambda shapes: reduce(and_, shapes),
            shape_comparisons
        )
        assert reduce(and_, case_comparisons)

        self.n_cases = len(cases)
        self.cases = cases
        case_idx = map(lambda case: range(len(case)), cases)
        timepoints_combo = map(
            lambda timepoint_idx: map(
                lambda i: map(
                    lambda j: (i, j),
                    timepoint_idx[i + 1:]
                ),
                timepoint_idx[:-1]
            ),
            case_idx
        )
        self.combos = map(
            lambda combo: np.concatenate(combo, axis=0),
            timepoints_combo
        )
        self.lesions = lesions
        self.masks = masks

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(self.masks[0].shape)
        patch_half = map(lambda p_length: p_length/2, patch_size)

        steps = map(lambda p_length: max(p_length - overlap, 1), patch_size)

        # Create bounding box and define
        min_bb = map(lambda mask: np.min(np.where(mask > 0), axis=-1), masks)
        min_bb = map(
            lambda min_bb_i: map(
                lambda (min_i, p_len): min_i + p_len,
                zip(min_bb_i, patch_half)
            ),
            min_bb
        )
        max_bb = map(lambda mask: np.max(np.where(mask > 0), axis=-1), masks)
        max_bb = map(
            lambda max_bb_i: map(
                lambda (max_i, p_len): max_i + p_len,
                zip(max_bb_i, patch_half)
            ),
            max_bb
        )

        dim_ranges = map(
            lambda (min_bb_i, max_bb_i): map(
                lambda t: np.arange(*t), zip(min_bb_i, max_bb_i, steps)
            ),
            zip(min_bb, max_bb)
        )

        self.patch_slices = map(
            lambda dim_range: map(
                lambda voxel: tuple(map(
                    lambda (idx, p_len): slice(idx - p_len, idx + p_len),
                    zip(voxel, patch_half)
                )),
                itertools.product(*dim_range)
            ),
            dim_ranges
        )

        case_slices = map(
            lambda (p, c): len(p) * len(c),
            zip(self.patch_slices, self.combos)
        )

        self.max_slice = np.cumsum(case_slices)

    def __getitem__(self, index):
        # We select the case
        case = np.min(np.where(self.max_slice > index))
        case_timepoints = self.timepoints[case]
        case_slices = self.patch_slices[case]
        case_combos = self.combos[case]
        case_lesion = self.lesions[case]
        case_mask = self.masks[case]

        # Now we just need to look for the desired slice
        n_slices = len(case_slices)
        combo_idx = index / n_slices
        patch_idx = index % n_slices
        source = case_timepoints[case_combos[combo_idx, 0]]
        target = case_timepoints[case_combos[combo_idx, 1]]
        patch_slice = case_slices[patch_idx]
        inputs_p = (
            np.expand_dims(source[patch_slice], 0),
            np.expand_dims(target[patch_slice], 0),
            np.expand_dims(case_lesion[patch_slice], 0),
            np.expand_dims(case_mask[patch_slice], 0)
        )
        target_p = np.expand_dims(target[patch_slice], 0)
        return inputs_p, target_p

    def __len__(self):
        return self.max_slice[-1]


def to_torch_var(
        np_array,
        device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        requires_grad=False
):
    var = torch.autograd.Variable(torch.from_numpy(np_array), requires_grad=requires_grad)
    return var.to(device)


def normalized_xcor(var_x, var_y):
    if len(var_x) > 1 and len(var_y) > 1:
        # Init
        var_x_flat = var_x.view(-1)
        var_y_flat = var_y.view(-1)
        # Computation
        var_x_norm = var_x - torch.mean(var_x_flat)
        var_y_norm = var_y - torch.mean(var_y_flat)
        var_x_den = torch.rsqrt(torch.sum(var_x_norm * var_x_norm))
        var_y_den = torch.rsqrt(torch.sum(var_y_norm * var_y_norm))

        return torch.sum(var_x_norm * var_y_norm) * var_x_den * var_y_den
    else:
        return torch.mean(torch.abs(var_x - var_y))


def normalized_xcor_loss(var_x, var_y):
    if len(var_x) > 0 and len(var_y) > 0:
        return 1 - normalized_xcor(var_x, var_y)
    else:
        return torch.tensor(0)


def df_gradient_mean(df, mask):
    grad_v = torch.tensor([-1, 0, 1], dtype=torch.float32).to(df.device)
    grad_x_k = torch.reshape(grad_v, (1, 1, -1)).repeat((3, 3, 1))
    grad_y_k = torch.reshape(grad_v, (1, -1, 1)).repeat((3, 1, 3))
    grad_z_k = torch.reshape(grad_v, (-1, 1, 1)).repeat((1, 3, 3))
    # grad_k_tensor = torch.stack([grad_x_k, grad_y_k, grad_z_k], dim=0)

    grad_x = F.conv3d(df, grad_x_k.repeat(3, 3, 1, 1, 1), padding=1)
    grad_y = F.conv3d(df, grad_y_k.repeat(3, 3, 1, 1, 1), padding=1)
    grad_z = F.conv3d(df, grad_z_k.repeat(3, 3, 1, 1, 1), padding=1)
    # gradient = F.conv3d(df, grad_k_tensor.repeat(3, 1, 1, 1, 1), padding=1)
    gradient = torch.cat([grad_x, grad_y, grad_z], dim=1)
    gradient = torch.sum(gradient * gradient, dim=1, keepdim=True)
    mean_grad = torch.mean(gradient[mask])

    return mean_grad


class VoxelMorph(nn.Module):
    def __init__(
            self,
            conv_filters=list([16, 32, 32, 32]),
            deconv_filters=list([32, 32, 32, 32, 16, 16]),
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
            loss_names=list([' xcor ', 'deform']),
            lambda_value=1,
    ):
        # Init
        super(VoxelMorph, self).__init__()
        self.lambda_value = lambda_value
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

        source, target = inputs
        input_s = torch.stack([source, target], dim=1).to(self.device)

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

        source_mov = self.trans_im([source, df])

        return source_mov, df

    def register(
            self,
            source,
            target,
            brain_mask,
            batch_size=32,
            batch_size_im=1,
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

        dataset_im = ImageListDataset(
            source,
            target, brain_mask
        )
        dataloader_im = DataLoader(
            dataset_im, batch_size_im, True, num_workers=num_workers
        )

        dataset = ImagesListCroppingDataset(
            source,
            target, brain_mask
        )
        dataloader = DataLoader(
            dataset, batch_size, True, num_workers=num_workers
        )

        l_names = [' loss '] + self.loss_names
        best_losses = [np.inf] * (len(l_names) - 1)

        for e in range(epochs):
            # Main epoch loop
            t_in = time.time()
            loss_tr, mid_losses = self.step(
                optimizer_alg,
                e,
                dataloader,
                dataloader_im
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
            dataloader_tr,
            dataloader_val
    ):
        # Again. This is supposed to be a step on the registration process,
        # there's no need for splitting data. We just compute the deformation,
        # then compute the global loss (and intermediate ones to show) and do
        # back propagation.
        with torch.autograd.set_detect_anomaly(True):
            n_batches = len(dataloader_tr.dataset) / dataloader_tr.batch_size + 1
            loss_list = []
            losses_list = []
            for batch_i, ((b_source, b_target, b_mask), b_gt) in enumerate(dataloader_tr):
                # We train the model and check the loss
                b_gt = b_gt[0].to(self.device)
                b_moved, b_df = self((b_source, b_target))
                b_losses = self.longitudinal_loss(
                    b_moved,
                    b_gt,
                    b_df,
                    b_mask
                )

                # Final loss value computation per batch
                batch_loss = sum(b_losses).to(self.device)
                b_loss_value = batch_loss.tolist()
                loss_list.append(b_loss_value)
                mean_loss = np.mean(loss_list)

                b_mid_losses = map(lambda l: l.tolist(), b_losses)
                losses_list.append(b_mid_losses)

                # Print the intermediate results
                whites = ' '.join([''] * 12)
                percent = 20 * batch_i / n_batches
                progress_s = ''.join(['-'] * percent)
                remaining_s = ''.join([' '] * (20 - percent))
                bar = '[' + progress_s + '>' + remaining_s + ']'
                curr_values_s = ' loss %f (%f)' % (b_loss_value, mean_loss)
                batch_s = '%sEpoch %03d (%02d/%02d) %s%s' % (
                    whites, epoch, batch_i, n_batches, bar, curr_values_s
                )
                print('\033[K', end='')
                print(batch_s, end='\r')
                sys.stdout.flush()

                # Backpropagation
                optimizer_alg.zero_grad()
                batch_loss.backward()
                optimizer_alg.step()

            # We compute the "validation loss" with the whole image
            with torch.no_grad():
                loss_list = []
                losses_list = []
                for batch_i, ((b_source, b_target, b_mask), b_gt) in enumerate(dataloader_val):
                    # We train the model and check the loss
                    b_gt = b_gt[0].to(self.device)
                    b_moved, b_df = self((b_source, b_target))
                    b_losses = self.longitudinal_loss(
                        b_moved,
                        b_gt,
                        b_df,
                        b_mask
                    )

                    # Final loss value computation per batch
                    batch_loss = sum(b_losses).to(self.device)
                    b_loss_value = batch_loss.tolist()
                    loss_list.append(b_loss_value)

                    b_mid_losses = map(lambda l: l.tolist(), b_losses)
                    losses_list.append(b_mid_losses)

            loss_value = np.mean(loss_list)
            mid_losses = np.mean(zip(*losses_list))

        return loss_value, mid_losses

    def get_deformation(
            self,
            source,
            target,
    ):
        # Init
        self.to(self.device)
        self.eval()

        source_tensor = to_torch_var(source)
        target_tensor = to_torch_var(target)

        with torch.no_grad():
            input_s = torch.cat([source_tensor, target_tensor], dim=1)

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

        return map(np.squeeze, df.cpu().numpy())

    def transform_image(
            self,
            source,
            target,
            verbose=True
    ):
        # Init
        # Init
        self.to(self.device)
        self.eval()

        source_tensor = to_torch_var(source)
        target_tensor = to_torch_var(target)

        with torch.no_grad():
            input_s = torch.cat([source_tensor, target_tensor], dim=1)

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

            source_mov = self.trans_im([source, df])

        if verbose:
            print(
                '\033[K%sTransformation finished' % ' '.join([''] * 12)
            )

        return map(np.squeeze, source_mov.cpu().numpy())

    def transform_mask(
            self,
            source,
            target,
            mask,
            verbose=True
    ):
        # Init
        # Init
        self.to(self.device)
        self.eval()

        source_tensor = to_torch_var(source)
        target_tensor = to_torch_var(target)

        with torch.no_grad():
            input_s = torch.cat([source_tensor, target_tensor], dim=1)

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

            mask_mov = self.trans_mask([mask, df])

        if verbose:
            print(
                '\033[K%sTransformation finished' % ' '.join([''] * 12)
            )

        return map(np.squeeze, mask_mov.cpu().numpy())

    def longitudinal_loss(
            self,
            moved,
            target,
            df,
            roi,
    ):
        # Init
        moved_roi = moved[roi > 0]
        target_roi = target[roi > 0]

        losses_dict = {
            ' xcor ': lambda: normalized_xcor_loss(moved_roi, target_roi),
            'deform': lambda: self.lambda_value * df_gradient_mean(df, roi),
        }

        losses = tuple(map(lambda l: losses_dict[l](), self.loss_names))

        return losses

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)
