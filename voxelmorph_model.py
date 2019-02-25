from __future__ import print_function
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
        min_bb = np.min(
            map(
                lambda mask: np.min(np.where(mask > 0), axis=-1),
                masks
            ),
            axis=0
        )
        max_bb = np.max(
            map(
                lambda mask: np.max(np.where(mask > 0), axis=-1),
                masks
            ),
            axis=0
        )
        self.bb = tuple(
            map(
                lambda (min_i, max_i): slice(min_i, max_i),
                zip(min_bb, max_bb)
            )
        )

    def __getitem__(self, index):
        source = np.expand_dims(self.sources[index][self.bb], axis=0)
        target = np.expand_dims(self.targets[index][self.bb], axis=0)
        mask = np.expand_dims(self.masks[index][self.bb], axis=0)

        inputs = (
            source,
            target,
            mask,
        )

        return inputs, target

    def __len__(self):
        return len(self.sources)


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
        input_s = torch.cat([source, target], dim=1).to(self.device)

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
            [source.to(self.device), df]
        )

        return source_mov, df

    def register(
            self,
            sources,
            targets,
            brain_masks,
            batch_size=1,
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
        e = 0
        best_e = 0

        t_start = time.time()

        # This is actually a registration approach. It uses the nn framework
        # but it doesn't actually do any supervised training. Therefore, there
        # is no real validation.
        # Due to this, we modified the generic fit algorithm.

        dataset = ImageListDataset(
            sources,
            targets,
            brain_masks
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
                'Registration finished in %d epochs (%fs)'
                ' with minimum loss = %f (epoch %d)' % (
                    e + 1, t_end, best_loss_tr, best_e)
            )

    def step(
            self,
            optimizer_alg,
            epoch,
            dataloader
    ):
        # Again. This is supposed to be a step on the registration process,
        # there's no need for splitting data. We just compute the deformation,
        # then compute the global loss (and intermediate ones to show) and do
        # back propagation.
        with torch.autograd.set_detect_anomaly(True):
            n_batches = len(dataloader.dataset) / dataloader.batch_size + 1
            loss_list = []
            losses_list = []
            for (
                    batch_i,
                    ((b_source, b_target, b_mask), b_gt)
            ) in enumerate(dataloader):
                # We train the model and check the loss
                b_gt = b_gt.to(self.device)
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
                mean_loss = np.asscalar(np.mean(loss_list))

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

            mid_losses = np.mean(zip(*losses_list), axis=1)
            loss_value = np.sum(mid_losses)

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

    def load_model(self, net_name):
        self.load_state_dict(
            torch.load(net_name)
        )
