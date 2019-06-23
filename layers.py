from operator import mul
import torch
import itertools
import numpy as np
from torch import nn
from torch.nn import functional as F


class SmoothingLayer(nn.Module):
    def __init__(
            self,
            length=5,
            init_sigma=0.5,
            trainable=False
    ):
        super(SmoothingLayer, self).__init__()
        if trainable:
            self.sigma = nn.Parameter(
                torch.tensor(
                    init_sigma,
                    dtype=torch.float,
                    requires_grad=True
                )
            )
        else:
            self.sigma = torch.tensor(
                    init_sigma,
                    dtype=torch.float
                )
        self.length = length

    def forward(self, x):
        dims = len(x.shape) - 2
        assert dims <= 3, 'Too many dimensions for convolution'

        kernel_shape = (self.length,) * dims
        lims = map(lambda s: (s - 1.) / 2, kernel_shape)
        grid = map(
            lambda g: torch.tensor(g, dtype=torch.float, device=x.device),
            np.ogrid[tuple(map(lambda l: slice(-l, l + 1), lims))]
        )
        sigma_square = self.sigma * self.sigma
        k = torch.exp(
            -sum(map(lambda g: g*g, grid)) / (2. * sigma_square.to(x.device))
        )
        sumk = torch.sum(k)
        if sumk.tolist() > 0:
            k = k / sumk

        kernel = torch.reshape(k, (1,) * 2 + kernel_shape).to(x.device)
        final_kernel = kernel.repeat((x.shape[1],) * 2 + (1,) * dims)
        conv_f = [F.conv1d, F.conv2d, F.conv3d]
        padding = self.length / 2

        smoothed_x = conv_f[dims - 1](x, final_kernel, padding=padding)

        return smoothed_x


class ScalingLayer(nn.Module):
    def __init__(
            self,
            shape_in,
            dtype=torch.float,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        super(ScalingLayer, self).__init__()
        self.weight = nn.Parameter(
            torch.unsqueeze(
                torch.rand(
                    shape_in, device=device, dtype=dtype, requires_grad=True
                ),
                dim=0
            )
        )
        self.weight.to(device)
        self.bias = nn.Parameter(
            torch.unsqueeze(
                torch.randn(
                    shape_in, device=device, dtype=dtype, requires_grad=True
                ),
                dim=0
            )
        )
        self.bias.to(device)

    def forward(self, x):
        return x * self.weight + self.bias


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer pytorch

    The Layer can handle dense transforms that are meant to give a 'shift'
    from the current position. Therefore, a dense transform gives displacements
    (not absolute locations) at each voxel,

    This code is a reimplementation of
    https://github.com/marianocabezas/voxelmorph/tree/master/ext/neuron in
    pytorch with some liberties taken. The goal is to adapt the code to
    some kind of hybrid method to both do dense registration and mask tracking.
    """

    def __init__(
            self,
            interp_method='linear',
            linear_norm=False,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            **kwargs
    ):
        """
        Parameters:
            :param df_shape: Shape of the deformation field.
            :param interp_method: 'linear' or 'nearest'.
        """
        super(SpatialTransformer, self).__init__(**kwargs)
        self.interp_method = interp_method
        self.device = device
        self.linear_norm = linear_norm

    def forward(self, inputs):
        """
        Transform (interpolation N-D volumes (features) given shifts at each
        location in pytorch. Essentially interpolates volume vol at locations
        determined by loc_shift.
        This is a spatial transform in the sense that at location [x] we now
        have the data from, [x + shift] so we've moved data.
        Parameters
            :param inputs: Input volume to be warped and deformation field.

            :return new interpolated volumes in the same size as df
        """

        # parse shapes
        n_inputs = len(inputs)
        if n_inputs > 2:
            vol, df, mesh = inputs
        else:
            vol, df = inputs
        df_shape = df.shape[2:]
        final_shape = vol.shape[:2] + df_shape
        nb_dims = len(df_shape)
        max_loc = map(lambda s: s - 1, vol.shape[2:])

        # location should be mesh and delta
        if n_inputs > 2:
            loc = [
                mesh[:, d, ...] + df[:, d, ...]
                for d in range(nb_dims)
            ]
        else:
            linvec = map(lambda s: torch.arange(0, s), df_shape)
            mesh = map(
                lambda m_i: m_i.type(dtype=torch.float32),
                torch.meshgrid(linvec)
            )
            loc = [
                mesh[d].to(df.device) + df[:, d, ...]
                for d in range(nb_dims)
            ]
        loc = map(
            lambda (l, m): torch.clamp(l, 0, m),
            zip(loc, max_loc)
        )

        # pre ind2sub setup
        d_size = np.cumprod((1,) + vol.shape[-1:2:-1])[::-1]

        # interpolate
        if self.interp_method == 'linear':
            loc0 = map(torch.floor, loc)

            # clip values
            loc0lst = map(
                lambda (l, m): torch.clamp(l, 0, m),
                zip(loc0, max_loc)
            )

            # get other end of point cube
            loc1 = map(
                lambda (l, m): torch.clamp(l + 1, 0, m),
                zip(loc0, max_loc)
            )
            locs = [
                map(lambda f: f.type(torch.long), loc0lst),
                map(lambda f: f.type(torch.long), loc1)
            ]

            # compute the difference between the upper value and the original value
            # differences are basically 1 - (pt - floor(pt))
            #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
            diff_loc1 = map(lambda (l1, l): l1 - l, zip(loc1, loc))
            diff_loc1 = map(lambda l: torch.clamp(l, 0, 1), diff_loc1)
            diff_loc0 = map(lambda diff: 1 - diff, diff_loc1)
            weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.

            # go through all the cube corners, indexed by a ND binary vector
            # e.g. [0, 0] means this "first" corner in a 2-D "cube"
            cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
            norm_factor = nb_dims * len(cube_pts) / 2.0

            def get_point_value(point):
                subs = map(lambda (i, cd): locs[cd][i], enumerate(point))
                loc_list_p = map(lambda (s, l): s * l, zip(subs, d_size))
                idx_p = torch.sum(torch.stack(loc_list_p, dim=0), dim=0)
                vol_val_flat = torch.stack(
                    map(
                        lambda (idx_i, vol_i): torch.take(vol_i, idx_i),
                        zip(idx_p, vol)
                    ),
                    dim=0
                )

                vol_val = torch.reshape(vol_val_flat, final_shape)
                # get the weight of this cube_pt based on the distance
                # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
                # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
                wts_lst = map(lambda (i, cd): weights_loc[cd][i], enumerate(point))
                if self.linear_norm:
                    wt = sum(wts_lst) / norm_factor
                else:
                    wt = reduce(mul, wts_lst)

                wt = torch.reshape(wt, final_shape)
                return wt * vol_val

            values = map(get_point_value, cube_pts)
            interp_vol = torch.sum(torch.stack(values, dim=0), dim=0)

        elif self.interp_method == 'nearest':
            # clip values
            roundloc = map(
                lambda (l, m): torch.clamp(l, 0, m).type(torch.long),
                zip(map(lambda l: torch.round(l), loc), max_loc)
            )

            # get values
            loc_list = map(lambda (s, l): s * l, zip(roundloc, d_size))
            idx = torch.sum(torch.stack(loc_list, dim=0), dim=0)
            interp_vol_flat = torch.stack(
                map(
                    lambda (idx_i, vol_i): torch.take(vol_i, idx_i),
                    zip(idx, vol)
                ),
                dim=0
            )
            interp_vol = torch.reshape(interp_vol_flat, final_shape)

        return interp_vol
