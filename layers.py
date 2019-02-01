from operator import mul
import torch
import itertools
import numpy as np
from torch import nn


class ScalingLayer(nn.Module):
    def __init__(
            self,
            shape_in,
            dtype=torch.float,
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ):
        super(ScalingLayer, self).__init__()
        self.weight = nn.Parameter(
            torch.unsqueeze(torch.rand(shape_in, device=device, dtype=dtype, requires_grad=True), dim=0)
        )
        self.weight.to(device)
        self.bias = nn.Parameter(
            torch.unsqueeze(torch.randn(shape_in, device=device, dtype=dtype, requires_grad=True), dim=0)
        )
        self.bias.to(device)

    def forward(self, x):
        return x * self.weight + self.bias


# TODO: Make it pytorch
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle dense transforms that are meant to give a 'shift'
    from the current position. Therefore, a dense transform gives displacements
    (not absolute locations) at each voxel,

    This code is a reimplementation of
    https://github.com/marianocabezas/voxelmorph/tree/master/ext/neuron in
    pytorch with some liberties taken. The goal is to adapt the code to
    some kind of hybrid method to both do dense registration and mask tracking.
    TODO: Seriously understand the code and tweak it to our needs.
    """

    def __init__(
            self,
            interp_method='linear',
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
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

    def forward(self, input):
        """
        Transform (interpolation N-D volumes (features) given shifts at each
        location in pytorch. Essentially interpolates volume vol at locations
        determined by loc_shift.
        This is a spatial transform in the sense that at location [x] we now
        have the data from, [x + shift] so we've moved data.
        Parameters
            :param input: Input volume to be warped and deformation field.

            :return new interpolated volumes in the same size as df
        """

        # parse shapes
        vol, df = input
        df_shape = df.shape[2:]
        nb_dims = len(df_shape)
        max_loc = map(lambda s: s - 1, vol.shape[2:])

        # location should be mesh and delta
        linvec = map(lambda s: torch.arange(0, s), df_shape)
        mesh = map(
            lambda m_i: m_i.type(dtype=torch.float32),
            torch.meshgrid(linvec)
        )
        loc = [mesh[d].cuda(df.device) + df[:, d, ...] for d in range(nb_dims)]
        loc = map(
            lambda (l, m): torch.clamp(l, 0, m),
            zip(loc, max_loc)
        )

        # pre ind2sub setup
        d_size = np.cumprod((1,) + vol.shape[2:-1])

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
                map(lambda f: f.type(dtype=torch.int32), loc0lst),
                map(lambda f: f.type(dtype=torch.int32), loc1)
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
                subs = map(lambda (l, cd): l[cd], zip(locs, point))

                loc_list_p = map(lambda (s, l): s * l, zip(subs, d_size))
                idx_p = torch.sum(torch.stack(loc_list_p, dim=0), dim=0)
                vol_val_flat = torch.take(vol, idx_p.type(torch.long))
                vol_val = torch.reshape(vol_val_flat, vol.shape)

                # get the weight of this cube_pt based on the distance
                # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
                # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
                wts_lst = map(lambda (l, cd): l[cd], zip(weights_loc, point))
                # wt = reduce(mul, wts_lst)
                wt = sum(wts_lst) / len(wts_lst)
                wt = torch.reshape(wt, vol.shape)
                return wt * vol_val

            values = map(get_point_value, cube_pts)
            interp_vol = torch.sum(torch.stack(values, dim=0), dim=0)

        elif self.interp_method == 'nearest':
            # clip values
            roundloc = map(
                lambda (l, m): torch.clamp(l, 0, m),
                zip(map(lambda l: torch.round(l), loc), max_loc)
            )

            # get values
            loc_list = map(lambda (s, l): s * l, zip(roundloc, d_size))
            idx = torch.sum(torch.stack(loc_list, dim=0), dim=0)
            interp_vol_flat = torch.take(vol, idx.type(torch.long))
            interp_vol = torch.reshape(interp_vol_flat, vol.shape)

        return interp_vol
