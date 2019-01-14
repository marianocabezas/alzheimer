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

    def __init__(self,
                 interp_method='linear',
                 **kwargs):
        """
        Parameters:
            :param interp_method: 'linear' or 'nearest'
        """
        self.interp_method = interp_method

        super(self.__class__, self).__init__(**kwargs)

    def forward(self, input):
        """
        Transform (interpolation N-D volumes (features) given shifts at each location in pytorch
        Essentially interpolates volume vol at locations determined by loc_shift.
        This is a spatial transform in the sense that at location [x] we now have the data from,
        [x + shift] so we've moved data.
        Parameters
            :param vol: Input volume to be warped
            :param df: Deformation field

        Return:
            new interpolated volumes in the same size as df
        """

        # parse shapes
        vol, df = input
        volshape = df.shape[:-1]
        nb_dims = len(volshape)

        # location should be mesh and delta
        linvec = [torch.range(0, d) for d in volshape]
        mesh = torch.meshgrid(linvec)
        loc = [mesh[d].type('float32') + df[..., d] for d in range(nb_dims)]

        # test single
        return self.interpn(vol, loc)

    def interpn(self, vol, loc):
        """
        N-D gridded interpolation in tensorflow
        vol can have more dimensions than loc[i], in which case loc[i] acts as a slice
        for the first dimensions
        Parameters:
            :param vol: volume with size vol_shape or [*vol_shape, nb_features]
            :param loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
                each tensor has to have the same size (but not nec. same size as vol)
                or a tensor of size [*new_vol_shape, D]
        Returns:
            new interpolated volume of the same size as the entries in loc
        """

        if isinstance(loc, (list, tuple)):
            loc = torch.stack(loc, -1)

        # since loc can be a list, nb_dims has to be based on vol.
        nb_dims = loc.shape[-1]

        if nb_dims != len(vol.shape[:-1]):
            raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                            % (nb_dims, len(vol.shape[:-1])))

        if nb_dims > len(vol.shape):
            raise Exception("Loc dimension %d does not match volume dimension %d"
                            % (nb_dims, len(vol.shape)))

        if len(vol.shape) == nb_dims:
            vol = torch.unsqueeze(vol, -1)

        # flatten and float location Tensors
        loc.type('float32')

        # pre ind2sub setup
        k = np.cumprod(vol.shape[1:][::-1])

        # interpolate
        if self.interp_method == 'linear':
            loc0 = torch.floor(loc)

            # clip values
            max_loc = [d - 1 for d in vol.get_shape().as_list()]
            loc0lst = [torch.clamp(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

            # get other end of point cube
            loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
            locs = [[f.type('int32') for f in loc0lst], [f.type('int32') for f in loc1]]

            # compute the difference between the upper value and the original value
            # differences are basically 1 - (pt - floor(pt))
            #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
            diff_loc1 = [loc1[d] - loc[..., d] for d in range(nb_dims)]
            diff_loc0 = [1 - d for d in diff_loc1]
            weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.

            # go through all the cube corners, indexed by a ND binary vector
            # e.g. [0, 0] means this "first" corner in a 2-D "cube"
            cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
            interp_vol = 0

            for c in cube_pts:
                subs = [locs[c[d]][d] for d in range(nb_dims)]

                idx = subs[-1]
                for i, v in enumerate(subs[:-1][::-1]):
                    idx += v * k[i]
                vol_val = torch.gather(torch.reshape(vol, [-1, vol.shape[-1]]), 0, idx)

                # get the weight of this cube_pt based on the distance
                # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
                # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
                wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
                # tf stacking is slow, we we will use prod_n()
                # wlm = tf.stack(wts_lst, axis=0)
                # wt = tf.reduce_prod(wlm, axis=0)
                wt = reduce(mul, wts_lst)
                wt = torch.unsqueeze(wt, -1)

                # compute final weighted value for each cube corner
                interp_vol += wt * vol_val

        elif self.interp_method == 'nearest':
            roundloc = torch.round(loc).type('int32')

            # clip values
            max_loc = [(d - 1).type('int32') for d in vol.shape]
            roundloc = [torch.clamp(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

            # get values
            idx = roundloc[-1]
            for i, v in enumerate(roundloc[:-1][::-1]):
                idx += v * k[i]
            interp_vol = torch.gather(torch.reshape(vol, [-1, vol.shape[-1]]), 0, idx)

        return interp_vol
