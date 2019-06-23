from operator import mul
from itertools import product
import torch
import torch.nn.functional as F


def normalised_xcor(var_x, var_y):
    """
        Function that computes the normalised cross correlation between two
         tensors.
        :param var_x: First tensor.
        :param var_y: Second tensor.
        :return: A tensor with the normalised cross correlation
    """
    # Init
    if isinstance(var_x, list) and isinstance(var_y, list):
        # Computation
        var_x_norm = map(lambda v_xi: v_xi - torch.mean(v_xi), var_x)
        var_y_norm = map(lambda v_yi: v_yi - torch.mean(v_yi), var_y)
        var_xy_norm = torch.stack(
            map(lambda (x, y): torch.mean(x * y), zip(var_x_norm, var_y_norm))
        )

        std_x = torch.stack(map(torch.std, var_x))
        std_y = torch.stack(map(torch.std, var_y))

        xcor = torch.abs(var_xy_norm / (std_x * std_y))

        return torch.mean(xcor[xcor == xcor])

    else:
        red_dim = var_x.shape[2:]
        if var_x.numel() > 1 and var_y.numel() > 1:
            # Computation
            var_x_norm = var_x - torch.mean(var_x, dim=red_dim)
            var_y_norm = var_y - torch.mean(var_y, dim=red_dim)
            var_xy_norm = torch.mean(var_x_norm * var_y_norm, dim=red_dim)
            inv_var_x_den = 1 / torch.std(var_x, dim=red_dim)
            inv_var_y_den = 1 / torch.std(var_y, dim=red_dim)
            xcor = torch.abs(var_xy_norm * inv_var_x_den * inv_var_y_den)

            return torch.mean(xcor)
        else:
            return torch.mean(torch.abs(var_x - var_y))


def torch_entropy(var_x, var_y=None, bins=100):
    """
        Function that computes the entropy of a tensor or the joint entropy
         of two tensors using their histogram.
        :param var_x: First tensor.
        :param var_y: Second tensor (optional).
        :param bins: Number of bins for the histogram.
        :return: A tensor with the histogram
    """
    if var_y is None:
        h = torch_hist(var_x, bins=bins)
    else:
        h = torch_hist2(var_x, var_y, bins=bins)
    h = h[h > 0]
    return -torch.sum(h * torch.log(h))


def normalised_mutual_information(var_x, var_y):
    """
        Function that computes the normalised mutual information between two
         tensors, based on their histograms.
        :param var_x: First tensor.
        :param var_y: Second tensor.
        :return: A tensor with the normalised cross correlation
    """
    if len(var_x) > 1 and len(var_y) > 1:
        entr_x = torch_entropy(var_x)
        entr_y = torch_entropy(var_y)
        entr_xy = torch_entropy(var_x, var_y)
        return (entr_x + entr_y - entr_xy) / entr_x
    else:
        return torch.mean(torch.abs(var_x - var_y))


def normalised_xcor_loss(var_x, var_y):
    """
        Loss function based on the normalised cross correlation between two
         tensors. Since we are using gradient descent, the final value is
         1 - the normalised cross correlation.
        :param var_x: First tensor.
        :param var_y: Second tensor.
        :return: A tensor with the loss
    """
    if len(var_x) > 0 and len(var_y) > 0:
        return 1 - normalised_xcor(var_x, var_y)
    else:
        return torch.tensor(0)


def normalised_mi_loss(var_x, var_y):
    """
        Loss function based on the normalised mutual information between two
         tensors. Since we are using gradient descent, the final value is
         1 - the normalised cross correlation.
        :param var_x: First tensor.
        :param var_y: Second tensor.
        :return: A tensor with the loss
    """
    if len(var_x) > 0 and len(var_y) > 0:
        return 1 - normalised_mutual_information(var_x, var_y)
    else:
        return torch.tensor(0)


def subtraction_loss(var_x, var_y, mask):
    """
        Loss function based on the mean gradient of the subtraction between two
        tensors.
        :param var_x: First tensor.
        :param var_y: Second tensor.
        :param mask: Mask that defines the region of interest where the loss
         should be evaluated.
        :return: A tensor with the loss
    """
    return gradient_mean(var_y - var_x, mask)


def weighted_subtraction_loss(var_x, var_y, mask):
    """
        Loss function based on the mean gradient of the subtraction between two
        tensors.
        :param var_x: First tensor.
        :param var_y: Second tensor.
        :param mask: Mask that defines the region of interest where the loss
         should be evaluated.
        :return: A tensor with the loss
    """
    h = torch.tensor([1, 2, 1])
    h_ = torch.tensor([1, 0, -1])
    patch_shape = var_x.shape[2:]
    n_dim = len(patch_shape)
    patches = var_x.shape[0]

    sub_gradient = gradient(var_y - var_x)

    if n_dim in [2, 3]:
        if n_dim is 2:
            sobel = torch.stack(
                (
                    h_ * h.reshape((3, 1)),
                    h * h_.reshape((3, 1))
                ),
                dim=0
            ).type(torch.float32)
            sobel_k = torch.unsqueeze(sobel, 1).to(var_x.device)
            edges = torch.abs(torch.conv2d(var_y, sobel_k, padding=1))
        else:
            sobel = torch.stack(
                (
                    h_ * h.reshape((3, 1)) * h.reshape((3, 1, 1)),
                    h * h_.reshape((3, 1)) * h.reshape((3, 1, 1)),
                    h * h.reshape((3, 1)) * h_.reshape((3, 1, 1)),
                ),
                dim=0
            ).type(torch.float32)
            sobel_k = torch.unsqueeze(sobel, 1).to(var_x.device)
            edges = torch.abs(torch.conv3d(var_y, sobel_k, padding=1))
        weights = torch.sum(edges, dim=1, keepdim=True)
        sum_w = torch.sum(weights.reshape(patches, -1), dim=1)
        sum_w = sum_w.reshape((-1,) + (1,) * (n_dim + 1))
        sum_w = sum_w.repeat((1, 1) + patch_shape)
        weighted_sub = sub_gradient * weights / sum_w
        sub_loss = torch.sum(weighted_sub[mask]) / patches
    else:
        sub_loss = torch.mean(sub_gradient[mask])

    return sub_loss


def df_loss(df, mask):
    """
        Loss function based on mean gradient of a deformation field.
        :param df: A deformation field tensor.
        :param mask: Mask that defines the region of interest where the loss
         should be evaluated.
        :return: A tensor with the loss
    """
    return gradient_mean(df, mask)


def mahalanobis_loss(var_x, var_y):
    """
        Loss function based on the Mahalanobis distance. this distance is
         computed between points and distributions, Therefore, we compute
         the distance between the mean of a distribution and the other
         distribution. To a guarantee that both standard deviations are taken
         into account, we compute a bidirectional loss (from one mean to the
         other distribution and viceversa).
        :param var_x: Predicted values.
        :param var_y: Expected values.
        :return: A tensor with the loss
    """
    # Init
    var_x_flat = var_x.view(-1)
    var_y_flat = var_y.view(-1)
    # Computation
    mu_x = torch.mean(var_x_flat)
    sigma_x = torch.std(var_x_flat)

    mu_y = torch.mean(var_y_flat)
    sigma_y = torch.std(var_y_flat)

    mu_diff = torch.abs(mu_x - mu_y)

    mahal = (sigma_x + sigma_y) * mu_diff

    return mahal / (sigma_x * sigma_y) if (sigma_x * sigma_y) > 0 else mahal


def torch_hist(var_x, bins=100, norm=True):
    """
        Function that computes a histogram using a torch tensor.
        :param var_x: Input tensor.
        :param bins: Number of bins for the histogram.
        :param norm: Whether or not to normalise the histogram. It is useful
        to define a probability density function of a given variable.
        :return: A tensor with the histogram
    """
    min_x = torch.floor(torch.min(var_x)).data
    max_x = torch.ceil(torch.max(var_x)).data
    if max_x > min_x:
        step = (max_x - min_x) / bins
        steps = torch.arange(
            min_x, max_x + step / 10, step
        ).to(var_x.device)
        h = map(
            lambda (min_i, max_i): torch.sum(
                (var_x >= min_i) & (var_x < max_i)
            ),
            zip(steps[:-1], steps[1:])
        )
        torch_h = torch.tensor(h).type(torch.float32).to(var_x.device)
        if norm:
            torch_h = torch_h / torch.sum(torch_h)
        return torch_h
    else:
        return None


def torch_hist2(var_x, var_y, bins=100, norm=True):
    """
        Function that computes a 2D histogram using two torch tensor.
        :param var_x: First tensor.
        :param var_y: Second tensor.
        :param bins: Number of bins for the histogram.
        :param norm: Whether or not to normalise the histogram. It is useful
        to define a joint probability density function of both variables.
        :return: A tensor with the histogram
    """
    min_x = torch.floor(torch.min(var_x)).data
    max_x = torch.ceil(torch.max(var_x)).data
    min_y = torch.floor(torch.min(var_y)).data
    max_y = torch.ceil(torch.max(var_y)).data
    if max_x > min_x and max_y > min_y:
        step_x = (max_x - min_x) / bins
        step_y = (max_y - min_y) / bins
        steps_x = torch.arange(
            min_x, max_x + step_x / 10, step_x
        ).to(var_x.device)
        steps_y = torch.arange(
            min_y, max_y + step_y / 10, step_y
        ).to(var_y.device)
        min_steps = product(steps_x[:-1], steps_y[:-1])
        max_steps = product(steps_x[1:], steps_y[1:])
        h = map(
            lambda ((min_i, min_j), (max_i, max_j)): torch.sum(
                (var_x >= min_i) & (var_x < max_i) &
                (var_y >= min_j) & (var_y < max_j)
            ),
            zip(min_steps, max_steps)
        )
        torch_h2 = torch.tensor(h).type(torch.float32).to(var_x.device)
        if norm:
            torch_h2 = torch_h2 / torch.sum(torch_h2)
        return torch_h2
    else:
        return None


def histogram_loss(var_x, var_y):
    """
        Loss function based on the histogram of the expected and predicted values.
        :param var_x: Predicted values.
        :param var_y: Expected values.
        :return: A tensor with the loss
    """
    # Histogram computation
    loss = 1
    hist_x = torch_hist(var_x)
    if hist_x is not None:
        hist_x = hist_x / torch.sum(hist_x)

        hist_y = torch_hist(var_y)
        if hist_y is not None:
            loss = torch.sum(torch.abs(hist_x - hist_y)) / 2

    return loss


def gradient(tensor):
    """
        Function to compute the gradient of a multidimensional tensor. We
         assume that the first two dimensions specify the number of samples and
         channels.
        :param tensor: Input tensor
        :return: The mean gradient tensor
    """

    # Init
    tensor_dims = len(tensor.shape)
    data_dims = tensor_dims - 2

    # Since we want this function to be generic, we need a trick to define
    # the gradient on each dimension.
    all_slices = (slice(0, None),) * (tensor_dims - 1)
    first = slice(0, -2)
    last = slice(2, None)
    slices = map(
        lambda i: (
            all_slices[:i + 2] + (first,) + all_slices[i + 2:],
            all_slices[:i + 2] + (last,) + all_slices[i + 2:],
        ),
        range(data_dims)
    )

    # Remember that gradients moved the image 0.5 pixels while also reducing
    # 1 voxel per dimension. To deal with that we are technically interpolating
    # the gradient in between these positions. These is the equivalent of
    # computing the gradient between voxels separated one space. 1D ex:
    # [a, b, c, d] -> gradient0.5 = [a - b, b - c, c - d]
    # gradient1 = 0.5 * [(a - b) + (b - c), (b - c) + (c - d)] =
    # = 0.5 * [a - c, b - d] ~ [a - c, b - d]
    no_pad = (0, 0)
    pad = (1, 1)
    paddings = map(
        lambda i: no_pad * i + pad + no_pad * (data_dims - i - 1),
        range(data_dims)[::-1]
    )

    gradients = map(
        lambda (p, (si, sf)): 0.5 * F.pad(tensor[si] - tensor[sf], p),
        zip(paddings, slices)
    )
    gradients = torch.cat(gradients, dim=1)

    return torch.sum(gradients * gradients, dim=1, keepdim=True)


def gradient_mean(tensor, mask):
    """
        Function to compute the mean gradient of a multidimensional tensor. We
         assume that the first two dimensions specify the number of samples and
         channels.
        :param tensor: Input tensor
        :param mask: Mask that defines the region of interest where the loss
         should be evaluated.
        :return: The mean gradient tensor
    """

    # Init
    tensor_dims = len(tensor.shape)
    data_dims = tensor_dims - 2

    # Since we want this function to be generic, we need a trick to define
    # the gradient on each dimension.
    all_slices = (slice(0, None),) * (tensor_dims - 1)
    first = slice(0, -2)
    last = slice(2, None)
    slices = map(
        lambda i: (
            all_slices[:i + 2] + (first,) + all_slices[i + 2:],
            all_slices[:i + 2] + (last,) + all_slices[i + 2:],
        ),
        range(data_dims)
    )

    # Remember that gradients moved the image 0.5 pixels while also reducing
    # 1 voxel per dimension. To deal with that we are technically interpolating
    # the gradient in between these positions. These is the equivalent of
    # computing the gradient between voxels separated one space. 1D ex:
    # [a, b, c, d] -> gradient0.5 = [a - b, b - c, c - d]
    # gradient1 = 0.5 * [(a - b) + (b - c), (b - c) + (c - d)] = [a - c, b - d]
    no_pad = (0, 0)
    pad = (1, 1)
    paddings = map(
        lambda i: no_pad * i + pad + no_pad * (data_dims - i - 1),
        range(data_dims)[::-1]
    )

    gradients = map(
        lambda (p, (si, sf)): 0.5 * F.pad(tensor[si] - tensor[sf], p),
        zip(paddings, slices)
    )
    gradients = torch.cat(gradients, dim=1)

    mod_gradients = torch.sum(gradients * gradients, dim=1, keepdim=True)
    mean_grad = torch.mean(mod_gradients[mask])

    return mean_grad


def df_modulo(df, mask):
    """
        Loss function to maximise the modulo of the deformation field. I first
         implemented it thinking it would help to get large deformations, but
         it doesn't seem to be necessary. In order to maximise, I am using the
         negative value of the modulo (avoiding the square root) and an
         exponential function. That might promote extremely large deformations.
        :param df: A deformation field tensor.
        :param mask: Mask that defines the region of interest where the loss
         should be evaluated.
        :return: The mean modulo tensor
    """
    modulo = torch.sum(df * df, dim=1, keepdim=True)
    mean_grad = torch.mean(torch.exp(-modulo[mask]))

    return mean_grad


def multidsc_loss(pred, target, smooth=1, averaged=True):
    """
    Loss function based on a multi-class DSC metric.
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, n_classes, data_shape]
    :param target: Ground truth values. This tensor can have multiple shapes:
     - [batch_size, n_classes, data_shape]: This is the expected output since
       it matches with the predicted tensor.
     - [batch_size, data_shape]: In this case, the tensor is labeled with
       values ranging from 0 to n_classes. We need to convert it to
       categorical.
    :param smooth: Parameter used to smooth the DSC when there are no positive
     samples.
    :param averaged: Parameter to decide whether to return the average DSC or
     a tensor with the different class DSC values.
    :return: The mean DSC for the batch
    """
    dims = pred.shape
    n_classes = dims[1]
    if target.shape != pred.shape:
        assert torch.max(target) <= n_classes, 'Wrong number of classes for GT'
        target = torch.cat(
            map(lambda i: target == i, range(n_classes)), dim=1
        )
        target = target.type_as(pred)

    reduce_dims = tuple(range(2, len(dims)))
    num = (2 * torch.sum(pred * target, dim=reduce_dims)) + smooth
    den = torch.sum(pred + target, dim=reduce_dims) + smooth
    dsc_k = num / den
    if averaged:
        dsc = 1 - torch.mean(torch.mean(dsc_k, dim=1))
    else:
        dsc = 1 - torch.mean(dsc_k, dim=0)

    return dsc


class GenericLossLayer(torch.nn.Module):
    def __init__(self, func_handle):
        super(GenericLossLayer, self).__init__()
        self.func = func_handle

    def forward(self, pred, target):
        return self.func(pred, target)
