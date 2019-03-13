import torch
import torch.nn.functional as F


def normalized_xcor(var_x, var_y):
    if len(var_x) > 1 and len(var_y) > 1:
        # Init
        var_x_flat = var_x.view(-1)
        var_y_flat = var_y.view(-1)
        # Computation
        var_x_norm = var_x - torch.mean(var_x_flat)
        var_y_norm = var_y - torch.mean(var_y_flat)
        var_xy_norm = torch.abs(torch.sum(var_x_norm * var_y_norm))
        inv_var_x_den = torch.rsqrt(torch.sum(var_x_norm * var_x_norm))
        inv_var_y_den = torch.rsqrt(torch.sum(var_y_norm * var_y_norm))

        return var_xy_norm * inv_var_x_den * inv_var_y_den
    else:
        return torch.mean(torch.abs(var_x - var_y))


def normalized_xcor_loss(var_x, var_y):
    if len(var_x) > 0 and len(var_y) > 0:
        return 1 - normalized_xcor(var_x, var_y)
    else:
        return torch.tensor(0)


def subtraction_loss(var_x, var_y, mask):
    return gradient_mean(var_y - var_x, mask)


def df_loss(var_x, mask):
    return gradient_mean(var_x, mask)


def mahalanobis_loss(var_x, var_y):
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


def torch_hist(var_x, bins=100):
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
        return torch.tensor(h).type(torch.float32).to(var_x.device)
    else:
        return None


def histogram_loss(var_x, var_y):
    '''
    Function that computes a loss based on the histogram of the expected
    and predicted values
    :param var_x: Predicted values.
    :param var_y: Expected values.
    :return: A tensor with the loss
    '''
    # Histogram computation
    loss = 1
    hist_x = torch_hist(var_x)
    if hist_x is not None:
        hist_x = hist_x / torch.sum(hist_x)

        hist_y = torch_hist(var_y)
        if hist_y is not None:
            hist_y = hist_y / torch.sum(hist_y)
            loss = torch.sum(torch.abs(hist_x - hist_y)) / 2

    return loss


def dsc_bin_loss(var_x, var_y):
    '''
    Function to compute the binary dice loss. There is no need to binarise
    the tensors. In fact, we cast the target values to float (for the gradient)
    :param var_x: Predicted values.
    :param var_y: Expected values.
    :return: A tensor with the loss
    '''
    var_y = var_y.type_as(var_x)
    intersection = torch.sum(var_x * var_y)
    sum_x = torch.sum(var_x)
    sum_y = torch.sum(var_y)
    sum_vals = sum_x + sum_y
    dsc_value = (2 * intersection / sum_vals) if sum_vals > 0 else 1.0
    return 1.0 - dsc_value


def gradient_mean(tensor, mask):
    '''
    Function to compute the mean of a multidimensional tensor. We assume that
     the first two dimensions specify the number of samples and channels.
    :param tensor: Input tensor
    :param mask: Mask that defines the region of interest where the loss should
     be evaluated.
    :return: The mean gradient tensor
    '''

    # Init
    tensor_dims = len(tensor.shape)
    data_dims = tensor_dims - 2

    # Since we want this function to be generic, we need a trick to define
    # the gradient on each dimension.
    all_slices = (slice(0, None),) * (tensor_dims - 1)
    first = slice(0, -1)
    last = slice(1, None)
    slices = map(
        lambda i: (
            all_slices[:i + 2] + (first,) + all_slices[i + 2:],
            all_slices[:i + 2] + (last,) + all_slices[i + 2:],
        ),
        range(data_dims)
    )

    gradients = map(
        lambda (si, sf): tensor[si] - tensor[sf],
        slices
    )

    # Remember that gradients moved the image 0.5 pixels while also reducing
    # 1 voxel per dimension. To deal with that we pad before and after. What
    # that actually means is that some gradients are checked twice.
    no_pad = (0, 0)
    pre_pad = (1, 0)
    post_pad = (0, 1)
    paddings = map(
        lambda i: (
            no_pad * i + pre_pad + no_pad * (data_dims - i - 1),
            no_pad * i + post_pad + no_pad * (data_dims - i - 1),
        ),
        range(data_dims)[::-1]
    )

    padded_gradients = map(
        lambda (g, (pi, pf)): F.pad(g, pi) + F.pad(g, pf),
        zip(gradients, paddings)
    )

    gradient = torch.cat(padded_gradients, dim=1)

    mod_grad = torch.sum(gradient * gradient, dim=1, keepdim=True)
    mean_grad = torch.mean(mod_grad[mask])

    return mean_grad


def df_modulo(df, mask):
    modulo = torch.sum(df * df, dim=1, keepdim=True)
    mean_grad = torch.mean(torch.exp(-modulo[mask]))

    return mean_grad
