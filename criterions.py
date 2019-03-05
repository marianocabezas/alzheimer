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
    return im_gradient_mean(var_y - var_x, mask)


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
    var_y = var_y.type_as(var_x)
    intersection = torch.sum(var_x * var_y)
    sum_x = torch.sum(var_x)
    sum_y = torch.sum(var_y)
    sum_vals = sum_x + sum_y
    dsc_value = (2 * intersection / sum_vals) if sum_vals > 0 else 1.0
    return 1.0 - dsc_value


def df_gradient_mean(df, mask):
    grad_v = torch.tensor([-1, 1, 0], dtype=torch.float32).to(df.device)
    grad_x_k = torch.reshape(grad_v, (1, 1, -1)).repeat((3, 3, 1))
    grad_y_k = torch.reshape(grad_v, (1, -1, 1)).repeat((3, 1, 3))
    grad_z_k = torch.reshape(grad_v, (-1, 1, 1)).repeat((1, 3, 3))

    grad_x = F.conv3d(df, grad_x_k.repeat((3, 3, 1, 1, 1)), padding=1)
    grad_y = F.conv3d(df, grad_y_k.repeat((3, 3, 1, 1, 1)), padding=1)
    grad_z = F.conv3d(df, grad_z_k.repeat((3, 3, 1, 1, 1)), padding=1)

    gradient = torch.cat([grad_x, grad_y, grad_z], dim=1)
    gradient = torch.sum(gradient * gradient, dim=1, keepdim=True)
    mean_grad = torch.mean(gradient[mask])

    return mean_grad


def im_gradient_mean(im, mask):
    n_images = im.shape[1]
    grad_v0 = torch.tensor([-1, 1, 0], dtype=torch.float32).to(im.device)
    grad_x_k0 = torch.reshape(grad_v0, (1, 1, -1)).repeat((3, 3, 1))
    grad_y_k0 = torch.reshape(grad_v0, (1, -1, 1)).repeat((3, 1, 3))
    grad_z_k0 = torch.reshape(grad_v0, (-1, 1, 1)).repeat((1, 3, 3))

    grad_v1 = torch.tensor([0, 1, -1], dtype=torch.float32).to(im.device)
    grad_x_k1 = torch.reshape(grad_v1, (1, 1, -1)).repeat((3, 3, 1))
    grad_y_k1 = torch.reshape(grad_v1, (1, -1, 1)).repeat((3, 1, 3))
    grad_z_k1 = torch.reshape(grad_v1, (-1, 1, 1)).repeat((1, 3, 3))

    grad_x = F.conv3d(
        im,
        grad_x_k0.repeat((n_images, n_images, 1, 1, 1)),
        padding=1
    ) + F.conv3d(
        im,
        grad_x_k1.repeat((n_images, n_images, 1, 1, 1)),
        padding=1
    )
    grad_y = F.conv3d(
        im,
        grad_y_k0.repeat((n_images, n_images, 1, 1, 1)),
        padding=1
    ) + F.conv3d(
        im,
        grad_y_k1.repeat((n_images, n_images, 1, 1, 1)),
        padding=1
    )
    grad_z = F.conv3d(
        im,
        grad_z_k0.repeat((n_images, n_images, 1, 1, 1)),
        padding=1
    ) + F.conv3d(
        im,
        grad_z_k1.repeat((n_images, n_images, 1, 1, 1)),
        padding=1
    )
    gradient = torch.cat([grad_x, grad_y, grad_z], dim=1)
    gradient = torch.sum(gradient * gradient, dim=1, keepdim=True)
    mean_grad = torch.mean(gradient[mask])

    return mean_grad


def df_modulo(df, mask):
    modulo = torch.sum(df * df, dim=1, keepdim=True)
    mean_grad = torch.mean(torch.exp(-modulo[mask]))

    return mean_grad
