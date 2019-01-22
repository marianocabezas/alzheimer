import torch
import torch.nn.functional as F


def normalized_xcor(var_x, var_y):
    if len(var_x) > 0 and len(var_y) > 0:
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
        return torch.zeros(1)


def bidirectional_mahalanobis(var_x, var_y):
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
    max_x = torch.ceil(torch.max(var_x)).data +1
    step = (max_x - min_x) / bins
    steps = torch.arange(min_x, max_x, step).to(var_x.device)
    h = map(
        lambda (min_i, max_i): torch.sum((var_x >= min_i) & (var_x < max_i)),
        zip(steps[:-1], steps[1:])
    )

    return torch.tensor(h).type(torch.float32).to(var_x.device)


def histogram_diff(var_x, var_y):
    # Histogram computation
    hist_x = torch_hist(var_x)
    hist_x = hist_x / torch.sum(hist_x)

    hist_y = torch_hist(var_y)
    hist_y = hist_y / torch.sum(hist_y)

    return torch.sum(torch.abs(hist_x - hist_y)) / 2


def df_gradient_mean(df):
    grad_x_k = [
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    ]
    grad_y_k = [
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    ]
    grad_z_k = [
        [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ]
    grad_k = [grad_x_k, grad_y_k, grad_z_k]

    grad_k_tensor = torch.tensor(grad_k, dtype=torch.float32).to(df.device)
    gradient = F.conv3d(df, grad_k_tensor.repeat(3, 1, 1, 1, 1))
    gradient_square = gradient * gradient
    mod_grad = torch.sqrt(torch.sum(gradient_square, 1))
    mean_grad = torch.mean(mod_grad.view(-1))

    return mean_grad


def longitudinal_loss(predictions, inputs):
    # Init
    moved, moved_mask, df = predictions
    source, _, mask, target = inputs

    # global xcor
    # global_xcor = normalized_xcor(moved, target)
    global_loss = torch.nn.MSELoss()(moved, target)

    # lesion DSC
    moved_mask = moved_mask > 0
    mask = mask > 0
    intersect = torch.sum(moved_mask & mask)
    moved_mask_vol = torch.sum(moved_mask)
    mask_vol = torch.sum(mask)
    dsc_masks = 2.0 * intersect / (moved_mask_vol + mask_vol)

    # histogram loss
    hist_loss = histogram_diff(moved[moved_mask], source[mask])

    # global_loss = (1 - global_xcor)
    mask_diff_loss = 1 - dsc_masks

    return global_loss, mask_diff_loss, hist_loss
