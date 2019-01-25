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


def df_gradient_mean(df, mask, weight=1000):
    grad_v = torch.tensor([-1, 0, 1], dtype=torch.float32).to(df.device)
    grad_x_k = torch.reshape(grad_v, (1, 1, -1)).repeat((3, 3, 1))
    grad_y_k = torch.reshape(grad_v, (1, -1, 1)).repeat((3, 1, 3))
    grad_z_k = torch.reshape(grad_v, (-1, 1, 1)).repeat((1, 3, 3))
    grad_k_tensor = torch.stack([grad_x_k, grad_y_k, grad_z_k], dim=0)

    gradient = F.conv3d(df, grad_k_tensor.repeat(3, 1, 1, 1, 1), padding=1)
    gradient = torch.sum(gradient * gradient, dim=1, keepdim=True)
    mean_grad = torch.mean(gradient[mask])

    return weight * mean_grad


def df_modulo(df, mask):
    modulo = torch.sum(df * df, dim=1, keepdim=True)
    mean_grad = torch.mean(torch.exp(-modulo[mask]))

    return mean_grad


def longitudinal_loss(predictions, inputs, roi):
    # Init
    moved, moved_mask, df = predictions
    source, _, mask, target = inputs
    # moved_mask = moved_mask > 0
    # mask = mask > 0

    # global loss
    global_xcor = normalized_xcor(moved[roi], target[roi])
    global_loss = (1 - global_xcor)
    # global_loss = torch.nn.MSELoss()(moved, target)

    # histogram loss
    # hist_loss = histogram_diff(moved[moved_mask], source[mask])

    # lesion DSC
    # intersect = torch.sum(moved_mask & mask)
    # moved_mask_vol = torch.sum(moved_mask)
    # mask_vol = torch.sum(mask)
    # dsc_masks = 2.0 * intersect / (moved_mask_vol + mask_vol)
    # mask_diff_loss = 1 - dsc_masks

    # gradient loss
    df_loss = df_gradient_mean(df, roi)

    # modulo loss
    # df_mod = df_modulo(df, roi)

    # return global_loss, mask_diff_loss, hist_loss
    return global_loss, df_loss
