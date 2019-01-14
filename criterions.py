import torch
import torch.nn.functional as F


def normalized_xcor(var_x, var_y):
    # Init
    var_x_flat = var_x.view(-1)
    var_y_flat = var_y.view(-1)
    # Computation
    var_x_norm = var_x - torch.mean(var_x_flat)
    var_y_norm = var_y - torch.mean(var_y_flat)
    var_x_den = torch.rsqrt(torch.sum(var_x_norm * var_x_norm))
    var_y_den = torch.rsqrt(torch.sum(var_y_norm * var_y_norm))

    return torch.sum(var_x_norm * var_y_norm) * var_x_den * var_y_den


def longitudinal_loss(predictions, inputs, max_diff=5):
    moved, moved_mask, df = predictions
    source, target, mask = inputs

    global_xcor = normalized_xcor(moved, target)
    masked_xcor = normalized_xcor(moved[moved_mask > 0], source[mask > 0])
    diff_masks = torch.abs(torch.sum(moved_mask) - torch.sum(mask))
    exp_diff = torch.exp(diff_masks - max_diff)

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

    grad_k_tensor = torch.tensor(grad_k, dtype=torch.float32)
    gradient = F.conv3d(df, grad_k_tensor.repeat(3, 1, 1, 1, 1))
    mod_grad = torch.sqrt(torch.sum(gradient * gradient, 1))
    mean_grad = torch.mean(mod_grad.view(-1))

    return (1 - global_xcor) + (1 - masked_xcor) + exp_diff + mean_grad


