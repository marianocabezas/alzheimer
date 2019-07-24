from __future__ import division
import torch
import torch.nn.functional as F


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

    reduce_dims = tuple(range(1, len(dims)))
    num = (2 * torch.sum(pred * target, dim=reduce_dims[1:])) + smooth
    den = torch.sum(pred + target, dim=reduce_dims[1:]) + smooth
    dsc_k = num / den
    if averaged:
        class_sum = torch.sum(target, dim=reduce_dims[1:])
        total_sum = torch.unsqueeze(torch.sum(target, dim=reduce_dims), dim=1)
        if (total_sum > 0).all():
            class_pr = 1 - class_sum / total_sum
            dsc = 1 - torch.sum(dsc_k * class_pr) / dims[0]
        else:
            dsc = 1 - torch.mean(dsc_k)
        # dsc = 1 - torch.mean(dsc_k)
    else:
        dsc = 1 - torch.mean(dsc_k, dim=0)

    return torch.clamp(dsc, 0., 1.)


class GenericLossLayer(torch.nn.Module):
    def __init__(self, func_handle):
        super(GenericLossLayer, self).__init__()
        self.func = func_handle

    def forward(self, pred, target):
        return self.func(pred, target)
