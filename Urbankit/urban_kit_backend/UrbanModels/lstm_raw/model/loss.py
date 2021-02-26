import torch.nn.functional as F


def l2_loss(output, target):
    return F.mse_loss(output, target)
