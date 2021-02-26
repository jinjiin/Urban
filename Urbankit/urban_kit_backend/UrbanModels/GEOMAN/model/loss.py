import torch.nn.functional as F
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

def l2_loss(output, target):
    return F.mse_loss(output, target)
