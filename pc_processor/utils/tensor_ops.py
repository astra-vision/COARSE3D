
import torch

def tensorNormalize(x):
    """
    x: N C H W
    """
    x_max = x.max(2, keepdim=True)[0].max(3, keepdim=True)[0].expand_as(x)
    x_min = x.min(2, keepdim=True)[0].min(3, keepdim=True)[0].expand_as(x)
    out = (x - x_min) / (x_max - x_min + 1e-6)
    return out

# def tensorNormalize(x):
#     x_max = x.max()
#     x_min = x.min()
#     out = (x - x_min)/(x_max-x_min + 1e-6)
#     return out

def entropy(x, mask=None):
    """
    x: N C H W, results of softmax
    mask: N H W
    """
    
    if mask is not None:
        if mask.sum() == 0:
            raise ValueError("no valid data in mask: mask.sum()==0")
        x = x * mask.unsqueeze(1)
    entropy = -x * torch.log(x.clamp(min=1e-5))
    mean_entropy = (entropy * mask.unsqueeze(1)).sum()/mask.sum() / x.size(1)
    return mean_entropy
    