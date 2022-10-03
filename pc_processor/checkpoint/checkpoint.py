import torch


class CheckPoint(object):
    def __init__(self):
        pass

    def saveModel(self, model, path):
        if isinstance(model, torch.nn.Module):
            saved_model = model.cpu()
