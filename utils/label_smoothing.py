import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothing(nn.Module):
    def __init__(self, alpha=0.1):
        super(LabelSmoothing, self).__init__()
        self.alpha = alpha
        self.certainty = 1.0 - alpha
        self.criterion = nn.KLDivLoss(reduction='mean')

    def forward(self, x, y):
        b, c = x.shape
        label = torch.full((b, c), self.alpha / (c - 1)).to(y.device)
        label = label.scatter(1, y.unsqueeze(1), self.certainty)
        return self.criterion(F.log_softmax(x, dim=1), label)