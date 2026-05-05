import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DWFL(nn.Module):

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
    
    def forward(self, logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        N, C = logits.shape

        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()

        if targets.dim() == 1:
            t = F.one_hot(targets, num_classes=C).float()
        else: 
            t = targets.float()

        focal_weight = (1 - p) ** self.gamma

        loss = weights.unsqueeze(0) * focal_weight * t * log_p

        return -loss.sum(dim=-1).mean()


def f1_per_class(y_pred: Tensor, y_true: Tensor, num_classes: int) -> Tensor:

    EPS = 1e-15
    
    y_pred = torch.argmax(y_pred, dim=1)

    f1_scores = []

    for _class in range(num_classes):
        tp = ((y_pred == _class) & (y_true == _class)).sum().float()
        fp = ((y_pred == _class) & (y_true != _class)).sum().float()
        fn = ((y_pred != _class) & (y_true == _class)).sum().float()

        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)

        f1 = 2 * precision * recall / (precision + recall + EPS)
        f1_scores.append(f1)

    return torch.stack(f1_scores)
        