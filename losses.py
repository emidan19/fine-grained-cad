import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def focal_loss(logits, labels, gamma=2, alpha=None):
    logpt = F.log_softmax(logits, dim=1)
    logpt = logpt.gather(1, labels)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    if alpha is not None:
        if isinstance(alpha, list):
            alpha = torch.Tensor(alpha)
        if alpha.type() != logits.data.type():
            alpha = alpha.type_as(logits.data)
        at = alpha.gather(0, labels.data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1-pt)**gamma * logpt
    return torch.mean(loss)


def cb_loss(logits, labels, samples_per_cls, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks. Whenever
    gamma = 0, beta > 0, this is equivalent to focal loss. Whenever gamma = 0,
    beta = 0, this is equivalent to multi-class cross-entropy loss.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      beta: float. 0 -> no class balancing, 1 -> Inverse class freq
      gamma: float. 0 -> Disable downweighting, inf -> Downweight 'easy' samples.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """

    no_of_classes = len(samples_per_cls)
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * no_of_classes

    return focal_loss(logits, labels, gamma, torch.tensor(weights))
