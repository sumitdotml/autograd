import numpy as np
from autograd.tensor import Tensor
# from autograd.arithmetic import Mean


class MSELoss:
    def __call__(self, pred, target):
        """
        Computes mean squared error loss with gradient support

        Args:
            pred (Tensor): predictions from model
            target (Tensor): ground truth values

        Returns:
            Tensor: mean squared error loss
        """
        if not isinstance(pred, Tensor):
            pred = Tensor(pred)
        if not isinstance(target, Tensor):
            target = Tensor(target)

        # Computing MSE using our operations
        diff = pred - target
        diff.requires_grad = True
        diff.retain_grad()

        squared = diff * diff
        squared.requires_grad = True
        squared.retain_grad()

        # Using sum and scaling by batch size
        n = np.prod(pred.data.shape)
        loss = squared.sum() / n
        loss.requires_grad = True

        return loss
