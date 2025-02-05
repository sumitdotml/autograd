import numpy as np
from value import Value


class MSELoss:
    def __call__(self, pred, target):
        """
        Computes mean squared error using Value operations

        Args:
            pred (Value): predictions from model
            target (Value): ground truth values

        Returns:
            Value: mean squared error loss
        """
        if not isinstance(pred, Value):
            pred = Value(pred)
        if not isinstance(target, Value):
            target = Value(target)

        squared = (pred - target) ** 2
        return squared.mean()
