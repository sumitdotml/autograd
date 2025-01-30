import numpy as np
from value import Value

class MSELoss:
    def __init__(self):
        pass

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
            
        # getting number of elements for mean calculation
        n = np.prod(pred.data.shape)
        
        # computing MSE using Value operations
        diff = pred - target
        squared_diff = diff ** 2
        loss = squared_diff * (1.0/n)
        
        return loss