import numpy as np
from chibigrad.tensor import Tensor
from chibigrad.arithmetic import Mean


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
        print("\nMSELoss forward pass:")
        print(f"Predictions shape: {pred.data.shape}")
        print(f"Target shape: {target.data.shape}")
        
        if not isinstance(pred, Tensor):
            pred = Tensor(pred)
        if not isinstance(target, Tensor):
            target = Tensor(target)

        # Computing MSE using our operations
        diff = pred - target
        print(f"Difference shape: {diff.data.shape}")
        diff.requires_grad = pred.requires_grad
        diff.retain_grad()  # Retain gradients for intermediate values

        squared = diff * diff
        print(f"Squared difference shape: {squared.data.shape}")
        squared.requires_grad = pred.requires_grad
        squared.retain_grad()  # Retain gradients for intermediate values

        # Match PyTorch's implementation exactly
        n = np.prod(pred.data.shape)
        print(f"Batch size (n): {n}")
        loss = squared.sum() / (2.0 * n)  # Note the factor of 2.0
        print(f"Loss value: {loss.data}")
        
        # Ensure gradient tracking if input requires grad
        if pred.requires_grad:
            loss.requires_grad = True
            print("Loss requires_grad set to True")
        
        return loss
