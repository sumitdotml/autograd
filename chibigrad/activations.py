from chibigrad.operation import Operation
import numpy as np


class ReLU(Operation):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of ReLU.
        Saves the input for backward pass.
        """
        ctx.save_for_backward(x)  # saving input for backward pass
        # ensuring output has same dtype as input
        return np.maximum(0, x).astype(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of ReLU.
        Gradient is 1 where input was > 0, and 0 elsewhere.
        """
        x, = ctx.saved_tensors
        # Ensure we're working with numpy arrays
        grad_output = np.asarray(grad_output)
        x = np.asarray(x)
        
        # creating gradient mask: 1 where input > 0, 0 elsewhere
        grad_mask = (x > 0).astype(x.dtype)
        return (grad_output * grad_mask).astype(grad_output.dtype)