from chibigrad.tensor import Tensor
import numpy as np
from chibigrad.module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize weights and bias
        self.in_features = in_features
        self.out_features = out_features
        
        # Match PyTorch's Kaiming initialization
        bound = 1 / np.sqrt(in_features)
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)),
            requires_grad=True
        )
        self.bias = Tensor(
            np.random.uniform(-bound, bound, (out_features,)),
            requires_grad=True
        )

    def __call__(self, x):
        """
        Forward pass with proper gradient tracking
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Ensure input has requires_grad if needed
        requires_grad = x.requires_grad or self.weight.requires_grad or self.bias.requires_grad
        
        # matrix multiplication with weight (note: weight is already in correct orientation)
        output = x @ self.weight.T
        output.requires_grad = requires_grad
        
        # adding bias with broadcasting
        output = output + self.bias
        output.requires_grad = requires_grad
        
        return output

    def forward(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        """Return the parameters of the layer"""
        return [self.weight, self.bias]

    def zero_grad(self):
        """Zero out the gradients of the parameters"""
        self.weight.grad = None
        self.bias.grad = None
