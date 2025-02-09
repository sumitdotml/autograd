from chibigrad.tensor import Tensor
import numpy as np
from chibigrad.module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # initializing weights and bias with proper requires_grad
        self.weight = Tensor(
            np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        """
        Forward pass with proper gradient tracking
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # matrix multiplication with weight
        output = x @ self.weight.T
        output.requires_grad = True
        output.retain_grad()

        # adding bias
        output = output + self.bias
        output.requires_grad = True
        output.retain_grad()

        return output

    def forward(self, x):
        return x @ self.weight + self.bias
