from tensor import Tensor
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(np.random.randn(in_features, out_features) / np.sqrt(in_features))
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias