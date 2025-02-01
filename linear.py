import numpy as np
from value import Value


class Linear:
    def __init__(self, in_features, out_features):

        # scale here is called Xavier/Glorot initialization
        scale = 1 / np.sqrt(in_features)
        W_data = np.random.randn(in_features, out_features) * scale
        b_data = np.zeros(out_features)

        # initializing weights and biases as Value objects
        self.W = Value(W_data, label="W")
        self.b = Value(b_data, label="b")

        # storing dimensions for shape checking
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        # ensuring x.data is at least 2D
        if x.data.ndim == 1:
            x.data = x.data.reshape(1, -1)

        # checking the shape
        if x.data.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input with last dimension {self.in_features}"
                f"but got shape {x.data.shape}"
            )

        # performing matrix multiplication and addition using Value objects
        out = (x @ self.W) + self.b
        return out
