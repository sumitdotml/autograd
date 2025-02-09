from value import Value
import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        scale = Linear.scale(in_features)
        self.W = Value(Linear.init_weights(in_features, out_features, scale))
        self.b = Value(Linear.init_bias(out_features))

        # storing dimensions for shape checking
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        # dimension validation
        assert x.data.shape[-1] == self.in_features, \
            f"Input features {x.data.shape[-1]} != {self.in_features}"
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

    @staticmethod
    def scale(in_features):
        return 1 / np.sqrt(in_features)

    @staticmethod
    def init_weights(in_features, out_features, scale):
        return np.random.randn(in_features, out_features) * scale

    @staticmethod
    def init_bias(out_features):
        return np.zeros(out_features)
