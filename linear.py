from value import Value
import math
import random
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        scale = 1 / np.sqrt(in_features)
        self.W = Value(np.random.randn(in_features, out_features) * scale)
        self.b = Value(np.zeros(out_features))

        # storing dimensions for shape checking
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        return x @ self.W + self.b
