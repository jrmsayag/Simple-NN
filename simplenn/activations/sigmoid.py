import numpy as np

from .. import activations

class Sigmoid(activations.Activation):

    def __init__(self):
        super().__init__()

    def apply(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def prime(self, X):
        A = self.apply(X)
        return A * (1.0 - A)
