import numpy as np

from .. import activations

class TanH(activations.Activation):

    def __init__(self):
        super().__init__()

    def apply(self, X):
        return (2.0 / (1.0 + np.exp(-2.0*X))) - 1.0

    def prime(self, X):
        A = self.apply(X)
        return 1.0 - A*A
