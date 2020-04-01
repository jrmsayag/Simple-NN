import numpy as np

from .. import loss

class NegLogLike(loss.Loss):

    def __init__(self, xs, ys):
        super().__init__(xs, ys)

    def apply(self, A, Y):
        return -(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    def prime(self, A, Y):
        return -Y / A + (1 - Y) / (1 - A)
