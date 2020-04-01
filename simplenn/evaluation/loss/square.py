import numpy as np

from .. import loss

class Square(loss.Loss):

    def __init__(self, xs, ys):
        super().__init__(xs, ys)

    def apply(self, A, Y):
        return 0.5 * np.square(A - Y)

    def prime(self, A, Y):
        return A - Y
