import numpy as np

from .. import loss

class Square(loss.Loss):

    def __init__(self):
        super().__init__()

    def apply(self, A, Y):
        return 0.5 * np.square(A - Y)

    def prime(self, A, Y):
        return A - Y
