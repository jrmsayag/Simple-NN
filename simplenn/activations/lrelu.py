from .. import activations

class LRelu(activations.Activation):

    def __init__(self, l=0.0):
        super().__init__()
        self.l = l

    def apply(self, X):
        A = X.copy()
        A[A <= 0.0] *= self.l
        return A

    def prime(self, X):
        Ap = X.copy()
        Ap[Ap < 0.0] = self.l
        Ap[Ap > 0.0] = 1.0
        return Ap
