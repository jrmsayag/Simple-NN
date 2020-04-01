from ... import evaluation

class Loss(evaluation.EvalFunc):

    def __init__(self, xs, ys):

        super().__init__()

        self.xs = xs
        self.ys = ys

    @property
    def isSimulation(self):

        return False

    @property
    def isLoss(self):

        return True

    def apply(self, A, Y):
        raise NotImplementedError()

    def prime(self, A, Y):
        raise NotImplementedError()
