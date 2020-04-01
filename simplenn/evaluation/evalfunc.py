class EvalFunc:

    def __init__(self):

        pass

    @property
    def isSimulation(self):

        raise NotImplementedError()

    @property
    def isLoss(self):

        raise NotImplementedError()
