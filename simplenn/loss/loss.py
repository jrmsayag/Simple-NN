class Loss:

    def __init__(self):
        pass

    def apply(self, A, Y):
        raise NotImplementedError()

    def prime(self, A, Y):
        raise NotImplementedError()
