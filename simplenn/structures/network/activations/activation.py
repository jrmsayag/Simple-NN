class Activation:

    def __init__(self):
        pass

    def apply(self, X):
        raise NotImplementedError()

    def prime(self, X):
        raise NotImplementedError()

    def printName(self):
        raise NotImplementedError()
