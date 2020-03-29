import numpy as np

class Layer:

    def __init__(self, nInputs, nUnits, act, weight_init="none", is_output=False):

        self.X = None
        self.Z = None
        self.A = None

        self.delta_W = None
        self.delta_B = None

        self.act = act
        self.weight_init = weight_init
        self.is_output = is_output

        if weight_init == "none":
            self.W = np.random.normal(size=(nUnits,nInputs))
            self.B = np.random.normal(size=(nUnits, 1))
        elif weight_init == "basic":
            self.W = np.random.normal(size=(nUnits,nInputs)) / np.sqrt(nInputs)
            self.B = np.random.normal(size=(nUnits, 1)) / np.sqrt(nInputs)
        elif weight_init == "xavier":
            self.W = np.random.uniform(-1.0, 1.0, size=(nUnits,nInputs)) * np.sqrt(6.0 / (nInputs + nUnits))
            self.B = np.random.uniform(-1.0, 1.0, size=(nUnits, 1)) * np.sqrt(6.0 / (nInputs + nUnits))
        elif weight_init == "kaiming":
            self.W = np.random.normal(size=(nUnits,nInputs)) * np.sqrt(2.0 / nInputs)
            self.B = np.zeros((nUnits, 1))
        else:
            raise ValueError("Unknown weight init. method : " + weight_init)

    def forward(self, X):

        self.X = X
        self.Z = self.W.dot(X) + self.B
        self.A = self.act.apply(self.Z)

        return self.A

    def update(self, lRate):

        self.W -= lRate * self.delta_W
        self.B -= lRate * self.delta_B

    def mutate(self, p, scale, relative):

        self.W += (
            scale *
            np.random.binomial(1, p, size=self.W.shape) *
            np.random.normal(size=self.W.shape) *
            (np.abs(self.W) if relative else 1.0)
        )

        self.B += (
            scale *
            np.random.binomial(1, p, size=self.B.shape) *
            np.random.normal(size=self.B.shape) *
            (np.abs(self.B) if relative else 1.0)
        )

    def crossover(self, otherLayer):

        wMask = np.random.binomial(1, 0.5, size=self.W.shape)
        bMask = np.random.binomial(1, 0.5, size=self.B.shape)

        newLayer = Layer(1, 1, self.act, self.weight_init, self.is_output)

        newLayer.W = wMask * self.W + (1 - wMask) * otherLayer.W
        newLayer.B = bMask * self.B + (1 - bMask) * otherLayer.B

        return newLayer

    def copy(self, init=False):

        if init:

            newLayer = Layer(
                self.W.shape[1],
                self.W.shape[0],
                self.act,
                self.weight_init,
                self.is_output
            )

        else:

            newLayer = Layer(1, 1, self.act, self.weight_init, self.is_output)

            newLayer.W = self.W.copy()
            newLayer.B = self.B.copy()

        if self.X is not None:
            newLayer.X = self.X.copy()
        if self.A is not None:
            newLayer.A = self.A.copy()
        if self.Z is not None:
            newLayer.Z = self.Z.copy()

        if self.delta_W is not None:
            newLayer.delta_W = self.delta_W.copy()
        if self.delta_B is not None:
            newLayer.delta_B = self.delta_B.copy()

        return newLayer

    def printTopology(self):

        return f"{self.act.printName()}({self.W.shape},{self.weight_init},{self.is_output})"
