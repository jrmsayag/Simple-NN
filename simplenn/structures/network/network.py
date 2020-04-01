import numpy as np

class Network:

    # Initialization

    def __init__(self, layers, actionSelectionMode="argmax"):

        self.layers = layers
        self.actionSelectionMode = actionSelectionMode

    # Acting methods

    def forward(self, X):

        A = X

        for layer in self.layers:

            A = layer.forward(A)

        return A

    def findAction(self, state):

        output = self.forward(state[..., np.newaxis])

        if self.actionSelectionMode == "argmax":
            return np.argmax(output)
        else:
            raise ValueError(f"Uknown action selection mode {self.actionSelectionMode}!")

    # Learning methods

    def backwardBatch(self, loss, batch_X, batch_Y, lRate):

        self.forward(batch_X)

        for layer in self.layers[::-1]:

            A = layer.A
            Z = layer.Z
            X = layer.X

            if layer.is_output:

                dLdZ = loss.prime(A, batch_Y) * layer.act.prime(Z)

            else:

                dLdZ = next_layer.W.T.dot(dLdZ) * layer.act.prime(Z)

            layer.delta_W = dLdZ.dot(X.T) / X.shape[1]
            layer.delta_B = dLdZ.sum(axis=1, keepdims=True) / X.shape[1]

            next_layer = layer

        for layer in self.layers:

            layer.update(lRate)

    def mutate(self, p=0.02, scale=0.05, relative=False):

        for layer in self.layers:

            layer.mutate(p, scale, relative)

    def crossover(self, otherNetwork):

        newLayers = []

        for i in range(len(self.layers)):

            layerA = self.layers[i]
            layerB = otherNetwork.layers[i]

            newLayers.append(layerA.crossover(layerB))

        return Network(newLayers, self.actionSelectionMode)

    # Administrative methods

    def copy(self, init=False):

        return Network([l.copy(init) for l in self.layers], self.actionSelectionMode)

    def printTopology(self):

        return "-".join([l.printTopology() for l in self.layers])
