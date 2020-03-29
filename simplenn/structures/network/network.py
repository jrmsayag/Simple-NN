class Network:

    def __init__(self, loss, layers):

        self.loss = loss
        self.layers = layers

    def forward(self, X):

        A = X

        for layer in self.layers:

            A = layer.forward(A)

        return A

    def backwardBatch(self, batch_X, batch_Y, lRate):

        self.forward(batch_X)

        for layer in self.layers[::-1]:

            A = layer.A
            Z = layer.Z
            X = layer.X

            if layer.is_output:

                dLdZ = self.loss.prime(A, batch_Y) * layer.act.prime(Z)

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

        return Network(self.loss, newLayers)

    def copy(self, init=False):

        return Network(self.loss, [l.copy(init) for l in self.layers])

    def printTopology(self):

        return "-".join([l.printTopology() for l in self.layers])
