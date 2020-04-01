class BackProp:

    def __init__(self):

        pass

    def run(self, net, loss, batchsize, lRate, epochs, verboseFreq=0, recordFreq=0):

        self.networks = []

        for i in range(epochs):

            j = 0

            while j * batchsize < loss.xs.shape[1]:

                xBatch = loss.xs[:, j*batchsize:(j+1)*batchsize]
                yBatch = loss.ys[:, j*batchsize:(j+1)*batchsize]

                j += 1

                net.backwardBatch(loss, xBatch, yBatch, lRate)

            if verboseFreq and i % verboseFreq == 0:

                error = loss.apply(net.forward(loss.xs), loss.ys).sum()
                print(f"Epoch {i}: {error}")

            if recordFreq and i % recordFreq == 0:

                self.record(net, loss, lRate)

        self.record(net, loss, lRate)

        return net

    def record(self, net, loss, lRate):

        netCopy = net.copy()

        netCopy.backwardBatch(loss, loss.xs, loss.ys, lRate)

        self.networks.append(netCopy)
