class BackProp:

    def __init__(self):

        pass

    def run(self, net, xs, ys, batchsize, lRate, epochs, verboseFreq=0, recordFreq=0):

        self.networks = []

        for i in range(epochs):

            j = 0

            while j * batchsize < xs.shape[1]:

                xBatch = xs[:, j*batchsize:(j+1)*batchsize]
                yBatch = ys[:, j*batchsize:(j+1)*batchsize]

                j += 1

                net.backwardBatch(xBatch, yBatch, lRate)

            if verboseFreq and i % verboseFreq == 0:

                error = net.loss.apply(net.forward(xs), ys).sum()
                print(f"Epoch {i}: {error}")

            if recordFreq and i % recordFreq == 0:

                self.record(net, xs, ys, lRate)

        self.record(net, xs, ys, lRate)

        return net

    def record(self, net, xs, ys, lRate):

        netCopy = net.copy()

        netCopy.backwardBatch(xs, ys, lRate)

        self.networks.append(netCopy)
