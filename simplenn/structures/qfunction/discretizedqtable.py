import numpy as np

from .. import qfunction

class DiscretizedQTable(qfunction.QTable):

    def __init__(
        self, actions, gamma, alphaRate, alphaCutoff, epsInit, epsFinal, epsRampLen,
        defaultVal, quantums
    ):

        super().__init__(
            actions, gamma, alphaRate, alphaCutoff, epsInit, epsFinal, epsRampLen,
            defaultVal
        )

        self.quantums = np.array(quantums)

    def copy(self):

        newQ = DiscretizedQTable(
            self.actions,
            self.gamma,
            self.alphaRate,
            self.alphaCutoff,
            self.epsInit,
            self.epsFinal,
            self.epsRampLen,
            self.defaultVal,
            self.quantums
        )

        newQ.data = self.copyData()

        return newQ

    def quantize(self, s):

        return tuple((s / self.quantums).astype(np.uint64))

    def getData(self, s, a=None):

        return super().getData(self.quantize(s), a)

    def setData(self, s, a, aData):

        return super().setData(self.quantize(s), a, aData)
