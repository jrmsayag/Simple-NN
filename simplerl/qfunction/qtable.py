from .. import qfunction

class QTable(qfunction.BaseQFunction):

    def __init__(self, actions, gamma, alphaRate, alphaCutoff, epsInit, epsFinal, epsRampLen, defaultVal):

        super().__init__(actions, epsInit, epsFinal, epsRampLen, defaultVal)

        self.gamma = gamma

        self.alphaRate = alphaRate
        self.alphaCutoff = alphaCutoff

        self.data = {}

    def copy(self):

        newQ = QTable(
            self.actions,
            self.gamma,
            self.alphaRate,
            self.alphaCutoff,
            self.epsInit,
            self.epsFinal,
            self.epsRampLen,
            self.defaultVal
        )

        newQ.data = self.copyData()

        return newQ

    def copyData(self):

        newData = {}

        for s, sData in self.data.items():

            newSData = {}
            newData[s] = newSData

            for a, aData in sData.items():

                newSData[a] = {
                    self.N_IDX: aData[self.N_IDX],
                    self.Q_IDX: aData[self.Q_IDX]
                }

        return newData

    def getData(self, s, a=None):

        sData = self.data.get(s, {})

        if a is None:
            return sData
        else:
            return sData.get(a, {})

    def setData(self, s, a, aData):

        sData = self.data.get(s, {})
        if not sData:
            self.data[s] = sData

        sData[a] = aData

    def update(self, newS, oldS, a, reward, done, targetPolicyQ=None, targetValueQ=None):

        aData = self.getData(oldS, a)

        alpha = self.computeAlpha(aData)

        if targetPolicyQ is None:
            targetPolicyQ = self
        if targetValueQ is None:
            targetValueQ = self

        if not done:

            maxA, _ = targetPolicyQ.findActionGreedy(targetPolicyQ.getData(newS))

            maxQ = targetValueQ.getData(newS, maxA).get(self.Q_IDX, targetValueQ.defaultVal)

            aData[self.Q_IDX] = (
                (1.0 - alpha) * aData.get(self.Q_IDX, self.defaultVal) +
                alpha * (reward + self.gamma * maxQ)
            )

        else:

            aData[self.Q_IDX] = (
                (1.0 - alpha) * aData.get(self.Q_IDX, self.defaultVal) +
                alpha * reward
            )

        try:
            aData[self.N_IDX] += 1
        except KeyError as e:
            aData[self.N_IDX] = 1

        self.setData(oldS, a, aData)

    def computeAlpha(self, aData):
        """
        Implements a descreasing learning rate scheme of the form:

        alpha(s, a) = max(alphaCutoff, 1 / n(s,a) ** alphaRate)

        where n(s,a) is the number of times action a has been chosen
        when in state s, and alphaRate is an attribute of this class
        set in the constructor.

        If the rate is chosen in ]0.5;1] and alphaCutoff is 0, then
        the basic hypothesis of Q-learning that requires alpha to not
        be integrable but to be square-integrable for the algorithm to
        converge is met.
        """

        n = aData.get(self.N_IDX, 0)

        return max(self.alphaCutoff, 1.0 / (1.0 + n) ** self.alphaRate)
