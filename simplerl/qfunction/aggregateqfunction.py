import numpy as np

from .. import qfunction

class AggregateQFunction(qfunction.BaseQFunction):

    def __init__(self, Qs):

        super().__init__(
            Qs[0].actions,
            Qs[0].epsInit,
            Qs[0].epsFinal,
            Qs[0].epsRampLen,
            np.mean([q.defaultVal for q in Qs])
        )

        self.Qs = Qs

    def copy(self):

        return AggregateQFunction([q.copy() for q in self.Qs])

    def getData(self, s, a=None):

        if a is None:
            return self.getStateData(s)
        else:
            return self.getActionData(s, a)

    def getStateData(self, s):

        sData = {}

        for a in self.actions:

            aData = self.getActionData(s, a)

            if aData:
                sData[a] = aData

        return sData

    def getActionData(self, s, a):

        nKnownTotal = 0
        nVisitsTotal = 0
        qValueTotal = 0.0

        for q in self.Qs:

            aData = q.getData(s, a)

            if aData:

                nVisits = aData[q.N_IDX]
                qValue = aData[q.Q_IDX]

                nKnownTotal += 1
                nVisitsTotal += nVisits
                qValueTotal += qValue

        if nKnownTotal:
            return {self.N_IDX: nVisitsTotal, self.Q_IDX: qValueTotal / nKnownTotal}
        else:
            return {}

    def getFinalQs(self):

        finalQs = []
        currentQs = self.Qs

        while currentQs:

            nextQs = []

            for q in currentQs:

                if isinstance(q, AggregateQFunction):
                    nextQs.extend(q.Qs)
                else:
                    finalQs.append(q)

            currentQs = nextQs

        return finalQs
