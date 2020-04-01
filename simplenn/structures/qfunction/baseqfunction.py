import random as rd
import numpy as np

class BaseQFunction:

    N_IDX = 'n'
    Q_IDX = 'q'

    def __init__(self, actions, epsInit, epsFinal, epsRampLen, defaultVal):

        self.actions = list(actions)
        self.epsInit = epsInit
        self.epsFinal = epsFinal
        self.epsRampLen = epsRampLen
        self.defaultVal = defaultVal

        self._isLearning = False

    @property
    def isLearning(self):

        return self._isLearning

    @isLearning.setter
    def isLearning(self, newVal):

        self._isLearning = newVal

    def copy(self):

        raise NotImplementedError("Must be implemented in subclasses!")

    def getData(self, s, a=None):

        raise NotImplementedError("Must be implemented in subclasses!")

    def findAction(self, state):

        if self.isLearning:
            return self.findActionEpsilon(state, None)[0]
        else:
            return self.findActionEpsilon(state, 0.0)[0]

    def findActionEpsilon(self, s, eps):

        sData = self.getData(s)

        if eps is None:
            eps = self.computeEpsilon(sData)

        if rd.uniform(0, 1) < eps:
            return self.findActionRandom(sData)
        else:
            return self.findActionGreedy(sData)

    def findActionRandom(self, sData):

        a = rd.choice(self.actions)

        aData = sData.get(a, {})

        return (a, aData.get(self.Q_IDX, self.defaultVal))

    def findActionGreedy(self, sData):

        bestActions = []
        bestQ = -np.inf

        for a, aData in sData.items():

            q = aData[self.Q_IDX]

            if q > bestQ:
                bestActions = [a]
                bestQ = q
            elif q == bestQ:
                bestActions.append(a)

        if bestActions:
            return (rd.choice(bestActions), bestQ)
        else:
            return self.findActionRandom(sData)

    def computeEpsilon(self, sData):
        """
        Implements an epsilon-greedy policy with varying epsilon.
        """

        n = sum(aData[self.N_IDX] for a, aData in sData.items())

        n_adjusted = np.floor(n / len(self.actions))

        if n_adjusted < self.epsRampLen:

            a = (self.epsFinal - self.epsInit) / self.epsRampLen
            b = self.epsInit

            return a * n_adjusted + b

        else:

            return self.epsFinal
