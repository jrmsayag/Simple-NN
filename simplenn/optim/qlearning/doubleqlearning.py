import random as rd

from .. import qlearning
from ...structures import qfunction

class DoubleQLearning(qlearning.BaseQLearning):

    def __init__(self, Q, replayMem):

        self.Qa = Q.copy()
        self.Qb = Q.copy()

        super().__init__(qfunction.AggregateQFunction([self.Qa, self.Qb]))

        self.replayMem = replayMem

    def learnStep(self, state, nextState, action, reward, done):

        self.replayMem.addSample((nextState, state, action, reward, done))

        if self.replayMem.ready:

            if rd.randint(0, 1):

                self.Qa.update(
                    *self.replayMem.getSample(),
                    targetPolicyQ=self.Qa,
                    targetValueQ=self.Qb
                )

            else:

                self.Qb.update(
                    *self.replayMem.getSample(),
                    targetPolicyQ=self.Qb,
                    targetValueQ=self.Qa
                )
