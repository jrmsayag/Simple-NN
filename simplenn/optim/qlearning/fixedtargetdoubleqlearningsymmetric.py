import random as rd

from .. import qlearning
from ...structures import qfunction

class FixedTargetDoubleQLearningSymmetric(qlearning.BaseQLearning):

    def __init__(self, Q, replayMem, targetQUpdateFreq):

        self.Qa = Q.copy()
        self.Qb = Q.copy()

        super().__init__(qfunction.AggregateQFunction([self.Qa, self.Qb]))

        self.replayMem = replayMem
        self.targetQUpdateFreq = targetQUpdateFreq

        self.learnStepCounter = 0

    def learnStep(self, state, nextState, action, reward, done):

        self.replayMem.addSample((nextState, state, action, reward, done))

        if self.replayMem.ready:

            if self.learnStepCounter % self.targetQUpdateFreq == 0:

                self.targetQa = self.Qa.copy()
                self.targetQb = self.Qb.copy()

            if rd.randint(0, 1):

                self.Qa.update(
                    *self.replayMem.getSample(),
                    targetPolicyQ=self.Qa,
                    targetValueQ=self.targetQb
                )

            else:

                self.Qb.update(
                    *self.replayMem.getSample(),
                    targetPolicyQ=self.Qb,
                    targetValueQ=self.targetQa
                )

            self.learnStepCounter += 1
