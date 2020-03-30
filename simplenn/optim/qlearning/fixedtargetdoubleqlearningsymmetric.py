import random as rd

from .. import qlearning
from ...structures import qfunction

class FixedTargetDoubleQLearningSymmetric(qlearning.BaseSimulation):

    def __init__(self, Q, replayMem, targetQUpdateFreq):

        self.Qa = Q.copy()
        self.Qb = Q.copy()

        super().__init__(qfunction.AggregateQFunction([self.Qa, self.Qb]))

        self.replayMem = replayMem
        self.targetQUpdateFreq = targetQUpdateFreq

        self.learnStepCounter = 0

    def learnStep(self):

        self.replayMem.addSample((
            self.s_nextState,
            self.s_state,
            self.s_action,
            self.s_reward,
            self.s_done
        ))

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
