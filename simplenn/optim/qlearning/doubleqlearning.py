import random as rd

from .. import qlearning
from ...structures import qfunction

class DoubleQLearning(qlearning.BaseSimulation):

    def __init__(self, Q, replayMem):

        self.Qa = Q.copy()
        self.Qb = Q.copy()

        super().__init__(qfunction.AggregateQFunction([self.Qa, self.Qb]))

        self.replayMem = replayMem

    def learnStep(self):

        self.replayMem.addSample((
            self.s_nextState,
            self.s_state,
            self.s_action,
            self.s_reward,
            self.s_done
        ))

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
