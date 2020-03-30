from .. import qlearning

class FixedTargetQLearning(qlearning.BaseSimulation):

    def __init__(self, Q, replayMem, targetQUpdateFreq):

        super().__init__(Q)

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

                self.targetQ = self.Q.copy()

            self.Q.update(
                *self.replayMem.getSample(),
                targetPolicyQ=self.targetQ,
                targetValueQ=self.targetQ
            )

            self.learnStepCounter += 1
