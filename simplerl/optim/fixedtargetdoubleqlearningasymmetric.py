from .. import optim

class FixedTargetDoubleQLearningAsymmetric(optim.BaseSimulation):

    def __init__(self, Q, replayMem, targetQUpdateFreq, quantums):

        super().__init__(Q, quantums)

        self.replayMem = replayMem
        self.targetQUpdateFreq = targetQUpdateFreq

        self.learnStepCounter = 0

    def learnStep(self):

        self.replayMem.addSample((
            self.s_nextStateQu,
            self.s_stateQu,
            self.s_action,
            self.s_reward,
            self.s_done
        ))

        if self.replayMem.ready:

            if self.learnStepCounter % self.targetQUpdateFreq == 0:

                self.targetQ = self.Q.copy()

            self.Q.update(
                *self.replayMem.getSample(),
                targetPolicyQ=self.Q,
                targetValueQ=self.targetQ
            )

            self.learnStepCounter += 1
