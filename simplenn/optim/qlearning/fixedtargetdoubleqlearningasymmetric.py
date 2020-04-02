from .. import qlearning

class FixedTargetDoubleQLearningAsymmetric(qlearning.BaseQLearning):

    def __init__(self, Q, replayMem, targetQUpdateFreq):

        super().__init__(Q)

        self.replayMem = replayMem
        self.targetQUpdateFreq = targetQUpdateFreq

        self.learnStepCounter = 0

    def learnStep(self, state, nextState, action, reward, done):

        self.replayMem.addSample((nextState, state, action, reward, done))

        if self.replayMem.ready:

            if self.learnStepCounter % self.targetQUpdateFreq == 0:

                self.targetQ = self.Q.copy()

            self.Q.update(
                *self.replayMem.getSample(),
                targetPolicyQ=self.Q,
                targetValueQ=self.targetQ
            )

            self.learnStepCounter += 1
