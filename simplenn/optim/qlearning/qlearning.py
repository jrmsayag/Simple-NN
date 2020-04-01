from .. import qlearning

class QLearning(qlearning.BaseSimulation):

    def __init__(self, Q, replayMem):

        super().__init__(Q)

        self.replayMem = replayMem

    def learnStep(self, state, nextState, action, reward, done):

        self.replayMem.addSample((nextState, state, action, reward, done))

        if self.replayMem.ready:

            self.Q.update(*self.replayMem.getSample())
