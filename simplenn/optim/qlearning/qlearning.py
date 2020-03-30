from .. import qlearning

class QLearning(qlearning.BaseSimulation):

    def __init__(self, Q, replayMem):

        super().__init__(Q)

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

            self.Q.update(*self.replayMem.getSample())
