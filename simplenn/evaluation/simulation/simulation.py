import numpy as np

from ... import evaluation

class Simulation(evaluation.EvalFunc):

    def __init__(self, env, nEpisodes=1, isScore=True):

        super().__init__()

        self.env = env
        self.nEpisodes = nEpisodes
        self.isScore = isScore

    @property
    def isSimulation(self):

        return True

    @property
    def isLoss(self):

        return False

    def performEpisodes(self, struct, nEpisodes=None):

        if not nEpisodes:
            nEpisodes = self.nEpisodes

        episodesResult = [self.performEpisode(struct) for _ in range(nEpisodes)]

        return (np.mean(episodesResult), episodesResult)

    def performEpisode(self, struct, render=False, learnCallback=None):

        # Initialize environment and state

        self.s_state = self.env.reset()
        self.s_totalReward = 0

        # Simulating the environment

        while True:

            # Rendering

            if render:
                self.env.render()

            # Choosing action

            self.s_action = struct.findAction(self.s_state)

            # Performing action

            self.s_nextState, self.s_reward, self.s_done, _ = self.env.step(self.s_action)
            self.s_totalReward += self.s_reward

            # Performing learning step

            if learnCallback is not None:

                learnCallback(
                    self.s_state,
                    self.s_nextState,
                    self.s_action,
                    self.s_reward,
                    self.s_done)

            # Checking game end

            if self.s_done:
                break
            else:
                self.s_state = self.s_nextState

        # Closing environment

        self.env.close()

        # Returning score

        return self.s_totalReward
