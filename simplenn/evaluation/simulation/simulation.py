import numpy as np

class Simulation:

    def __init__(self, env, nEpisodes, actionSelectionMode="argmax", isScore=True):

        self.env = env
        self.nEpisodes = nEpisodes
        self.actionSelectionMode = actionSelectionMode
        self.isScore = isScore

    def apply(self, n):

        episodesResult = [self.performEpisode(n) for _ in range(self.nEpisodes)]

        return (np.mean(episodesResult), episodesResult)

    def performEpisode(self, n, render=False):

        # Initialize environment and state

        self.s_state = self.env.reset()
        self.s_totalReward = 0

        # Simulating the environment

        while True:

            # Rendering

            if render:
                self.env.render()

            # Choosing action

            output = n.forward(self.s_state[..., np.newaxis])

            if self.actionSelectionMode == "argmax":
                self.s_action = np.argmax(output)
            else:
                raise ValueError(f"Uknown action selection mode {self.actionSelectionMode}!")

            # Performing action

            self.s_nextState, self.s_reward, self.s_done, _ = self.env.step(self.s_action)
            self.s_totalReward += self.s_reward

            # Checking game end

            if self.s_done:
                break
            else:
                self.s_state = self.s_nextState

        # Closing environment

        self.env.close()

        # Returning score

        return self.s_totalReward
