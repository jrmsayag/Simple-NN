import numpy as np

from .. import qfunction

class BaseSimulation:

    def __init__(self, Q):

        self.Q = Q
        self.qs = []

    def findActionStandard(self):

        return self.Q.findActionEpsilon(self.s_state, 0.0)

    def findActionLearn(self):

        return self.Q.findActionEpsilon(self.s_state, None)

    def learnStep(self):

        raise NotImplementedError("Learn not implemented!")

    def performEpisode(self, env, learn=False, render=False):

        # Initialize environment and state

        self.s_state = env.reset()
        self.s_totalReward = 0

        # Simulating the environment

        while True:

            # Rendering

            if render and not learn:
                env.render()

            # Choosing action

            if learn:
                self.s_action, _ = self.findActionLearn()
            else:
                self.s_action, _ = self.findActionStandard()

            # Performing action

            self.s_nextState, self.s_reward, self.s_done, _ = env.step(self.s_action)
            self.s_totalReward += self.s_reward

            # Updating Q

            if learn:
                self.learnStep()

            # Checking game end

            if self.s_done:
                break
            else:
                self.s_state = self.s_nextState

        # Closing environment

        env.close()

        # Returning score

        return self.s_totalReward

    def learn(self, env, nEpisodes, nTest=0, verboseFreq=0, recordFreq=0, wandb=None):

        for episode in range(nEpisodes + 1):

            self.currentEpisode = episode

            # Performing one episode and updating Q

            self.performEpisode(env, learn=True)

            # Performance evaluation

            if verboseFreq and nTest and episode % verboseFreq == 0:

                self.log(env, nTest, wandb)

            # Recording intermediate Q function

            if (episode == nEpisodes) or (recordFreq and episode % recordFreq == 0):

                self.record()

        return self.Q

    def log(self, env, nTest, wandb):

        testScores = [self.performEpisode(env) for _ in range(nTest)]

        if wandb is None:
            self.logConsole(testScores)
        else:
            self.logWandb(testScores, wandb)

    def logConsole(self, testScores):

        meanTestScore = np.mean(testScores).round(2)
        meanTestScoreError = (np.std(testScores) / np.sqrt(len(testScores))).round(2)

        print(f"Episode {self.currentEpisode}: {meanTestScore} +/- {meanTestScoreError}")

    def logWandb(self, testScores, wandb):

        res = {
            "scores":wandb.Histogram(testScores),
            "score":np.mean(testScores),
            "episode":self.currentEpisode
        }

        finalQs = (
            self.Q.getFinalQs()
            if isinstance(self.Q, qfunction.AggregateQFunction)
            else [self.Q]
        )

        for i in range(len(finalQs)):

            q = finalQs[i]

            res[f"nStates_{i}"] = len(q.data)

            qValues = []
            actionsN = []

            for s, sData in q.data.items():

                for a, aData in sData.items():

                    qValues.append(aData[q.Q_IDX])
                    actionsN.append(aData[q.N_IDX])

            res[f"qValues_{i}"] = wandb.Histogram(qValues, num_bins=50)
            res[f"actionsN_{i}"] = wandb.Histogram(actionsN, num_bins=250)

        wandb.log(res)

    def record(self):

        self.qs.append(self.Q.copy())
