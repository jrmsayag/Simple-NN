import numpy as np

from ...structures import qfunction

class BaseQLearning:

    def __init__(self, Q):

        self.Q = Q
        self.qs = []

    def learnStep(self, state, nextState, action, reward, done):

        raise NotImplementedError("Learn not implemented!")

    def learn(
        self,
        sim,
        nEpisodes,
        verboseFreq=0,
        recordFreq=0,
        wandb=None):

        oldIsLearning = self.Q.isLearning

        for episode in range(nEpisodes + 1):

            self.currentEpisode = episode

            # Performing one episode and updating Q

            self.Q.isLearning = True
            sim.performEpisode(self.Q, stepCallback=self.learnStep)
            self.Q.isLearning = oldIsLearning

            # Performance evaluation

            if verboseFreq and episode % verboseFreq == 0:

                self.log(sim.performEpisodes(self.Q)[1], wandb)

            # Recording intermediate Q function

            if (episode == nEpisodes) or (recordFreq and episode % recordFreq == 0):

                self.record()

        return self.Q

    def log(self, testScores, wandb):

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
