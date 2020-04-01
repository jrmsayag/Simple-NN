import numpy as np

class Genetic:

    def __init__(self):

        self.population = []

    def run(
        self,
        net,
        evalFunc,
        popSize,
        tournamentSize,
        nElitism,
        mutationProba,
        mutationScale,
        mutationRelative,
        nGenerationsBatch,
        verboseFreq=0,
        recordFreq=0,
        wandb=None):

        if not self.population:

            self.currentGen = 0
            self.currentEpisode = 0
            self.networks = []
            self.population = [net.copy(True) for _ in range(popSize)]

        for _ in range(nGenerationsBatch):

            # Applying loss/fitness function.

            penalties = [(n, *self.computeLoss(n, evalFunc, True)) for n in self.population]
            penalties.sort(key = lambda penalty: self.sortKeyFor(penalty, evalFunc))

            # Showing the result.

            if verboseFreq and self.currentGen % verboseFreq == 0:

                self.log(penalties, wandb)

            if recordFreq and self.currentGen % recordFreq == 0:

                self.record(penalties[0][0], evalFunc)

            # Creating the next generation's population...

            newPop = []

            # ...by crossing over best individuals

            while len(newPop) < popSize - nElitism:

                # Performing tournaments to choose the parents.

                candidatesP1 = [penalties[np.random.randint(len(penalties))] for _ in range(tournamentSize)]
                candidatesP2 = [penalties[np.random.randint(len(penalties))] for _ in range(tournamentSize)]

                p1 = candidatesP1[0]
                for c in candidatesP1[1:]:
                    if self.sortKeyFor(c, evalFunc) < self.sortKeyFor(p1, evalFunc):
                        p1 = c

                p2 = candidatesP2[0]
                for c in candidatesP2[1:]:
                    if self.sortKeyFor(c, evalFunc) < self.sortKeyFor(p2, evalFunc):
                        p2 = c

                # Crossing over the selected parents.

                newNet = p1[0].crossover(p2[0])
                newNet.mutate(p=mutationProba, scale=mutationScale, relative=mutationRelative)

                newPop.append(newNet)

            # ...and by keeping the nElitism best individuals.

            for penalty in penalties:

                if len(newPop) >= popSize:
                    break
                else:
                    newPop.append(penalty[0].copy())

            self.population = newPop
            self.currentGen += 1

        penalties = [(n, *self.computeLoss(n, evalFunc, False)) for n in self.population]
        penalties.sort(key = lambda penalty: self.sortKeyFor(penalty, evalFunc))

        self.record(penalties[0][0], evalFunc)

        return penalties[0][0]

    def computeLoss(self, n, evalFunc, countEpisodes):

        if evalFunc.isSimulation:

            if countEpisodes:
                self.currentEpisode += evalFunc.nEpisodes

            return evalFunc.performEpisodes(n)

        else:

            if countEpisodes:
                self.currentEpisode += evalFunc.ys.shape[-1]

            res = evalFunc.apply(n.forward(evalFunc.xs), evalFunc.ys)

            return (res.sum(), res.sum(axis=0))

    def sortKeyFor(self, penalty, evalFunc):

        if evalFunc.isSimulation:

            if evalFunc.isScore:

                return -penalty[1]

            else:

                return penalty[1]

        else:

            return penalty[1]

    def log(self, penalties, wandb):

        if wandb is None:
            self.logConsole(penalties)
        else:
            self.logWandb(penalties, wandb)

    def logConsole(self, penalties):

        print(f"Generation {self.currentGen} (ep. {self.currentEpisode}): {penalties[0][1]}")

    def logWandb(self, penalties, wandb):

        res = {
            "score":penalties[0][1],
            "scores":wandb.Histogram(penalties[0][2]),
            "populationScores":wandb.Histogram([p[1] for p in penalties]),
            "episode":self.currentEpisode,
            "generation":self.currentGen
        }

        wandb.log(res)

    def record(self, net, evalFunc):

        netCopy = net.copy()

        if evalFunc.isLoss:
            netCopy.forward(evalFunc.xs)

        self.networks.append(netCopy)
