import numpy as np

class Genetic:

    def __init__(self):

        pass

    def run(
        self,
        net,
        xs,
        ys,
        popSize,
        tournamentSize,
        nElitism,
        mutationProba,
        mutationScale,
        mutationRelative,
        generations,
        verboseFreq=0,
        recordFreq=0):

        self.networks = []

        population = [net.copy(True) for _ in range(popSize)]

        for gen in range(generations):

            # Applying loss/fitness function.

            penalties = [(n, n.loss.apply(n.forward(xs), ys).sum()) for n in population]
            penalties.sort(key = lambda penalty: penalty[1])

            # Showing the result.

            if verboseFreq and gen % verboseFreq == 0:

                best = penalties[0][0]
                error = best.loss.apply(best.forward(xs), ys).sum()
                print(f"Generation {gen}: {error}")

            if recordFreq and gen % recordFreq == 0:

                self.record(penalties[0][0], xs, ys)

            # Creating the next generation's population...

            newPop = []

            # ...by crossing over best individuals

            while len(newPop) < popSize - nElitism:

                # Performing tournaments to choose the parents.

                candidatesP1 = [penalties[np.random.randint(popSize)] for _ in range(tournamentSize)]
                candidatesP2 = [penalties[np.random.randint(popSize)] for _ in range(tournamentSize)]

                p1 = candidatesP1[0]
                for c in candidatesP1[1:]:
                    if c[1] < p1[1]:
                        p1 = c

                p2 = candidatesP2[0]
                for c in candidatesP2[1:]:
                    if c[1] < p2[1]:
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

            population = newPop

        population.sort(key = lambda net: net.loss.apply(net.forward(xs), ys).sum())

        self.record(population[0], xs, ys)

        return population[0]

    def record(self, net, xs, ys):

        netCopy = net.copy()

        netCopy.forward(xs)

        self.networks.append(netCopy)
