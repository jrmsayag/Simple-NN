import random as rd

class ReplayMemory:

    MODE_CYCLING = 0
    MODE_RANDOM = 1

    def __init__(self, maxSize, minSize, mode):

        self.maxSize = maxSize
        self.minSize = minSize
        self.mode = mode

        self.counter = 0
        self.samples = []

    def addSample(self, sample):

        if self.mode == self.MODE_CYCLING:

            self.addSampleCycling(sample)

        elif self.mode == self.MODE_RANDOM:

            self.addSampleRandom(sample)

        else:

            raise NotImplementedError(f"Unknown mode {self.mode}")

    def addSampleCycling(self, sample):

        if len(self.samples) < self.maxSize:

            self.samples.append(sample)

        else:

            self.samples[self.counter % self.maxSize] = sample

        self.counter += 1

    def addSampleRandom(self, sample):

        if len(self.samples) < self.maxSize:

            self.samples.append(sample)

        else:

            self.samples[rd.randint(0, len(replayMem)-1)] = sample

        self.counter += 1

    def getSample(self):

        if self.ready:

            return rd.choice(self.samples)

        else:

            raise RuntimeError("Not enough history!")

    @property
    def ready(self):

        return len(self.samples) >= self.minSize
