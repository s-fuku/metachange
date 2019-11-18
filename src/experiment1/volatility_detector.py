import numpy as np

class VolatilityDetector:
    #def __init__(self, b=32, r=32, beta=0.1, seed=123):
    def __init__(self, b=32, r=32, seed=123):
        np.random.seed(seed)
        # initialize buffer and reservoir
        self.buffer = []
        self.reservoir = []
        #self.reservoir = np.repeat(np.nan, r)

        self.b = b
        self.r = r
        #self.beta = beta

    def detect(self, x=None):
        if x is not None:
            j = self.addToBuffer(x)
            if j is not None:
                self.addToReservoir(j)
        if (len(self.buffer) == self.b) & (len(self.reservoir) == self.r):
        #if np.all(~np.isnan(self.reservoir)):
            #relative_variance = np.std(self.buffer) / np.std(self.reservoir)
            var_buffer = np.var(self.buffer)
            var_reservoir = np.var(self.reservoir)
            #if np.abs(var_reservoir) < 1e-6:
            #    var_reservoir = 1e-6
            if (np.abs(var_reservoir) < 1e-10) & (np.abs(var_buffer) < 1e-10):
                return 1.0
            #relative_variance = np.var(self.buffer) / np.var(self.reservoir)
            relative_variance = var_buffer / var_reservoir
            #return np.logical_or(relative_variance <= 1.0 - self.beta, relative_variance >= 1.0 + self.beta), relative_variance
            return relative_variance

        #return False
        #return None
        #return False, np.nan
        return np.nan

    def addToBuffer(self, k):
        self.buffer.append(k)
        if len(self.buffer) == self.b + 1:
            return self.buffer.pop(0)
        return None

    def addToReservoir(self, k):
        if len(self.reservoir) < self.r:
            self.reservoir.append(k)
        else:
            rPos = np.random.randint(0, self.r)
            self.reservoir[rPos] = k
        #rPos = np.random.randint(0, self.r)
        #self.reservoir[rPos] = k
