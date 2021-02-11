import numpy as np

class DoubleLogistic(object):
    def __init__(self):
        self.L = 0.1845
        self.c = 1.0

    def apply(self, x):
        out = np.zeros_like(x)
        x_shifted = x - self.L / 2.0
        # Left part
        mask = x_shifted <= 0.0
        out[mask] = self.L / (1.0 + np.exp(-4.0 * self.c / self.L * x_shifted[mask]))
        # Right part
        out[~mask] = (2.0 - self.L) / (1.0 + np.exp(-4.0 * self.c / (2.0 - self.L) * x_shifted[~mask])) + self.L - 1
        return out
