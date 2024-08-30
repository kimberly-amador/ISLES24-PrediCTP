import numpy as np


class StepDecay:
    """
    Reference code: https://pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
    """

    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # Store the base initial learning rate, drop factor, and epochs
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # Compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        return float(alpha)
