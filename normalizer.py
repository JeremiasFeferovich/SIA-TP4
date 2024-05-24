import numpy as np

class Normalizer:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def normalize(self, data):
        return (data - self.mean) / self.std
    
    def denormalize(self, data):
        return data * self.std + self.mean