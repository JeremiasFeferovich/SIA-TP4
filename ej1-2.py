import numpy as np
import pandas as pd
from normalizer import Normalizer
from sklearn.decomposition import PCA

class OjaNeuron:
    def __init__(self, max_iter, dimensions, learning_rate=1e-3):
        self.weights = np.random.rand(dimensions)
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def train(self, data_input):
        i = 0
        while i < self.max_iter:
            for value in data_input:
                activation = sum(value * self.weights)
                delta = self.learning_rate * (activation * value - activation**2 * self.weights)
                self.weights = self.weights + delta
            i += 1

        return self.weights
    
def main():
    data = pd.read_csv('europe.csv')
    data_input = data.values

    # Remove first column
    data_input = np.delete(data_input, 0, axis=1)

    normalizers = []

    for i in range(len(data_input[0])):
        normalizers.append(Normalizer(data_input[:, i]))
        data_input[:, i] = normalizers[i].normalize(data_input[:, i])

    standarized_data = data_input

    oja_neuron = OjaNeuron(max_iter=1000, dimensions=len(standarized_data[0]), learning_rate=1e-3)
    weights = oja_neuron.train(standarized_data)

    pca = PCA(n_components=2)
    pca.fit(standarized_data)
    

    print(f'Difference: {pca.components_[0]+weights}')
    print(f'Mean difference: {np.mean(pca.components_[0]+weights)}')

    print(weights)

    
if __name__ == "__main__":
    main()