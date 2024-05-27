import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from normalizer import Normalizer
import numpy as np
import math
import sys
from sklearn.decomposition import PCA


class KohonenNetwork:
    def __init__(self, input_size, grid_size, learning_rate=1, radius=1.0, initial_weights=None):
        self.input_size = input_size
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(grid_size**2, input_size).reshape(grid_size, grid_size, input_size) if initial_weights is None else initial_weights
        self.radius = radius
        self.initial_radius = radius
        self.radius_time_constant = 0

    def train(self, data, epochs):
        self.radius_time_constant = (epochs / 1.2) / np.log(self.initial_radius)
        udm_history = []
        quantization_errors = []
        for epoch in range(1, epochs + 1):
            input_value = data[np.random.randint(0, len(data))]
            winner = self.get_winner(input_value)[0]
            self.update_weights(input_value, winner, epoch)

            udm_history.append(self.get_unified_distance_matrix()[1])
            quantization_errors.append(self.compute_quantization_error(data))

            self.radius = self.radius_decay(epoch, epochs)
        
        return udm_history, quantization_errors

    def compute_quantization_error(self, data):
        error = 0
        for input_value in data:
            winner, similarity = self.get_winner(input_value)
            error += similarity
        return error / len(data)
    
    def get_quantization_error_matrix(self, data):
        error_matrix = np.zeros((self.grid_size, self.grid_size))
        error_matrix_count = np.zeros((self.grid_size, self.grid_size))
        for input_value in data:
            winner, similarity = self.get_winner(input_value)
            error_matrix[winner] += similarity
            error_matrix_count[winner] += 1
        return error_matrix / error_matrix_count   

    def get_winner(self, input_value):
        min_similarity = sys.maxsize
        winner = None
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                similarity = self.get_similarity(input_value, self.weights[i][j])
                if similarity < min_similarity:
                    min_similarity = similarity
                    winner = i, j
        return winner, min_similarity

    def get_neighbours(self, winner, radius=None):
        if radius is None:
            radius = self.radius
        neighbours = []
        for i in range(max(0, round(winner[0]-self.radius)), min(self.grid_size, round(winner[0]+self.radius+1))):
            for j in range(max(0, round(winner[1]-self.radius)), min(self.grid_size, round(winner[1]+self.radius+1))):
                distance = math.sqrt((winner[0]-i)**2 + (winner[1]-j)**2)
                if distance <= self.radius:
                    neighbours.append((i, j))
        return neighbours

    def get_similarity(self, x, w):   
        return np.linalg.norm(x - w)

    def radius_decay(self, current_epoch, epochs):
        new_radius = self.initial_radius * math.exp(-current_epoch / self.radius_time_constant)
        # new_radius = (self.initial_radius / (current_epoch +1)) * epochs / 3
        # new_radius = self.initial_radius
        if current_epoch % 1000 == 0:
            print(f'New radius: {new_radius} at epoch {current_epoch} of {epochs}')
        return new_radius if new_radius > 1 else 1

    def update_weights(self, input_data, winner, epoch):
        neighbours = self.get_neighbours(winner)
        for neighbour in neighbours:
            learning_rate = self.learning_rate / epoch
            delta = (input_data - self.weights[neighbour])*learning_rate
            self.weights[neighbour] += delta

    def visualize(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(f'{self.weights[i][j]}', end=" ")
            print("\n")
            
    def get_count_per_neuron(self, data):
        count_per_neuron = np.zeros((self.grid_size, self.grid_size))
        for input_value in data:
            winner = self.get_winner(input_value)[0]
            count_per_neuron[winner] += 1
        return count_per_neuron

    def get_countries_per_neuron(self, data, countries):
        countries_per_neuron = np.zeros((self.grid_size, self.grid_size), dtype=object)
        similarities = [[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for i, input_value in enumerate(data):
            winner, similarity = self.get_winner(input_value)
            similarities[winner[0]][winner[1]].append(similarity)
            if countries_per_neuron[winner] == 0:
                countries_per_neuron[winner] = countries[i]
            else:
                countries_per_neuron[winner] = countries_per_neuron[winner] + ', ' + countries[i]

        #  mean of similarities
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if similarities[i][j] != 0:
                    similarities[i][j] = np.mean(similarities[i][j])
                else:
                    similarities[i][j] = 0
        return countries_per_neuron, similarities
    
    def get_unified_distance_matrix(self):
        distance_matrix = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbours = self.get_neighbours((i, j), 1)
                neighbours.remove((i, j))
                distance_acum = 0
                for neighbour in neighbours:
                    distance_acum += self.get_similarity(self.weights[i][j], self.weights[neighbour[0]][neighbour[1]])
                distance_acum = distance_acum / len(neighbours)
                distance_matrix[i][j] = distance_acum

        return distance_matrix, sum(sum(distance_matrix)) / (self.grid_size**2)

    def get_data_per_neuron(self, data):
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

    grid_size = 4
    extended_data = np.copy(data_input)
    np.random.shuffle(extended_data)

    initial_weights = extended_data[:grid_size*grid_size].reshape(grid_size, grid_size, len(extended_data[0]))
    network = KohonenNetwork(input_size=len(data_input[0]), grid_size=grid_size, learning_rate=1, radius=3, initial_weights=initial_weights)

    udm_history, quantization_errors = network.train(data_input, 10000)

    plt.figure()
    plt.plot(quantization_errors, label='Quantization Error')
    plt.title('Quantization  Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.savefig('results/quantization_error.png')
    # plt.ylim(1.3, 1.8)


    plt.figure()
    plt.plot(udm_history, label='Unified Distance')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.legend()
    plt.ylim(0, 1)
    plt.title('Unified Distance Matrix')
    plt.savefig('results/udm.png')


    plt.figure()
    unified_distance_matrix, udm_mean = network.get_unified_distance_matrix()
    sns.heatmap(unified_distance_matrix, annot=True, fmt='0.2f', cmap='Greys')
    plt.title('Distancia unificada')
    plt.xlabel('Neurona')
    plt.ylabel('Neurona')
    plt.savefig('results/udm_heatmap.png')


    count_per_neuron = network.get_count_per_neuron(data_input)
    data_per_neuron = network.get_data_per_neuron(data_input)

    countries_per_neuron, quantization_errors_matrix = network.get_countries_per_neuron(data_input, data['Country'].values)

    flattened_data = []
    for i in range(data_per_neuron.shape[0]):
        for j in range(data_per_neuron.shape[1]):
            denormalized_data = [normalizers[k].denormalize(data_per_neuron[i][j][k]) for k in range(len(data_per_neuron[i][j]))]
            # denormalized_data = data_per_neuron[i][j]
            flattened_data.append([i, j, *denormalized_data])
    # 
    df = pd.DataFrame(flattened_data, columns=['Row', 'Column', 'Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment'])

    matrixes = {
        'Area': 'Area',
        'GDP': 'GDP',
        'Inflation': 'Inflation',
        'Life.expect': 'Life.expect',
        'Military': 'Military',
        'Pop.growth': 'Pop.growth',
        'Unemployment': 'Unemployment'
    }

    for key in matrixes.keys():
        matrix = np.zeros((network.grid_size, network.grid_size))
        for i in range(data_per_neuron.shape[0]):
            for j in range(data_per_neuron.shape[1]):
                matrix[i][j] = df[(df['Row'] == i) & (df['Column'] == j)][key].values[0]
        # 
        matrixes[key] = matrix

    for j in range(countries_per_neuron.shape[1]):
        for i in range(countries_per_neuron.shape[0]):
            if countries_per_neuron[i][j] != 0:
                countries_per_neuron[i][j] = countries_per_neuron[i][j].replace(',', '\n')
            else:
                countries_per_neuron[i][j] = ''

    # Create a heatmap
    plt.figure()
    sns.heatmap(count_per_neuron, annot=countries_per_neuron, fmt="", cmap='coolwarm')
    plt.title('Paises por neurona')
    plt.xlabel('Neurona')
    plt.ylabel('Neurona')
    plt.savefig('results/countries_per_neuron.png')

    plt.figure()
    sns.heatmap(quantization_errors_matrix, annot=True, fmt='0.2f', cmap='Greys')
    plt.title('Error de cuantizacion')
    plt.xlabel('Neurona')
    plt.ylabel('Neurona')
    plt.savefig('results/quantization_error_heatmap.png')

    plt.figure()
    sns.heatmap(count_per_neuron, annot=True, cmap='coolwarm', fmt='0.0f')
    plt.title('Cantidad de paises por neurona')
    plt.xlabel('Neurona')
    plt.ylabel('Neurona')
    plt.savefig('results/count_per_neuron.png')

    for key in matrixes.keys():
        plt.figure()
        sns.heatmap(matrixes[key], annot=True, fmt='0.2f', cmap='coolwarm')
        plt.title(f'{key} por neurona')
        plt.xlabel('Neurona')
        plt.ylabel('Neurona')
        plt.savefig(f'results/{key}_heatmap.png')

    plt.show()
    
    
if __name__ == "__main__":
    main()