import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

class KohonenNetwork:
    def __init__(self, input_size, grid_size, learning_rate=1, radius=1.0, initial_weights=None):
        self.input_size = input_size
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(grid_size**2, input_size).reshape(grid_size, grid_size, input_size) if initial_weights is None else initial_weights
        self.radius = radius

    def train(self, data, epochs):
        for epoch in range(1, epochs+1):
            for input_value in data:
                winner = self.get_winner(input_value)
                self.update_weights(input_value, winner, epoch)

            self.radius = self.radius_decay(epoch, epochs)

    def get_winner(self, input_value):
        min_similarity = sys.maxsize
        winner = None
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                similarity = self.get_similarity(self.weights[i][j], input_value)
                if similarity < min_similarity:
                    min_similarity = similarity
                    winner = i, j
        return winner

    def get_neighbours(self, winner):
        neighbours = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                distance = math.sqrt((winner[0]-i)**2 + (winner[1]-j)**2)
                if distance <= self.radius:
                    neighbours.append((i, j))
        return neighbours

    def get_similarity(self, x, w):   
        x = normalize(x)
        w = normalize(w)
        return np.linalg.norm(x - w)

    def radius_decay(self, current_epoch, epochs):
        return self.radius * math.exp(-current_epoch * math.log(self.radius) / epochs)


    def update_weights(self, input_data, winner, epoch):
        neighbours = self.get_neighbours(winner)
        for neighbour in neighbours:
            delta = np.array(np.subtract(input_data, self.weights[neighbour]))* (self.learning_rate / epoch)
            self.weights[neighbour] = np.add(self.weights[neighbour], delta)
    

    def visualize(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(f'{self.weights[i][j]}', end=" ")
            print("\n")
            
    def get_count_per_neuron(self, data):
        count_per_neuron = np.zeros((self.grid_size, self.grid_size))
        for input_value in data:
            winner = self.get_winner(input_value)
            count_per_neuron[winner] += 1
        return count_per_neuron

    def get_countries_per_neuron(self, data, countries):
        countries_per_neuron = np.zeros((self.grid_size, self.grid_size), dtype=object)
        for i, input_value in enumerate(data):
            winner = self.get_winner(input_value)
            if countries_per_neuron[winner] == 0:
                countries_per_neuron[winner] = countries[i]
            else:
                countries_per_neuron[winner] = countries_per_neuron[winner] + ', ' + countries[i]
        return countries_per_neuron
    
    def get_data_per_neuron(self, data):
        return self.weights

def normalize(data):
    return (data - np.mean(data)) / np.std(data)

def main():
    data = pd.read_csv('europe.csv')
    data_input = data.values

    # Remove first column

    data_input = np.delete(data_input, 0, axis=1)

    # for i in range(len(data_input)):
    #    data_input[i] = normalize(data_input[i])

    network = KohonenNetwork(input_size=len(data_input[0]), grid_size=4, learning_rate=1, radius=math.sqrt(32), initial_weights=data_input[:4*4].reshape(4, 4, len(data_input[0])))

    network.train(data_input, 1000)

    count_per_neuron = network.get_count_per_neuron(data_input)
    data_per_neuron = network.get_data_per_neuron(data_input)

    countries_per_neuron = network.get_countries_per_neuron(data_input, data['Country'].values)

    flattened_data = []
    for i in range(data_per_neuron.shape[0]):
        for j in range(data_per_neuron.shape[1]):
            flattened_data.append([i, j, *data_per_neuron[i][j]])
    
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
        
        matrixes[key] = matrix

    for j in range(countries_per_neuron.shape[1]):
        for i in range(countries_per_neuron.shape[0]):
            countries_per_neuron[i][j] = countries_per_neuron[i][j].replace(',', '\n')

    # Create a heatmap
    sns.heatmap(count_per_neuron, annot=countries_per_neuron, fmt="", cmap='coolwarm')
    plt.title('Paises por neurona')
    plt.xlabel('Neurona')
    plt.ylabel('Neurona')

    plt.figure()


    sns.heatmap(count_per_neuron, annot=True, cmap='coolwarm', fmt='0.0f')
    plt.title('Cantidad de paises por neurona')
    plt.xlabel('Neurona')
    plt.ylabel('Neurona')

    for key in matrixes.keys():
        plt.figure()
        sns.heatmap(matrixes[key], annot=True, fmt='0.2f', cmap='coolwarm')
        plt.title(f'{key} por neurona')
        plt.xlabel('Neurona')
        plt.ylabel('Neurona')

    plt.show()
    
    
if __name__ == "__main__":
    main()