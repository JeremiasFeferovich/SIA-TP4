import numpy as np
import pandas as pd
from letter_printer import get_letters
import itertools
import matplotlib.pyplot as plt
from noise import salt_and_pepper_noise, print_number

class HopfieldNetwork:
    def __init__(self, patterns):
        self.weights = np.zeros((len(patterns[0]), len(patterns[0])))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns[0])
        self.iterations_without_change = 0

    def predict(self, pattern):
        self.iterations_without_change = 0
        while True:
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):
                self.iterations_without_change += 1
                if self.iterations_without_change == 10:
                    return new_pattern
            else:
                self.iterations_without_change = 0
            pattern = new_pattern

def main():
    letters = get_letters()
    flat_letters = {
        k: v.flatten() for k, v in letters.items()
    }

    # all_groups = itertools.combinations(flat_letters.keys(), 4)

    # avg_dot_product = []
    # max_dot_product = []

    # for g in all_groups:
    #     group = np.array([v for k, v in flat_letters.items() if k in g])
    #     orto_matrix = group.dot(group.T)
        
    #     np.fill_diagonal(orto_matrix, 0)
    #     row, col = orto_matrix.shape
    #     avg_dot_product.append((np.abs(orto_matrix).sum() / (orto_matrix.size - row), g))
    #     max_v = np.abs(orto_matrix).max()
    #     max_dot_product.append(((max_v, np.count_nonzero(np.abs(orto_matrix) == max_v) / 2), g))

    # df = pd.DataFrame(sorted(avg_dot_product), columns=['|<,>| medio', 'Grupo'])
    # print(df.head(15).style.format({'|<,>| medio': '{:.2f}'}).hide(axis='index').data)
    # print(df.tail(5).style.format({'|<,>| medio': '{:.2f}'}).hide(axis='index').data)
    group_letters = ['A', 'B', 'C', 'D']
    group_1 = np.array([v for k, v in flat_letters.items() if k in group_letters])
    copied_group_1 = np.copy(group_1)
    noisy_group_1 = np.array([salt_and_pepper_noise(v, intensity=0.1, shape=(5,5)) for v in copied_group_1])
    hopfield_network = HopfieldNetwork(group_1)

    # FIXME check
    for i in range(len(group_1)):
        result = hopfield_network.predict(noisy_group_1[i])

        # Find result in group_1
        found = False
        for j, letter in enumerate(group_1):
            if np.array_equal(result, letter):
                print(f'Should be: {group_letters[i]}. Letter found: {group_letters[j]}')
                found = True
                break
        if not found:
            print(f'Should be: {group_letters[i]}. Letter not found.')
    
    plt.show()

if __name__ == "__main__":
    main()