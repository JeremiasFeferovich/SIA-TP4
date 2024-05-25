import numpy as np
import pandas as pd
from letter_printer import get_letters
import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from noise import salt_and_pepper_noise, gaussian_noise, print_letter, print_group

class HopfieldNetwork:
    def __init__(self, patterns):
        self.weights = np.zeros((len(patterns[0]), len(patterns[0])))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns[0])
        self.iterations_without_change = 0

    def predict(self, pattern, name='Pattern'):
        self.iterations_without_change = 0
        iterations = 0
        # print_letter(f'Predicted {iterations}: {name} o', pattern)

        pattern_history = [pattern]

        while True:
            new_pattern = np.sign(np.dot(self.weights, pattern))

            if np.array_equal(new_pattern, pattern):
                self.iterations_without_change += 1
                if self.iterations_without_change == 3:
                    return new_pattern, pattern_history
            else:
                self.iterations_without_change = 0
                # print_letter(f'Predicted {iterations}: {name}', new_pattern)
                pattern_history.append(new_pattern)

            pattern = new_pattern
            iterations += 1
    

def main():
    letters = get_letters()
    flat_letters = {
        k: v.flatten() for k, v in letters.items()
    }

    all_groups = itertools.combinations(flat_letters.keys(), 4)

    avg_dot_product = []
    max_dot_product = []

    for g in all_groups:
        group = np.array([v for k, v in flat_letters.items() if k in g])
        orto_matrix = group.dot(group.T)
        
        np.fill_diagonal(orto_matrix, 0)
        row, col = orto_matrix.shape
        avg_dot_product.append((np.abs(orto_matrix).sum() / (orto_matrix.size - row), g))
        max_v = np.abs(orto_matrix).max()
        max_dot_product.append(((max_v, np.count_nonzero(np.abs(orto_matrix) == max_v) / 2), g))

    df = pd.DataFrame(sorted(avg_dot_product), columns=['|<,>| medio', 'Grupo'])
    # print(df.head(15).style.format({'|<,>| medio': '{:.2f}'}).hide(axis='index').data)
    # print(df.tail(5).style.format({'|<,>| medio': '{:.2f}'}).hide(axis='index').data)
    chosen_group_letters = df.iloc[0]['Grupo']
    chosen_group = np.array([v for k, v in flat_letters.items() if k in chosen_group_letters])
    copied_group = np.copy(chosen_group)
    # noisy_group_1 = np.array([salt_and_pepper_noise(v, intensity=0.3, shape=(5,5)) for v in copied_group])
    noisy_group_1 = np.array([gaussian_noise(v, intensity=1, shape=(5,5)) for v in copied_group])
    hopfield_network = HopfieldNetwork(chosen_group)

    chosen_correctly = []

    for i in range(len(chosen_group)):
        result, pattern_history = hopfield_network.predict(noisy_group_1[i], name=f'Noisy {chosen_group_letters[i]}')

        fps = 2  # frames per second
        delay_seconds = 2
        extra_frames = fps * delay_seconds
        pattern_history = pattern_history + [pattern_history[-1]] * extra_frames

        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            ax.clear()
            ax.imshow(pattern_history[frame].reshape(5, 5), cmap='gray_r')
            ax.axis('off')
            return ax,

        ani = FuncAnimation(fig, update, frames=len(pattern_history), init_func=init, blit=False,  interval=500, repeat_delay=1000)
        ani.save(f'results/result_animation_{i}.gif', writer='pillow', fps=fps)

        # Find result in group_1
        found = False
        for j, letter in enumerate(chosen_group):
            if np.array_equal(result, letter):
                print(f'Should be: {chosen_group_letters[i]}. Letter found: {chosen_group_letters[j]}')
                found = True
                chosen_correctly.append(chosen_group_letters[j])
                break
        if not found:
            print(f'Should be: {chosen_group_letters[i]}. Letter not found.')
            chosen_correctly.append(False)

   


    print_group(chosen_group, 'Original group')
    print_group(noisy_group_1, 'Noisy group', chosen_correctly)

    plt.show()

if __name__ == "__main__":
    main()