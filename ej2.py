import numpy as np
import pandas as pd
from letter_printer import get_letters
import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from noise import salt_and_pepper_noise, gaussian_noise, print_letter, print_group, noisify

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

        while iterations < 50:
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
        
        return pattern, pattern_history
    

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
    print(df.head(15).style.format({'|<,>| medio': '{:.2f}'}).hide(axis='index').data)
    print(df.tail(5).style.format({'|<,>| medio': '{:.2f}'}).hide(axis='index').data)
    chosen_group_letters = df.iloc[0]['Grupo']
    chosen_group = np.array([v for k, v in flat_letters.items() if k in chosen_group_letters])

    correct_rates = []
    # for k in range(0, 25):
    #     correct_rate = []
    #     for _ in range(50):
    #         copied_group = np.copy(chosen_group)
    #         noisy_group_1 = np.array([noisify(v, k, shape=(5,5)) for v in copied_group])
    #         hopfield_network = HopfieldNetwork(chosen_group)

    #         chosen_correctly = []

    #         for i in range(len(chosen_group)):
    #             result, pattern_history = hopfield_network.predict(noisy_group_1[i], name=f'Noisy {chosen_group_letters[i]}')

    #             # fps = 2  # frames per second
    #             # delay_seconds = 2
    #             # extra_frames = fps * delay_seconds
    #             # pattern_history = pattern_history + [pattern_history[-1]] * extra_frames

    #             # fig, ax = plt.subplots()
    #             # line, = ax.plot([], [], lw=2)

    #             # def init():
    #             #     line.set_data([], [])
    #             #     return line,

    #             # def update(frame):
    #             #     ax.clear()
    #             #     ax.imshow(pattern_history[frame].reshape(5, 5), cmap='gray_r')
    #             #     ax.axis('off')
    #             #     return ax,

    #             # ani = FuncAnimation(fig, update, frames=len(pattern_history), init_func=init, blit=False,  interval=500, repeat_delay=1000)
    #             # ani.save(f'results/result_animation_{i}.gif', writer='pillow', fps=fps)
                

    #             # Find result in group_1
    #             found = False
    #             for j, letter in enumerate(chosen_group):
    #                 if np.array_equal(result, letter):
    #                     print(f'Should be: {chosen_group_letters[i]}. Letter found: {chosen_group_letters[j]}')
    #                     found = True
    #                     chosen_correctly.append(True)
    #                     break
    #             if not found:
    #                 print(f'Should be: {chosen_group_letters[i]}. Letter not found.')
    #                 chosen_correctly.append(False)
                
    #             correct_rate.append(chosen_correctly.count(True) / len(chosen_correctly))
    #     correct_rates.append(correct_rate)

    # plt.figure()
    # x_values = [(i/25)*100 for i in range(0, 25)]
    # plt.errorbar(x_values, np.mean(correct_rates, axis=1), yerr=np.std(correct_rates, axis=1), fmt='-o', capsize=6)
    # plt.xlabel('Cantidad de ruido (%)')
    # plt.ylabel('Tasa de aciertos')
    # plt.title('Tasa de aciertos en función de la cantidad de ruido')
    # plt.savefig('results/accuracy_vs_noise.png', dpi=300, bbox_inches='tight')
    # plt.close()


    # copied_group = np.copy(chosen_group)
    # noisy_group_1 = np.array([noisify(v, 20, shape=(5,5)) for v in copied_group])
    # hopfield_network = HopfieldNetwork(chosen_group)

    # chosen_correctly = []

    # for i in range(len(chosen_group)):
    #     result, pattern_history = hopfield_network.predict(noisy_group_1[i], name=f'Noisy {chosen_group_letters[i]}')

    #     fps = 2  # frames per second
    #     delay_seconds = 2
    #     extra_frames = fps * delay_seconds
    #     pattern_history = pattern_history + [pattern_history[-1]] * extra_frames

    #     fig, ax = plt.subplots()
    #     line, = ax.plot([], [], lw=2)

    #     def init():
    #         line.set_data([], [])
    #         return line,

    #     def update(frame):
    #         ax.clear()
    #         ax.imshow(pattern_history[frame].reshape(5, 5), cmap='gray_r')
    #         ax.axis('off')
    #         return ax,

    #     ani = FuncAnimation(fig, update, frames=len(pattern_history), init_func=init, blit=False,  interval=500, repeat_delay=1000)
    #     ani.save(f'results/result_animation_{i}.gif', writer='pillow', fps=fps)
        
    #     # Find result in group_1
    #     found = False
    #     for j, letter in enumerate(chosen_group):
    #         if np.array_equal(result, letter):
    #             print(f'Should be: {chosen_group_letters[i]}. Letter found: {chosen_group_letters[j]}')
    #             found = True
    #             chosen_correctly.append(chosen_group_letters[j])
    #             break
    #     if not found:
    #         print(f'Should be: {chosen_group_letters[i]}. Letter not found.')
    #         chosen_correctly.append(False)

    # print_group(chosen_group, 'Original group')
    # print_group(noisy_group_1, 'Noisy group', chosen_correctly)


    # Define corner colors in RGB
    C00 = np.array([255, 0, 0])     # Red
    C03 = np.array([0, 255, 0])     # Green
    C30 = np.array([0, 0, 255])     # Blue
    C33 = np.array([255, 255, 0])   # Yellow

    # Define a function for linear interpolation
    def interpolate(color1, color2, t):
        return (1 - t) * color1 + t * color2

    # Initialize a 4x4x3 matrix to hold RGB values
    matrix = np.zeros((4, 4, 3), dtype=np.uint8)

    # Populate the matrix with interpolated colors
    for i in range(4):
        for j in range(4):
            t_row = i / 3  # Interpolation factor for rows
            t_col = j / 3  # Interpolation factor for columns
            
            # Interpolate top row colors (C00 to C03) and bottom row colors (C30 to C33)
            top_color = interpolate(C00, C03, t_col)
            bottom_color = interpolate(C30, C33, t_col)
            
            # Interpolate between the top and bottom interpolated colors
            matrix[i, j] = interpolate(top_color, bottom_color, t_row)

    # Display the result using Matplotlib
    plt.imshow(matrix)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()