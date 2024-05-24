import numpy as np
import matplotlib.pyplot as plt

def salt_and_pepper_noise(vector, intensity=0.1, shape=(5, 5)):
    noise_matrix = vector.reshape(shape[0], shape[1])

    for i in range(noise_matrix.shape[0]):
        for j in range(noise_matrix.shape[1]):
            if np.random.random() < intensity:
                if noise_matrix[i][j] == 1:
                    noise_matrix[i][j] = -1
                else:
                    noise_matrix[i][j] = 1

    return noise_matrix.flatten()

def print_number(number, data, dims=(5,5)):
    # Sample 2D list
    # Convert the list of lists to a numpy array
    array_data = np.array(data).reshape(dims[0], dims[1])

    # Display the data as an image
    plt.figure()
    plt.title(f'Number: {number}')
    plt.imshow(array_data, cmap='gray_r')  # 'gray_r' is reversed grayscale: 0=white, 1=black
    plt.axis('off')  # Turn off axis numbers and ticks
    # plt.savefig(f'plot{number}.png', dpi=300, bbox_inches='tight')  # Adjust dpi and bbox as needed
    # plt.show()