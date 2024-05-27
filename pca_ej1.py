from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('europe.csv')
    data_input = data.values
    data_input = np.delete(data_input, 0, axis=1)

    data_keys = data.keys() 
 
    standarized_data = np.zeros(data_input.shape)
    for i in range(data_input.shape[1]):
        standarized_data[:, i] = (data_input[:, i] - np.mean(data_input[:, i])) / np.std(data_input[:, i])

    pca = PCA(n_components=2)
    pca.fit(standarized_data)
    
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

    plt.imshow(matrix)
    plt.axis('off')
    plt.savefig('results/interpolated_colors.png')

    countries_matrix = [
        [["Bulgaria", "Estonia", "Hungary", "Latvia", "Lithuania", "Ukrania"], ["Slovakia"], ["Belgium, Czech Republic"], ["Austria", "Denmark", "Luxembourg", "Netherlands", "Switzerland"]],
        [["Croatia", "Poland"], ["Slovenia"], [], ["Iceland"]],
        [["Greece"], ["Portugal"], [], ["Ireland"]],
        [["Finland", "Germany", "Norway", "United Kingdom"], [], ["Italy"], ["Spain", "Sweden"]]
    ]

    plt.figure(figsize=(15, 10))
    data_pca = pca.transform(standarized_data)
    plt.scatter(data_pca[:, 0], data_pca[:, 1])
    for i, name in enumerate(data['Country']):
        #  I want to color the background of the country name with the corresponding color
        for j in range(4):
            for k in range(4):
                if name in countries_matrix[j][k]:
                    plt.text(data_pca[i, 0], data_pca[i, 1], name, backgroundcolor=np.append(matrix[j, k]/255, 0.5), color='black').set_rotation(35)
                    break

        # plt.text(data_pca[i, 0], data_pca[i, 1], name).set_rotation(30)
        
    components_scaled = pca.components_ * np.sqrt(pca.explained_variance_[:, np.newaxis])
    print(f'Components scaled: {components_scaled}')
    for i in range(components_scaled.shape[1]):
        comp = components_scaled[:, i]
        plt.arrow(0, 0, comp[0], comp[1], color='r', alpha=0.5)
        if data_keys is not None:
            plt.text(comp[0]* 1.15, comp[1] * 1.15, data_keys[i+1], color = 'g', ha = 'center', va = 'center')

    plt.title('Valores de las Componentes Principales 1 y 2 ')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('results/pca.png')
    
    plt.show()




if __name__ == "__main__":
    main()

