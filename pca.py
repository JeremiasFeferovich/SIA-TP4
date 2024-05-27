from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('europe.csv')
    data_input = data.values
    data_input = np.delete(data_input, 0, axis=1)

    data_keys = data.keys()

    # plt.boxplot(data_input, labels=data_keys[1:])
    # plt.xticks(rotation=45)
 
    standarized_data = np.zeros(data_input.shape)
    for i in range(data_input.shape[1]):
        standarized_data[:, i] = (data_input[:, i] - np.mean(data_input[:, i])) / np.std(data_input[:, i])

    # plt.figure(figsize=(10, 10))
    # plt.boxplot(standarized_data, labels=data_keys[1:])    
    # plt.xticks(rotation=45)
    # plt.title('Boxplot de los datos estandarizados')
    # plt.savefig('results/boxplot.png')
    pca = PCA(n_components=2)
    pca.fit(standarized_data)
    
    # print(f'Components: {pca.components_}; Explained variance: {pca.explained_variance_}; Explained variance ratio: {pca.explained_variance_ratio_}')
    
    # plt.figure(figsize=(15, 10))
    data_pca = pca.transform(standarized_data)
    plt.scatter(data_pca[:, 0], data_pca[:, 1])
    for i, name in enumerate(data['Country']):
        plt.text(data_pca[i, 0], data_pca[i, 1], name).set_rotation(30)
        
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


    country_names = data['Country']
    first_component = data_pca[:, 0]
    sorted_indices = np.argsort(first_component)
    country_names = country_names[sorted_indices]
    first_component = first_component[sorted_indices]
    # plt.figure(figsize=(10, 10))
    # plt.bar(country_names, first_component)
    # plt.xlabel('Country')
    # plt.ylabel('First PCA Component')
    # plt.title('First PCA Component for each Country')
    # plt.xticks(rotation=90)  # Rotate country names for better visibility
    # plt.savefig('results/first_component.png')
    plt.show()


if __name__ == "__main__":
    main()