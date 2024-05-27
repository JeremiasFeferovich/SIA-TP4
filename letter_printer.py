import itertools
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_letter_plot(letter, ax, cmap='Blues'):
    p= sns.heatmap(letter, ax=ax, annot=False, cbar=False, cmap=cmap, square=True, linewidth=2,linecolor='black')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    return p

def print_letters_line(letters, cmap='Blues', cmaps=[], ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, len(letters))
        fig.set_dpi(1000)
    if not cmaps:
        cmaps = [cmap]*len(letters)
    if len(cmaps) != len(letters):
        raise Exception('Number of cmaps must be equal to number of letters')
    for i, subplot in enumerate(ax):
        create_letter_plot(letters[i].reshape(5,5), ax=subplot, cmap=cmaps[i])

def get_letters():
    with open('letters.txt', 'r') as fp:
        letters = {}
        current = np.ones((5,5))*-1
        idx = 0
        for line in fp:
            if line[0]== '=':
                letters[string.ascii_uppercase[len(letters)]] = current
                current = np.ones((5,5))*-1
                idx = 0
            else:
                for i,c in enumerate(line.strip('\n')):
                    current[idx][i] = 1 if c == '*' else -1
                idx += 1
    return letters

def test_front():
    letters = list(get_letters().values())
    n = 6
    letters += [np.ones((5,5))*-1]*(n - len(letters)%n)
    fig, ax = plt.subplots(len(letters)//n, n)
    for i, letter_group in enumerate([letters[i*n:(i+1)*n] for i in range(len(letters)//n)]):
        print_letters_line(letter_group, ax=ax[i])

    plt.savefig('results/letters.png', dpi=1000, bbox_inches='tight')
    plt.show()

def main():
    test_front()

if __name__ == "__main__":
    main()