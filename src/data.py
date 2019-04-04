# Contains a function which returns feature and label matrices given a
# csv file.

import numpy as np
import pandas as pd

# Returns X, y parsed from file name.
def parseData(fileName, delimiter=';'):
    allData = pd.read_csv(fileName, delimiter)
    X = allData.values[:, :-1]
    y = allData.values[:, -1:]
    return X, y

def splitData(n, percentTraining):
    samples = np.arange(n)
    np.random.shuffle(samples)
    numberTraining = np.ceil(n * percentTraining).astype(int)
    trainingSamples = samples[:numberTraining]
    valSamples = samples[numberTraining:]
    return trainingSamples, valSamples

def main():
    parseData('../data/winequality-red.csv')
    print(splitData(100, .871))
    print(splitData(100, 1))

if __name__ == '__main__':
    main()
