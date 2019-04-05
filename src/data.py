# Contains a function which returns feature and label matrices given a
# csv file.

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

# Returns X, y parsed from file name.
def parseData(fileName, delimiter=';', norm=True):
    allData = pd.read_csv(fileName, delimiter).values
    if norm:
        allData = normalize(allData, axis=0)
    X = allData[:, :-1]
    y = allData[:, -1:].ravel()
    return X, y

def splitData(n, percentTraining=0.8):
    samples = np.arange(n)
    np.random.shuffle(samples)
    numberTraining = np.ceil(n * percentTraining).astype(int)
    trainingSamples = samples[:numberTraining]
    valSamples = samples[numberTraining:]
    return trainingSamples, valSamples

def parseAndSplit(fileName, delimiter=';', percentTraining=0.8):
    allX, allY = parseData(fileName, delimiter)
    trainingSamples, valSamples = splitData(allY.size, percentTraining)
    training = (allX[trainingSamples], allY[trainingSamples])
    val = (allX[valSamples], allY[valSamples])
    return training, val

def main():
    X, y = parseData('../data/winequality-red.csv')
    print(splitData(100, .871))
    print(splitData(100, 1))
    print(X)

if __name__ == '__main__':
    main()
