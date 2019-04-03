# Contains a function which returns feature and label matrices given a
# csv file.

import pandas as pd

# Returns X, y parsed from file name.
def parseData(fileName, delimiter=';'):
    allData = pd.read_csv(fileName, delimiter)
    X = allData.values[:, :-1]
    y = allData.values[:, -1:]
    return X, y

def main():
    parseData('../data/winequality-red.csv')

if __name__ == '__main__':
    main()
