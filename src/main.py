import src.data as data

WHITE_WINE_FILENAME = '../data/winequality-white.csv'

PERCENT_TRAINING = 0.8

def main():
    whiteTrainingData, whiteValData = data.parseAndSplit(WHITE_WINE_FILENAME,
                                                         ';',
                                                         PERCENT_TRAINING)
    whiteTrainingX, whiteTrainingY = whiteTrainingData
    whiteValX, whiteValY = whiteValData


if __name__ == '__main__':
    main()