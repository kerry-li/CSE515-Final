import src.data as data
import src.learn.blr as blr
import src.learn.gpr as gpr

WHITE_WINE_FILENAME = '../data/winequality-white.csv'

PERCENT_TRAINING = 0.8

def main():
    whiteTrainingData, whiteValData = data.parseAndSplit(WHITE_WINE_FILENAME,
                                                         ';',
                                                         PERCENT_TRAINING)
    whiteTrainingX, whiteTrainingY = whiteTrainingData
    whiteValX, whiteValY = whiteValData

    whiteLinear = blr.fit(whiteTrainingX, whiteTrainingY)
    whiteGp = gpr.trainGaussianProcess(whiteTrainingX, whiteTrainingY)

    whiteLinearPredict = whiteLinear.predict(whiteValX)
    whiteGpPredict = whiteGp.predict(whiteValX)

    whiteLinearMse = ((whiteLinearPredict - whiteValY) ** 2).mean()
    whiteGpMse = ((whiteGpPredict - whiteValY) ** 2).mean()

    print(whiteLinearMse)
    print(whiteGpMse)




if __name__ == '__main__':
    main()