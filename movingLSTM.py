import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from os import listdir
from os.path import isfile, join
import sklearn.metrics
import sys

if len(sys.argv) > 1:
    wandb.init(name=sys.argv[1].__str__() + " perfectTracklets movingCamera", project="tfg",
               group="definitive moving LSTM")
else:
    wandb.init(name="perfectTracklets movingCamera", project="tfg", group="prueba moving LSTM")

wandb.config.inputSize = 103
wandb.config.hiddenSize = 102
wandb.config.epochs = 100
if len(sys.argv) > 2:
    wandb.config.learningRate = float(sys.argv[2])
else:
    wandb.config.learningRate = 0.04
wandb.config.numberOfLSTMFCLayers = 0
wandb.config.numberOfClassifierFCLayers = 1
wandb.config.normalizeImageDistances = True
wandb.config.softAccuracyThreshold = 0.5
wandb.config.positiveMargin = 0.3
wandb.config.negativeMargin = 0.7
if len(sys.argv) > 1:
    wandb.config.modelSavingName = "modelPerfectTrackletsMovingCamera_" + sys.argv[1] + ".pt"
else:
    wandb.config.modelSavingName = "modelPerfectTrackletsMovingCamera.pt"

wandb.summary['maxf1ScoreValidation'] = 0

firstPass = True
lastFrame = 0

# FULL VARIABLE READING (this will contain all the information for the whole static identifications)
generalPath = r'C:\Users\rausanto\Documents\TFG\MOT17'

staticFolders = ('MOT17-05-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM', 'MOT17-13-DPM')

file = open(generalPath + chr(92) + r'FEATURES' + chr(92) + staticFolders[0] + r'.txt', 'r')
fullVariable = file.read()
file.close()

file = open(generalPath + chr(92) + r'FEATURES' + chr(92) + staticFolders[1] + r'.txt', 'r')
fullVariable = fullVariable + file.read()
file.close()

file = open(generalPath + chr(92) + r'FEATURES' + chr(92) + staticFolders[2] + r'.txt', 'r')
fullVariable = fullVariable + file.read()
file.close()

file = open(generalPath + chr(92) + r'FEATURES' + chr(92) + staticFolders[3] + r'.txt', 'r')
fullVariable = fullVariable + file.read()
file.close()

fullVariable = fullVariable.splitlines()
fullVariable = np.stack([np.array(line.split('|')) for line in fullVariable], axis=0)

fullTrainingVariable = np.empty([1, 8])

for trainingIdentity in [f.replace(".txt", "") for f in listdir(generalPath + r"\DATA\LSTMTRACKLET\movingCamera\train")
                         if isfile(join(generalPath + r"\DATA\LSTMTRACKLET\movingCamera\train", f))]:
    fullTrainingVariable = np.append(fullTrainingVariable,
                                     np.squeeze(fullVariable[np.where(fullVariable[:, 3] == trainingIdentity), :],
                                                axis=0), axis=0)

fullTrainingVariable = np.delete(fullTrainingVariable, 0, axis=0)

fullTestingVariable = np.empty([1, 8])

for testingIdentity in [f.replace(".txt", "") for f in listdir(generalPath + r"\DATA\LSTMTRACKLET\movingCamera\test") if
                        isfile(join(generalPath + r"\DATA\LSTMTRACKLET\movingCamera\test", f))]:
    fullTestingVariable = np.append(fullTestingVariable,
                                    np.squeeze(fullVariable[np.where(fullVariable[:, 3] == testingIdentity), :],
                                               axis=0), axis=0)

fullTestingVariable = np.delete(fullTestingVariable, 0, axis=0)

fullValidationVariable = np.empty([1, 8])

for validationIdentity in [f.replace(".txt", "") for f in listdir(generalPath + r"\DATA\LSTMTRACKLET\movingCamera\val")
                           if isfile(join(generalPath + r"\DATA\LSTMTRACKLET\movingCamera\val", f))]:
    fullValidationVariable = np.append(fullValidationVariable,
                                       np.squeeze(fullVariable[np.where(fullVariable[:, 3] == validationIdentity), :],
                                                  axis=0), axis=0)

fullValidationVariable = np.delete(fullValidationVariable, 0, axis=0)

del fullVariable


def get_info(index: int, loadingState: int, normalize: bool = False, maximumWidth: int = 1920,
             maximumHeight: int = 1080):
    global firstPass, lastFrame

    if loadingState == 0:  # Training
        fullVariable = fullTrainingVariable
    elif loadingState == 1:  # Validation
        fullVariable = fullValidationVariable
    elif loadingState == 2:  # Testing
        fullVariable = fullTestingVariable
    else:
        raise ValueError("The loadingState must be either:\n0 = Training\n1 = Validation\n2 = Testing")

    if not normalize:
        maximumWidth = 1
        maximumHeight = 1

    identification = fullVariable[index, :]

    featuresFileName = identification[0]
    actualFrame = int(identification[2])
    boundingBoxWidth = float(identification[6]) / maximumWidth
    boundingBoxHeight = float(identification[7]) / maximumHeight

    # Calculate the difference of frames
    if firstPass:
        lastFrame = actualFrame
        frameDifference = 0
        firstPass = False
    else:
        frameDifference = actualFrame - lastFrame
        lastFrame = actualFrame

    # THIS WILL LOAD THE INPUT DATA FOR ONE DETECTION
    inputData = []

    inputData.append(frameDifference)
    inputData.append(boundingBoxWidth)
    inputData.append(boundingBoxHeight)

    featuresFiles = open(generalPath + r'\FEATURES' + chr(92) + featuresFileName, 'r')

    features = featuresFiles.read()
    featuresFiles.close()

    features = features.split('|')

    for feature in features:
        inputData.append(float(feature))

    inputData = torch.unsqueeze(torch.unsqueeze(torch.tensor(inputData), dim=0), dim=0)

    # THIS WILL LOAD POSITIVE FEATURES FOR THE BACK-PROPAGATION WITH POSITIVE LABEL
    positiveOutputData = []

    positiveIdentification = fullVariable[index + 1, :]

    positiveOutputData.append(float(positiveIdentification[6]) / maximumWidth)
    positiveOutputData.append(float(positiveIdentification[7]) / maximumHeight)

    positiveOutputFeaturesFile = open(generalPath + r'\FEATURES' + chr(92) + positiveIdentification[0], 'r')

    positiveOutputFeatures = positiveOutputFeaturesFile.read()
    positiveOutputFeaturesFile.close()

    positiveOutputFeatures = positiveOutputFeatures.split('|')
    for positiveOutputFeature in positiveOutputFeatures:
        positiveOutputData.append(float(positiveOutputFeature))

    positiveOutputData = torch.unsqueeze(torch.unsqueeze(torch.tensor(positiveOutputData), dim=0), dim=0)

    # THIS WILL LOAD RANDOM NEGATIVE FEATURES (from the same frame as the positive)
    try:
        negativeOutputData = []

        if len(np.where(np.logical_and(np.logical_and(fullVariable[:, 3] != fullVariable[index + 1, 3],
                                                      fullVariable[:, 1] == fullVariable[index + 1, 1]),
                                       fullVariable[:, 2] == fullVariable[index + 1, 2]))[0]) != 0:
            randomNegativeIndex = np.random.choice(np.squeeze(np.asarray(np.where(np.logical_and(
                np.logical_and(fullVariable[:, 3] != fullVariable[index + 1, 3],
                               fullVariable[:, 1] == fullVariable[index + 1, 1]),
                fullVariable[:, 2] == fullVariable[index + 1, 2])))))
        elif len(np.where(np.logical_and(fullVariable[:, 3] != fullVariable[index + 1, 3],
                                         fullVariable[:, 1] == fullVariable[index + 1, 1]))[0]) != 0:
            randomNegativeIndex = np.random.choice(np.squeeze(np.asarray(np.where(
                np.logical_and(fullVariable[:, 3] != fullVariable[index + 1, 3],
                               fullVariable[:, 1] == fullVariable[index + 1, 1])))))
        else:
            randomNegativeIndex = np.random.choice(
                np.squeeze(np.asarray(np.where(fullVariable[:, 3] != fullVariable[index + 1, 3]))))

        negativeIdentification = fullVariable[randomNegativeIndex, :]

        negativeOutputData.append(float(negativeIdentification[6]) / maximumWidth)
        negativeOutputData.append(float(negativeIdentification[7]) / maximumHeight)

        negativeOutputFeaturesFile = open(
            generalPath + r'\FEATURES' + chr(92) + fullVariable[randomNegativeIndex, 0],
            'r')

        negativeOutputFeatures = negativeOutputFeaturesFile.read()
        negativeOutputFeaturesFile.close()

        negativeOutputFeatures = negativeOutputFeatures.split('|')

        for negativeOutputFeature in negativeOutputFeatures:
            negativeOutputData.append(float(negativeOutputFeature))

        negativeOutputData = torch.unsqueeze(torch.unsqueeze(torch.tensor(negativeOutputData), dim=0), dim=0)

    except:
        negativeOutputData = None

    return inputData, positiveOutputData, negativeOutputData


class StaticLSTM(nn.Module):
    def __init__(self, inputSize: int, hiddenSize: int, numberOfLSTMFCLayers: int, numberOfClassifierFCLayers: int):
        super(StaticLSTM, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numberOfLSTMFCLayers = numberOfLSTMFCLayers
        self.numberOfClassifierFCLayers = numberOfClassifierFCLayers

        self.lstm = nn.LSTM(input_size=self.inputSize, hidden_size=self.hiddenSize)

        if numberOfLSTMFCLayers > 0:
            self.fc1 = []
            for __ in range(numberOfLSTMFCLayers - 1):
                self.fc1.append(nn.Sequential(nn.Linear(in_features=self.hiddenSize, out_features=self.hiddenSize),
                                              nn.ReLU(inplace=True)))
            self.fc1.append(nn.Linear(in_features=self.hiddenSize, out_features=102))

        if numberOfClassifierFCLayers > 1:
            self.fc2 = []
            fcLayerSizes = np.linspace(2 * 102, 2, num=numberOfClassifierFCLayers + 1)
            for count in range(numberOfClassifierFCLayers - 1):
                self.fc2.append(nn.Sequential(
                    nn.Linear(in_features=int(fcLayerSizes[count]), out_features=int(fcLayerSizes[count + 1])),
                    nn.ReLU(inplace=True)))
            self.fc2.append(nn.Linear(in_features=int(fcLayerSizes[numberOfClassifierFCLayers - 1]), out_features=2))
        elif numberOfClassifierFCLayers == 1:
            self.fc2 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(in_features=102 * 2, out_features=2))

    def forward(self, inputData: torch.tensor, hidden: tuple, positiveDetection: torch.tensor = None,
                negativeDetection: torch.tensor = None):
        outputData, hidden = self.lstm(inputData, hidden)

        if self.numberOfLSTMFCLayers > 0:
            for fcLayer in self.fc1:
                outputData = fcLayer(outputData)

        if self.numberOfClassifierFCLayers > 1:
            if positiveDetection is not None and negativeDetection is not None:
                positiveOutputData = torch.cat((outputData, positiveDetection), dim=2)
                for fcLayer in self.fc2:
                    positiveOutputData = fcLayer(positiveOutputData)

                negativeOutputData = torch.cat((outputData, negativeDetection), dim=2)
                for fcLayer in self.fc2:
                    negativeOutputData = fcLayer(negativeOutputData)

                return positiveOutputData, hidden, negativeOutputData

            elif positiveDetection is not None:
                positiveOutputData = torch.cat((outputData, positiveDetection), dim=2)
                for fcLayer in self.fc2:
                    positiveOutputData = fcLayer(positiveOutputData)

                return positiveOutputData, hidden, None

            elif negativeDetection is not None:
                negativeOutputData = torch.cat((outputData, negativeDetection), dim=2)
                for fcLayer in self.fc2:
                    negativeOutputData = fcLayer(negativeOutputData)

                return None, hidden, negativeOutputData

        else:
            return outputData, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        return (weight.new(1, 1, self.hiddenSize).zero_(),
                weight.new(1, 1, self.hiddenSize).zero_())


class NormalizedContrastiveLoss(nn.Module):

    def __init__(self, marginPositive: float = 0.3, marginNegative: float = 0.7):
        super(NormalizedContrastiveLoss, self).__init__()
        self.marginPositive = marginPositive
        self.marginNegative = marginNegative

    def forward(self, output1: torch.tensor, output2: torch.tensor, label: torch.tensor,
                return_normalizedDistance: bool = False):
        output1 = torch.squeeze(output1, 0)
        output2 = torch.squeeze(output2, 0)

        # Euclidean distance of two output feature vectors
        euclideanDistance = F.pairwise_distance(output1, output2)
        euclideanDistance = torch.pow(euclideanDistance, 2)

        # Normalization of the distance
        normalizedDistance = 2 * (1 / (1 + torch.exp(-euclideanDistance)) - 0.5)

        # # perform contrastive loss calculation with the distance
        # loss_contrastive = torch.mean((1 - label) * torch.pow(euclideanDistance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclideanDistance, min=0.0), 2))

        loss = 0.5 * torch.mean(label * torch.clamp(normalizedDistance - self.marginPositive, min=0.0) + (1 - label) *
                                torch.clamp(self.marginNegative - normalizedDistance, min=0.0))

        if not return_normalizedDistance:
            return loss
        else:
            return loss, normalizedDistance


# NET DECLARATION
net = StaticLSTM(inputSize=wandb.config.inputSize, hiddenSize=wandb.config.hiddenSize,
                 numberOfLSTMFCLayers=wandb.config.numberOfLSTMFCLayers,
                 numberOfClassifierFCLayers=wandb.config.numberOfClassifierFCLayers)

# LOSS FUNCTION DECLARATION
normalizedContrastiveLossFunction = NormalizedContrastiveLoss(marginPositive=wandb.config.positiveMargin,
                                                              marginNegative=wandb.config.negativeMargin)
crossEntropyLossFunction = nn.CrossEntropyLoss()

# OPTIMIZER DECLARATION
optimizer = optim.Adagrad(net.parameters(), lr=wandb.config.learningRate)


def train():
    global firstPass

    for epoch in range(wandb.config.epochs):
        net.train()

        labels = []
        distances = []
        totalLoss = []

        for trainingFilePath in glob.glob(generalPath + r"\DATA\LSTMTRACKLET\movingCamera\train\*.txt"):
            trainingFile = open(trainingFilePath, 'r')

            firstLine = trainingFile.readline()
            trainingFile.close()

            firstLine = firstLine.split('|')
            indexes = np.where(fullTrainingVariable[:, 3] == firstLine[3])

            firstPass = True
            hidden = net.init_hidden()  # Initializes

            # net.zero_grad()

            for index in indexes[0][:-1]:  # This for loop runs all the indexes of the ID except for the last one
                inputData, positiveOutputFeatures, negativeOutputFeatures = get_info(index, loadingState=0,
                                                                                     normalize=wandb.config.normalizeImageDistances)

                if wandb.config.numberOfClassifierFCLayers > 1:  # In case the classifier is selected
                    net.zero_grad()
                    positiveOutputData, hidden, negativeOutputData = net(inputData, hidden, positiveOutputFeatures,
                                                                         negativeOutputFeatures)
                    positiveLoss = crossEntropyLossFunction(positiveOutputData[0, :], torch.tensor([1]))

                    positiveLoss.backward(retain_graph=True)

                    # METRICS
                    totalLoss.append(positiveLoss.item())
                    labels.append(True)
                    distances.append(nn.functional.softmax(positiveOutputData[0, 0, :], dim=0)[0].item())

                    if negativeOutputData is not None:
                        negativeLoss = crossEntropyLossFunction(negativeOutputData[0, :], torch.tensor([0]))

                        negativeLoss.backward(retain_graph=True)

                        # METRICS
                        totalLoss.append(negativeLoss.item())
                        labels.append(False)
                        distances.append(nn.functional.softmax(negativeOutputData[0, 0, :], dim=0)[0].item())

                    optimizer.step()

                else:  # In case there is not classifier
                    net.zero_grad()
                    outputData, hidden = net(inputData, hidden)
                    positiveLoss, normalizedDistance = normalizedContrastiveLossFunction(outputData,
                                                                                         positiveOutputFeatures,
                                                                                         label=1,
                                                                                         return_normalizedDistance=True)
                    positiveLoss.backward(retain_graph=True)
                    # optimizer.step()

                    # METRICS
                    totalLoss.append(positiveLoss.item())
                    labels.append(True)
                    distances.append(normalizedDistance)

                    if negativeOutputFeatures is not None:
                        negativeLoss, normalizedDistance = normalizedContrastiveLossFunction(outputData,
                                                                                             negativeOutputFeatures,
                                                                                             label=0,
                                                                                             return_normalizedDistance=True)
                        negativeLoss.backward(retain_graph=True)
                        # optimizer.step()

                        # METRICS
                        totalLoss.append(negativeLoss.item())
                        labels.append(False)
                        distances.append(normalizedDistance)

                    optimizer.step()

            # optimizer.step()

        # METRICS
        truePositive = np.asarray(
            np.where(np.logical_and(labels, np.squeeze(distances) < wandb.config.softAccuracyThreshold))).size
        falseNegative = np.asarray(
            np.where(np.logical_and(labels, np.squeeze(distances) >= wandb.config.softAccuracyThreshold))).size
        trueNegative = np.asarray(
            np.where(np.logical_and(np.logical_not(labels),
                                    np.squeeze(distances) > wandb.config.softAccuracyThreshold))).size
        falsePositive = np.asarray(
            np.where(np.logical_and(np.logical_not(labels),
                                    np.squeeze(distances) <= wandb.config.softAccuracyThreshold))).size

        softAccuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)
        hardAccuracy = (np.squeeze(np.asarray(
            np.where(np.logical_and(labels, np.squeeze(distances) < wandb.config.positiveMargin)))).size
                        + np.squeeze(np.asarray(np.where(
                    np.logical_and(np.logical_not(labels), np.squeeze(distances) > wandb.config.negativeMargin))))).size \
                       / len(labels)

        try:
            # Precision minimizes False Positives
            precision = truePositive / (truePositive + falsePositive)
        except:
            precision = None
        try:
            # Recall minimizes False Negatives
            recall = truePositive / (truePositive + falseNegative)
        except:
            recall = None
        try:
            f1Score = 2 * precision * recall / (precision + recall)
        except:
            f1Score = None
        try:
            matthewsCorrelationCoefficient = (truePositive * trueNegative - falsePositive * falseNegative) / \
                                             np.sqrt((truePositive + falsePositive) * (truePositive + falseNegative) *
                                                     (trueNegative + falsePositive) * (trueNegative + falseNegative))
        except:
            matthewsCorrelationCoefficient = None

        print('TRAINING EPOCH: ' + epoch.__str__())
        print('\tmeanLossTraining: ' + np.mean(totalLoss).__str__())
        print('\ttruePositiveTraining: ' + truePositive.__str__())
        print('\ttrueNegativeTraining: ' + trueNegative.__str__())
        print('\tfalsePositiveTraining: ' + falsePositive.__str__())
        print('\tfalseNegativeTraining: ' + falseNegative.__str__())
        print('\tsoftAccuracyTraining: ' + softAccuracy.__str__())
        print('\thardAccuracyTraining: ' + hardAccuracy.__str__())
        print('\tprecisionTraining: ' + precision.__str__())
        print('\trecallTraining: ' + recall.__str__())
        print('\tf1ScoreTraining: ' + f1Score.__str__())
        print('\tmatthewsCorrelationCoefficientTraining: ' + matthewsCorrelationCoefficient.__str__())

        wandb.log({
            'meanLossTraining': np.mean(totalLoss),
            'truePositiveTraining': truePositive,
            'trueNegativeTraining': trueNegative,
            'falsePositiveTraining': falsePositive,
            'falseNegativeTraining': falseNegative,
            'softAccuracyTraining': softAccuracy,
            'hardAccuracyTraining': hardAccuracy,
            'precisionTraining': precision,
            'recallTraining': recall,
            'f1ScoreTraining': f1Score,
            'matthewsCorrelationCoefficientTraining': matthewsCorrelationCoefficient
        }, step=epoch)

        # CONFUSION MATRIX
        confusionMatrixData = np.array([[trueNegative / len(labels), falsePositive / len(labels)],
                                        [falseNegative / len(labels), truePositive / len(labels)]])

        wandb.log({'confusionMatrixTraining': wandb.plots.HeatMap(['0 Predicted', '1 Predicted'], ['0 True', '1 True'],
                                                                  confusionMatrixData, show_text=True)}, step=epoch)
        #
        # plt.matshow(confusionMatrixData, cmap='plasma', vmin=0, vmax=1)
        # plt.title("Confusion Matrix. Training Epoch=" + epoch.__str__())
        # ax = plt.gca()
        # ax.xaxis.tick_bottom()
        # plt.colorbar()
        # plt.ylabel("True label")
        # plt.xlabel("Predicted label")
        #
        # wandb.log({"confusionMatrixTraining": plt}, step=epoch)

        validation(epoch=epoch)


def validation(epoch: int):
    global firstPass

    net.eval()

    labels = []
    distances = []
    totalLoss = []

    with torch.no_grad():
        for validationFilePath in glob.glob(generalPath + r"\DATA\LSTMTRACKLET\movingCamera\val\*.txt"):
            validationFile = open(validationFilePath, 'r')

            firstLine = validationFile.readline()
            validationFile.close()

            firstLine = firstLine.split('|')
            indexes = np.where(fullValidationVariable[:, 3] == firstLine[3])

            firstPass = True
            hidden = net.init_hidden()  # Initializes

            for index in indexes[0][:-1]:  # This for loop runs all the indexes of the ID except for the last one
                inputData, positiveOutputFeatures, negativeOutputFeatures = get_info(index, loadingState=1,
                                                                                     normalize=wandb.config.normalizeImageDistances)

                if wandb.config.numberOfClassifierFCLayers > 1:  # In case the classifier is selected
                    positiveOutputData, hidden, negativeOutputData = net(inputData, hidden, positiveOutputFeatures,
                                                                         negativeOutputFeatures)
                    positiveLoss = crossEntropyLossFunction(positiveOutputData[0, :], torch.tensor([1]))

                    # METRICS
                    totalLoss.append(positiveLoss.item())
                    labels.append(True)
                    distances.append(nn.functional.softmax(positiveOutputData[0, 0, :], dim=0)[0].item())

                    if negativeOutputData is not None:
                        negativeLoss = crossEntropyLossFunction(negativeOutputData[0, :], torch.tensor([0]))

                        # METRICS
                        totalLoss.append(negativeLoss.item())
                        labels.append(False)
                        distances.append(nn.functional.softmax(negativeOutputData[0, 0, :], dim=0)[0].item())

                else:  # In case there is not classifier
                    outputData, hidden = net(inputData, hidden)
                    positiveLoss, normalizedDistance = normalizedContrastiveLossFunction(outputData,
                                                                                         positiveOutputFeatures,
                                                                                         label=1,
                                                                                         return_normalizedDistance=True)

                    # METRICS
                    totalLoss.append(positiveLoss.item())
                    labels.append(True)
                    distances.append(normalizedDistance)

                    if negativeOutputFeatures is not None:
                        negativeLoss, normalizedDistance = normalizedContrastiveLossFunction(outputData,
                                                                                             negativeOutputFeatures,
                                                                                             label=0,
                                                                                             return_normalizedDistance=True)

                        # METRICS
                        totalLoss.append(negativeLoss.item())
                        labels.append(False)
                        distances.append(normalizedDistance)

    # METRICS
    truePositive = np.asarray(
        np.where(np.logical_and(labels, np.squeeze(distances) < wandb.config.softAccuracyThreshold))).size
    falseNegative = np.asarray(
        np.where(np.logical_and(labels, np.squeeze(distances) >= wandb.config.softAccuracyThreshold))).size
    trueNegative = np.asarray(
        np.where(
            np.logical_and(np.logical_not(labels), np.squeeze(distances) > wandb.config.softAccuracyThreshold))).size
    falsePositive = np.asarray(
        np.where(
            np.logical_and(np.logical_not(labels), np.squeeze(distances) <= wandb.config.softAccuracyThreshold))).size

    softAccuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)
    hardAccuracy = (np.squeeze(np.asarray(
        np.where(np.logical_and(labels, np.squeeze(distances) < wandb.config.positiveMargin)))).size
                    + np.squeeze(np.asarray(np.where(
                np.logical_and(np.logical_not(labels), np.squeeze(distances) > wandb.config.negativeMargin))))).size \
                   / len(labels)

    try:
        # Precision minimizes False Positives
        precision = truePositive / (truePositive + falsePositive)
    except:
        precision = None
    try:
        # Recall minimizes False Negatives
        recall = truePositive / (truePositive + falseNegative)
    except:
        recall = None
    try:
        f1Score = 2 * precision * recall / (precision + recall)
    except:
        f1Score = None
    try:
        matthewsCorrelationCoefficient = (truePositive * trueNegative - falsePositive * falseNegative) / \
                                         np.sqrt((truePositive + falsePositive) * (truePositive + falseNegative) *
                                                 (trueNegative + falsePositive) * (trueNegative + falseNegative))
    except:
        matthewsCorrelationCoefficient = None

    print('TESTING EPOCH: ' + epoch.__str__())
    print('\tmeanLossValidation: ' + np.mean(totalLoss).__str__())
    print('\ttruePositiveValidation: ' + truePositive.__str__())
    print('\ttrueNegativeValidation: ' + trueNegative.__str__())
    print('\tfalsePositiveValidation: ' + falsePositive.__str__())
    print('\tfalseNegativeValidation: ' + falseNegative.__str__())
    print('\tsoftAccuracyValidation: ' + softAccuracy.__str__())
    print('\thardAccuracyValidation: ' + hardAccuracy.__str__())
    print('\tprecisionValidation: ' + precision.__str__())
    print('\trecallValidation: ' + recall.__str__())
    print('\tf1ScoreValidation: ' + f1Score.__str__())
    print('\tmatthewsCorrelationCoefficientValidation: ' + matthewsCorrelationCoefficient.__str__())

    wandb.log({
        'meanLossValidation': np.mean(totalLoss),
        'truePositiveValidation': truePositive,
        'trueNegativeValidation': trueNegative,
        'falsePositiveValidation': falsePositive,
        'falseNegativeValidation': falseNegative,
        'softAccuracyValidation': softAccuracy,
        'hardAccuracyValidation': hardAccuracy,
        'precisionValidation': precision,
        'recallValidation': recall,
        'f1ScoreValidation': f1Score,
        'matthewsCorrelationCoefficientValidation': matthewsCorrelationCoefficient
    }, step=epoch)

    # CONFUSION MATRIX
    confusionMatrixData = np.array([[trueNegative / len(labels), falsePositive / len(labels)],
                                    [falseNegative / len(labels), truePositive / len(labels)]])

    wandb.log({'confusionMatrixValidation': wandb.plots.HeatMap(['0 Predicted', '1 Predicted'], ['0 True', '1 True'],
                                                                confusionMatrixData, show_text=True)}, step=epoch)

    if f1Score > wandb.summary['maxf1ScoreValidation']:
        wandb.summary['maxf1ScoreValidation'] = f1Score
        wandb.summary['maxEpoch'] = epoch

        torch.save({'model_state_dict': net.state_dict(),
                    'inputSize': wandb.config.inputSize,
                    'hiddenSize': wandb.config.hiddenSize,
                    'numberOfLSTMFCLayers': wandb.config.numberOfLSTMFCLayers,
                    'numberOfClassifierFCLayers': wandb.config.numberOfClassifierFCLayers},
                   wandb.config.modelSavingName)

        wandb.save(join(wandb.run.dir, wandb.config.modelSavingName))


def test():
    global firstPass

    try:
        savedParameters = torch.load(wandb.config.modelSavingName)

        model = StaticLSTM(inputSize=savedParameters['inputSize'], hiddenSize=savedParameters['hiddenSize'],
                           numberOfLSTMFCLayers=savedParameters['numberOfLSTMFCLayers'],
                           numberOfClassifierFCLayers=savedParameters['numberOfClassifierFCLayers'])

        model.load_state_dict(savedParameters['model_state_dict'])
    except Exception as e:
        print(e)
        model = StaticLSTM(inputSize=wandb.config.inputSize, hiddenSize=wandb.config.hiddenSize,
                           numberOfLSTMFCLayers=wandb.config.numberOfLSTMFCLayers,
                           numberOfClassifierFCLayers=wandb.config.numberOfClassifierFCLayers)

    model.eval()

    labels = []
    distances = []
    totalLoss = []

    with torch.no_grad():
        for testingFilePath in glob.glob(generalPath + r"\DATA\LSTMTRACKLET\movingCamera\test\*.txt"):
            testingFile = open(testingFilePath, 'r')

            firstLine = testingFile.readline()
            testingFile.close()

            firstLine = firstLine.split('|')
            indexes = np.where(fullTestingVariable[:, 3] == firstLine[3])

            firstPass = True
            hidden = model.init_hidden()  # Initializes

            for index in indexes[0][:-1]:  # This for loop runs all the indexes of the ID except for the last one
                inputData, positiveOutputFeatures, negativeOutputFeatures = get_info(index, loadingState=2,
                                                                                     normalize=wandb.config.normalizeImageDistances)

                if wandb.config.numberOfClassifierFCLayers > 1:  # In case the classifier is selected
                    positiveOutputData, hidden, negativeOutputData = model(inputData, hidden, positiveOutputFeatures,
                                                                           negativeOutputFeatures)
                    positiveLoss = crossEntropyLossFunction(positiveOutputData[0, :], torch.tensor([1]))

                    # METRICS
                    totalLoss.append(positiveLoss.item())
                    labels.append(True)
                    distances.append(nn.functional.softmax(positiveOutputData[0, 0, :], dim=0)[0].item())

                    if negativeOutputData is not None:
                        negativeLoss = crossEntropyLossFunction(negativeOutputData[0, :], torch.tensor([0]))

                        # METRICS
                        totalLoss.append(negativeLoss.item())
                        labels.append(False)
                        distances.append(nn.functional.softmax(negativeOutputData[0, 0, :], dim=0)[0].item())

                else:  # In case there is not classifier
                    outputData, hidden = model(inputData, hidden)
                    positiveLoss, normalizedDistance = normalizedContrastiveLossFunction(outputData,
                                                                                         positiveOutputFeatures,
                                                                                         label=1,
                                                                                         return_normalizedDistance=True)

                    # METRICS
                    totalLoss.append(positiveLoss.item())
                    labels.append(True)
                    distances.append(normalizedDistance)

                    if negativeOutputFeatures is not None:
                        negativeLoss, normalizedDistance = normalizedContrastiveLossFunction(outputData,
                                                                                             negativeOutputFeatures,
                                                                                             label=0,
                                                                                             return_normalizedDistance=True)

                        # METRICS
                        totalLoss.append(negativeLoss.item())
                        labels.append(False)
                        distances.append(normalizedDistance)

    # METRICS
    truePositive = np.asarray(
        np.where(np.logical_and(labels, np.squeeze(distances) < wandb.config.softAccuracyThreshold))).size
    falseNegative = np.asarray(
        np.where(np.logical_and(labels, np.squeeze(distances) >= wandb.config.softAccuracyThreshold))).size
    trueNegative = np.asarray(
        np.where(
            np.logical_and(np.logical_not(labels), np.squeeze(distances) > wandb.config.softAccuracyThreshold))).size
    falsePositive = np.asarray(
        np.where(
            np.logical_and(np.logical_not(labels), np.squeeze(distances) <= wandb.config.softAccuracyThreshold))).size

    softAccuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)
    hardAccuracy = (np.squeeze(np.asarray(
        np.where(np.logical_and(labels, np.squeeze(distances) < wandb.config.positiveMargin)))).size
                    + np.squeeze(np.asarray(np.where(
                np.logical_and(np.logical_not(labels), np.squeeze(distances) > wandb.config.negativeMargin))))).size \
                   / len(labels)

    try:
        # Precision minimizes False Positives
        precision = truePositive / (truePositive + falsePositive)
    except:
        precision = None
    try:
        # Recall minimizes False Negatives
        recall = truePositive / (truePositive + falseNegative)
    except:
        recall = None
    try:
        f1Score = 2 * precision * recall / (precision + recall)
    except:
        f1Score = None
    try:
        matthewsCorrelationCoefficient = (truePositive * trueNegative - falsePositive * falseNegative) / \
                                         np.sqrt((truePositive + falsePositive) * (truePositive + falseNegative) *
                                                 (trueNegative + falsePositive) * (trueNegative + falseNegative))
    except:
        matthewsCorrelationCoefficient = None

    print('\tmeanLossTesting: ' + np.mean(totalLoss).__str__())
    print('\ttruePositiveTesting: ' + truePositive.__str__())
    print('\ttrueNegativeTesting: ' + trueNegative.__str__())
    print('\tfalsePositiveTesting: ' + falsePositive.__str__())
    print('\tfalseNegativeTesting: ' + falseNegative.__str__())
    print('\tsoftAccuracyTesting: ' + softAccuracy.__str__())
    print('\thardAccuracyTesting: ' + hardAccuracy.__str__())
    print('\tprecisionTesting: ' + precision.__str__())
    print('\trecallTesting: ' + recall.__str__())
    print('\tf1ScoreTesting: ' + f1Score.__str__())
    print('\tmatthewsCorrelationCoefficientTesting: ' + matthewsCorrelationCoefficient.__str__())
    print('\tAUC: ' + sklearn.metrics.roc_auc_score(labels, distances).__str__())

    wandb.summary['meanLossTesting'] = np.mean(totalLoss)
    wandb.summary['truePositiveTesting'] = truePositive
    wandb.summary['trueNegativeTesting'] = trueNegative
    wandb.summary['falsePositiveTesting'] = falsePositive
    wandb.summary['falseNegativeTesting'] = falseNegative
    wandb.summary['softAccuracyTesting'] = softAccuracy
    wandb.summary['hardAccuracyTesting'] = hardAccuracy
    wandb.summary['precisionTesting'] = precision
    wandb.summary['recallTesting'] = recall
    wandb.summary['f1ScoreTesting'] = f1Score
    wandb.summary['matthewsCorrelationCoefficientTesting'] = matthewsCorrelationCoefficient
    wandb.summary['AUC'] = 1 - sklearn.metrics.roc_auc_score(labels, distances)

    # CONFUSION MATRIX
    confusionMatrixData = np.array([[trueNegative / len(labels), falsePositive / len(labels)],
                                    [falseNegative / len(labels), truePositive / len(labels)]])

    wandb.log({'confusionMatrixTesting': wandb.plots.HeatMap(['0 Predicted', '1 Predicted'], ['0 True', '1 True'],
                                                             confusionMatrixData, show_text=True)})

    tpr, fpr, thresholds = sklearn.metrics.roc_curve(labels, distances)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % (1 - sklearn.metrics.roc_auc_score(labels, distances)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)

    wandb.log({'ROC': plt})


train()

test()
