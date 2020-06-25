import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import time
import numpy as np
import os

# Init wandb
import wandb
wandb.init(name="VGG11_modified_wandb", project="tfg")


start_time = time.time()


generalPath = r"C:\Users\rausanto\Documents\TFG\MOT17"

savePath = "model.pt"
# savePath = "VGG11_modified_wandb_"
loadPath = "VGG11_modified_wandb_16000.pt"

# Configuration parameters
wandb.config.batchSizeTraining = 64
wandb.config.batchSizeValidation = 32
wandb.config.learningRate = 1e-4
wandb.config.numberOfIterations = 16000
wandb.config.numberOfIterationsValidation = 100
wandb.config.showEvery = 100
wandb.config.saveEvery = 4000

training = True
loading = False
wandb.config.training = training
wandb.config.loading = loading


# Pair tracklets paths. There two files fro training data and two for testing.
train_aPath = generalPath + r"\DATA\PAIR\train_a.txt"
train_bPath = generalPath + r"\DATA\PAIR\train_b.txt"
test_aPath = generalPath + r"\DATA\PAIR\val_a.txt"
test_bPath = generalPath + r"\DATA\PAIR\val_b.txt"

train_a = open(train_aPath, 'r')
train_b = open(train_bPath, 'r')
test_a = open(test_aPath, 'r')
test_b = open(test_bPath, 'r')


# THIS FUNCTION READS AND RETURNS THE DATA
def load_data_train():
    # This function reads a single line in both txt files,
    # returns both images as tensors already treated and the positive o negative label

    global train_a, train_b, test_a, test_b

    correct = False

    # This loop it's used to avoid problems in case the detection it's not found.
    while not correct:
        try:
            if training:
                line_a = train_a.readline()
                line_b = train_b.readline()
            else:
                line_a = test_a.readline()
                line_b = test_b.readline()

            if line_a == "":  # End Of File, both files get re-opened so they start from the beginning again
                if training:
                    train_a.close()
                    train_b.close()

                    train_a = open(train_aPath, 'r')
                    train_b = open(train_bPath, 'r')

                    line_a = train_a.readline()
                    line_b = train_b.readline()

                else:
                    test_a.close()
                    test_b.close()

                    test_a = open(test_aPath, 'r')
                    test_b = open(test_bPath, 'r')

                    line_a = test_a.readline()
                    line_b = test_b.readline()

            line_a = line_a[0: line_a.find('\n')]
            line_b = line_b[0: line_b.find('\n')]

            line_a = line_a.replace('/', '\\')
            line_b = line_b.replace('/', '\\')

            line_a = line_a.split(' ')
            line_b = line_b.split(' ')

            image_a = Image.open(generalPath + "\SAMPLES\\train\\" + line_a[0])
            image_b = Image.open(generalPath + "\SAMPLES\\train\\" + line_b[0])

            correct = True
        except Exception as e:
            # print(e)
            correct = False

    # Preparing the data as Pytorch Tensors
    toTensorTransform = transforms.ToTensor()
    resizeTransform = transforms.Resize((128, 64))  # Images are resized to 128 x 64 pixels

    image_a = resizeTransform(image_a)
    image_b = resizeTransform(image_b)

    image_a = toTensorTransform(image_a)
    image_b = toTensorTransform(image_b)

    image_a = torch.unsqueeze(image_a, 0)  # Add the forth dimension for the batch size
    image_b = torch.unsqueeze(image_b, 0)

    # Calculates the label by checking if the two detections correspond to the same identity
    if line_a[1] == line_b[1]:
        label = torch.ones(1)
    else:
        label = torch.zeros(1)

    return image_a, image_b, label


# FUNCTION THAT RETURNS THE BATCHES
def prepare_batch():
    if training:
        images_a = torch.empty([wandb.config.batchSizeTraining, 3, 128, 64])  #
        images_b = torch.empty([wandb.config.batchSizeTraining, 3, 128, 64])
        labels = torch.empty(wandb.config.batchSizeTraining)

        for ii in range(wandb.config.batchSizeTraining):
            image_a, image_b, label = load_data_train()

            images_a[ii] = image_a
            images_b[ii] = image_b
            labels[ii] = label

    else:
        images_a = torch.empty([wandb.config.batchSizeValidation, 3, 128, 64])  #
        images_b = torch.empty([wandb.config.batchSizeValidation, 3, 128, 64])
        labels = torch.empty(wandb.config.batchSizeValidation)

        for ii in range(wandb.config.batchSizeValidation):
            image_a, image_b, label = load_data_train()

            images_a[ii] = image_a
            images_b[ii] = image_b
            labels[ii] = label

    return images_a, images_b, labels


# NEURAL NETWORK ARCHITECTURE DECLARATION
class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()

        self.cnnLayers = nn.Sequential(
            # cnn_1_1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # cnn_2_1
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # cnn_3_1
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # cnn_3_2
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # cnn_4_1
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # cnn_4_2
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # cnn_5_1
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # cnn_5_2
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fccLayers = nn.Sequential(
            # fc_6
            nn.Linear(in_features=4 * 2 * 512, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # fc_7
            nn.Linear(in_features=4096, out_features=1080),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # fc_8
            nn.Linear(in_features=1080, out_features=100),
        )

    def forward_one(self, t):
        # Forward pass
        output = self.cnnLayers(t)
        output = output.reshape(-1, 4 * 2 * 512)
        output = self.fccLayers(output)

        return output

    def forward(self, input_a, input_b):
        # Forward pass for input_a
        output_a = self.forward_one(input_a)

        # Forward pass for input_b
        output_b = self.forward_one(input_b)

        return output_a, output_b


# DECLARATION OF THE LOSS FUNCTION
class NormalizedContrastiveLoss(nn.Module):

    def __init__(self, margin_positive=0.3, margin_negative=0.7):
        super(NormalizedContrastiveLoss, self).__init__()
        self.margin_positive = margin_positive
        self.margin_negative = margin_negative

    def forward(self, output1, output2, label):
        # Euclidean distance of two output feature vectors
        euclidean_distance = F.pairwise_distance(output1, output2)
        euclidean_distance = torch.pow(euclidean_distance, 2)

        # Normalization of the distance
        normalized_distance = 2 * (1 / (1 + torch.exp(-euclidean_distance)) - 0.5)

        # Contrastive Loss Function
        # # perform contrastive loss calculation with the distance
        # loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # Normalized Double-Margin Contrastive Loss Function
        loss = 0.5 * torch.mean(label * torch.clamp(normalized_distance - self.margin_positive, min=0.0) + (1 - label) *
                                torch.clamp(self.margin_negative - normalized_distance, min=0.0))

        return loss, normalized_distance


# NET DECLARATION
net = SiameseCNN()

# LOSS FUNCTION DECLARATION
loss_function = NormalizedContrastiveLoss()

# OPTIMIZER DECLARATION
optimizer = optim.Adagrad(net.parameters(), lr=wandb.config.learningRate)


# TRAINING LOOP
def train(iteration=0, lossHistoryTraining=[], lossHistoryValidation=[], hardAccuracyTraining=[], hardAccuracyValidation=[],
          softAccuracyTraining=[], softAccuracyValidation=[], xAxis=[]):
    global training

    lossIterationHistory = []
    distanceIterationHistory = []
    softPredictionsIterationHistory = []
    hardPredictionsIterationHistory = []

    if iteration != 0:
        iteration += 1

    for ii in range(iteration, wandb.config.numberOfIterations + iteration + 1):
        images_a, images_b, labels = prepare_batch()

        optimizer.zero_grad()  # The gradients get reset

        output_a, output_b = net(images_a, images_b)

        loss, normalizedDistance = loss_function(output_a, output_b, labels)
        loss.backward()

        optimizer.step()

        # The following is just metrics calculations
        lossIterationHistory.append(loss.item())
        distanceIterationHistory = np.concatenate((distanceIterationHistory, normalizedDistance.detach().numpy()))

        labelsBinary = np.ma.getmaskarray(np.ma.masked_greater(labels.detach().numpy(), 0.5))  # Gets the labels as
        # a binary array

        softPredictions = np.ma.getmaskarray(np.ma.masked_less(normalizedDistance.detach().numpy(), 0.5))  # Gets as a
        # binary array the distance (as if the were predictions), where True y less than 0.5
        correctSoftPredictions = np.logical_not(np.logical_xor(softPredictions, labelsBinary))  # Gets as a binary
        # array which predictions matches the labels
        softPredictionsIterationHistory = np.concatenate(
            (softPredictionsIterationHistory, correctSoftPredictions))  # Gets the history predictions

        hardPositivePredictions = np.ma.getmaskarray(np.ma.masked_less(normalizedDistance.detach().numpy(), 0.3))
        correctHardPositivePredictions = np.logical_and(hardPositivePredictions, labelsBinary)

        hardNegativePredictions = np.ma.getmaskarray(np.ma.masked_greater(normalizedDistance.detach().numpy(), 0.7))
        correctHardNegativePredictions = np.logical_and(hardNegativePredictions, np.logical_not(labelsBinary))

        correctHardPredictions = np.logical_or(correctHardPositivePredictions, correctHardNegativePredictions)

        hardPredictionsIterationHistory = np.concatenate((hardPredictionsIterationHistory, correctHardPredictions))

        if ii % wandb.config.showEvery == 0 and training:
            meanLoss = np.mean(lossIterationHistory)
            lossHistoryTraining.append(meanLoss)

            softAccuracyIteration = len(np.argwhere(softPredictionsIterationHistory)) / len(
                softPredictionsIterationHistory)
            hardAccuracyIteration = len(np.argwhere(hardPredictionsIterationHistory)) / len(
                hardPredictionsIterationHistory)

            softAccuracyTraining.append(softAccuracyIteration)
            hardAccuracyTraining.append(hardAccuracyIteration)

            print("TRAINING")
            print("Iteration: " + ii.__str__())
            print("Mean Loss: " + meanLoss.__str__())
            print("Soft Accuracy: " + softAccuracyIteration.__str__())
            print("Hard Accuracy: " + hardAccuracyIteration.__str__())
            print("Number < 0.3: " + len(np.argwhere(
                np.ma.getmaskarray(np.ma.masked_less(distanceIterationHistory, 0.3)))).__str__() + "/" + len(
                distanceIterationHistory).__str__())
            print("Number > 0.7: " + len(np.argwhere(
                np.ma.getmaskarray(np.ma.masked_greater(distanceIterationHistory, 0.7)))).__str__() + "/" + len(
                distanceIterationHistory).__str__())

            print("--- %s seconds ---" % (time.time() - start_time))

            print("")

            xAxis.append(ii)

            training = False

            lossHistoryValidationIteration, softAccuracyValidationIteration, hardAccuracyValidationIteration = test()

            wandb.log({'iteration': ii,
                       'lossTraining': meanLoss,
                       'softAccuracyTraining': softAccuracyIteration,
                       'hardAccuracyTraining': hardAccuracyIteration,
                       'lossValidation': lossHistoryValidationIteration,
                       'softAccuracyValidation': softAccuracyValidationIteration,
                       'hardAccuracyValidation': hardAccuracyValidationIteration}, commit=True)

            # step += 1


            print("VALIDATION")
            print("Iteration: " + ii.__str__())
            print("Mean Loss: " + lossHistoryValidationIteration.__str__())
            print("Soft Accuracy: " + softAccuracyValidationIteration.__str__())
            print("Hard Accuracy: " + hardAccuracyValidationIteration.__str__())

            print("--- %s seconds ---" % (time.time() - start_time))

            print("")

            lossIterationHistory = []
            softPredictionsIterationHistory = []
            hardPredictionsIterationHistory = []
            distanceIterationHistory = []

            lossHistoryValidation.append(lossHistoryValidationIteration)
            softAccuracyValidation.append(softAccuracyValidationIteration)
            hardAccuracyValidation.append(hardAccuracyValidationIteration)

        if ii % wandb.config.saveEvery == 0 or ii == wandb.config.numberOfIterations + iteration + 1 and ii != 0:
            torch.save({'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration': iteration,
                        'lossHistoryTraining': lossHistoryTraining,
                        'lossHistoryValidation': lossHistoryValidation,
                        'hardAccuracyTraining': hardAccuracyTraining,
                        'hardAccuracyValidation': hardAccuracyValidation,
                        'softAccuracyTraining': softAccuracyTraining,
                        'softAccuracyValidation': softAccuracyValidation,
                        'xAxis': xAxis,
                        'lastIteration': ii}, savePath + ii.__str__() + ".pt")
            # wandb.save('model.pt')

    print("--- TOTAL %s seconds ---" % (time.time() - start_time))

    plt.plot(xAxis, lossHistoryTraining, 'r-')
    plt.plot(xAxis, lossHistoryValidation, 'b-')
    plt.show()

    plt.plot(xAxis, softAccuracyTraining, 'r-')
    plt.plot(xAxis, softAccuracyValidation, 'b-')
    plt.show()

    plt.plot(xAxis, hardAccuracyTraining, 'r-')
    plt.plot(xAxis, hardAccuracyValidation, 'b-')
    plt.show()

    return net, xAxis[-1], lossHistoryTraining, lossHistoryValidation, hardAccuracyTraining, hardAccuracyValidation, softAccuracyTraining, softAccuracyValidation, xAxis

# TESTING LOOP
def test():
    global training

    lossIterationHistory = []
    distanceIterationHistory = []
    softPredictionsIterationHistory = []
    hardPredictionsIterationHistory = []

    for ii in range(wandb.config.numberOfIterationsValidation):
        images_a, images_b, labels = prepare_batch()
        # images_a, images_b, labels = images_a.cuda(), images_b.cuda(), labels.cuda()
        output_a, output_b = net(images_a, images_b)

        loss, normalizedDistance = loss_function(output_a, output_b, labels)

        # THE FOLLOWING IS JUST MEASUREMENT VARIABLES
        lossIterationHistory.append(loss.item())
        distanceIterationHistory = np.concatenate((distanceIterationHistory, normalizedDistance.detach().numpy()))

        labelsBinary = np.ma.getmaskarray(np.ma.masked_greater(labels.detach().numpy(), 0.5))  # Gets the labels as
        # a binary array

        softPredictions = np.ma.getmaskarray(np.ma.masked_less(normalizedDistance.detach().numpy(), 0.5))  # Gets as a
        # binary array the distance (as if the were predictions), where True y less than 0.5
        correctSoftPredictions = np.logical_not(np.logical_xor(softPredictions, labelsBinary))  # Gets as a binary
        # array which predictions matches the labels
        softPredictionsIterationHistory = np.concatenate(
            (softPredictionsIterationHistory, correctSoftPredictions))  # Gets the history predictions

        hardPositivePredictions = np.ma.getmaskarray(np.ma.masked_less(normalizedDistance.detach().numpy(), 0.3))
        correctHardPositivePredictions = np.logical_and(hardPositivePredictions, labelsBinary)

        hardNegativePredictions = np.ma.getmaskarray(np.ma.masked_greater(normalizedDistance.detach().numpy(), 0.7))
        correctHardNegativePredictions = np.logical_and(hardNegativePredictions, np.logical_not(labelsBinary))

        correctHardPredictions = np.logical_or(correctHardPositivePredictions, correctHardNegativePredictions)

        hardPredictionsIterationHistory = np.concatenate((hardPredictionsIterationHistory, correctHardPredictions))

    meanLoss = np.mean(lossIterationHistory)

    softAccuracyIteration = len(np.argwhere(softPredictionsIterationHistory)) / len(
        softPredictionsIterationHistory)
    hardAccuracyIteration = len(np.argwhere(hardPredictionsIterationHistory)) / len(
        hardPredictionsIterationHistory)

    training = True

    return meanLoss, softAccuracyIteration, hardAccuracyIteration

# ONCE THE MODEL HAS BEEN TRAINED THIS FUNCTION EXTRACTS THE FEATURES OF ALL DETECTIONS INTO TEXT FILES
def features_extraction():
    mainFileRead = open(generalPath + r'\FEATURES\my_train_samples.txt', 'r')
    readVariable = mainFileRead.read()
    mainFileRead.close()

    readVariable = readVariable.splitlines()
    readVariable = np.stack([np.array(line.split('|')) for line in readVariable], axis=0)

    for path, folderName, imageFrame, identity, bbLeft, bbTop, bbWidth, bbHeight in readVariable:

        image = Image.open(generalPath + r"\SAMPLES\\train\\" + path.replace(".txt", ".png"))
        toTensorTransform = transforms.ToTensor()
        resizeTransform = transforms.Resize((128, 64))

        image = resizeTransform(image)
        image = toTensorTransform(image)
        image = torch.unsqueeze(image, 0)  # Add the forth dimension for the batch size

        features = net.forward_one(image)

        try:
            # Create target Directory
            os.mkdir(generalPath + r'\FEATURES' + chr(92) + folderName)
        except FileExistsError:
            pass
        featuresFile = open(generalPath + r'\FEATURES' + chr(92) + path.replace('/', chr(92)), 'w')

        features = features.detach().numpy()
        features = np.squeeze(features)
        features = [feature.item().__str__() for feature in features]
        features = '|'.join(features)
        featuresFile.write(features)

        featuresFile.close()


if training and loading:
    savedParameters = torch.load(loadPath)
    net.load_state_dict(savedParameters['model_state_dict'])
    optimizer.load_state_dict(savedParameters['optimizer_state_dict'])
    iteration = savedParameters['iteration']
    lossHistoryTraining = savedParameters['lossHistoryTraining']
    lossHistoryValidation = savedParameters['lossHistoryValidation']
    hardAccuracyTraining = savedParameters['hardAccuracyTraining']
    hardAccuracyValidation = savedParameters['hardAccuracyValidation']
    softAccuracyTraining = savedParameters['softAccuracyTraining']
    softAccuracyValidation = savedParameters['softAccuracyValidation']
    xAxis = savedParameters['xAxis']
    iteration = savedParameters['lastIteration']

    net, iteration, lossHistoryTraining, lossHistoryValidation, hardAccuracyTraining, hardAccuracyValidation, softAccuracyTraining, softAccuracyValidation, xAxis = train(
        iteration, lossHistoryTraining, lossHistoryValidation, hardAccuracyTraining, hardAccuracyValidation,
        softAccuracyTraining, softAccuracyValidation, xAxis)

    # torch.save({'model_state_dict': net.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'iteration': iteration,
    #                 'lossHistoryTraining': lossHistoryTraining,
    #                 'lossHistoryValidation': lossHistoryValidation,
    #                 'hardAccuracyTraining': hardAccuracyTraining,
    #                 'hardAccuracyValidation': hardAccuracyValidation,
    #                 'softAccuracyTraining': softAccuracyTraining,
    #                 'softAccuracyValidation': softAccuracyValidation,
    #                 'xAxis': xAxis,
    #                 'lastIteration': iteration}, savePath)

elif training:
    net, iteration, lossHistoryTraining, lossHistoryValidation, hardAccuracyTraining, hardAccuracyValidation, softAccuracyTraining, softAccuracyValidation, xAxis = train()

    # torch.save({'model_state_dict': net.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'iteration': iteration,
    #             'lossHistoryTraining': lossHistoryTraining,
    #             'lossHistoryValidation': lossHistoryValidation,
    #             'hardAccuracyTraining': hardAccuracyTraining,
    #             'hardAccuracyValidation': hardAccuracyValidation,
    #             'softAccuracyTraining': softAccuracyTraining,
    #             'softAccuracyValidation': softAccuracyValidation,
    #             'xAxis': xAxis,
    #             'lastIteration': iteration}, savePath)

else:
    net.load_state_dict(torch.load(loadPath)['model_state_dict'])
    features_extraction()

train_a.close()
train_b.close()
test_a.close()
test_b.close()
