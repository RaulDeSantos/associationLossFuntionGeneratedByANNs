# This file reads from the file that contains all the identifications, looks if the image exists and changes it's ID
# number so they are all contiguous. Then writes it back in a file called my_train_samples.txt

import os.path
import numpy as np

generalPath = r'C:\Users\rausanto\Documents\TFG\MOT17\\'

mainFileRead = open(generalPath + r'SAMPLES\train\train_samples_list.txt', 'r')
mainFileWrite = open(generalPath + r'FEATURES\my_train_samples.txt', 'w')
secondFileWrite = open("inservible.txt", 'w')

readVariable = mainFileRead.read()
readVariable = readVariable.splitlines()
# readVariable = readVariable.split(',')

writeVariable = []

idCounter = -1
numberOfIdentifications = 0
lastID = 0
lastFile = ""

for ii, line in enumerate(readVariable):
    aux = line.split(',')
    aux = [auxi.strip() for auxi in aux]

    # print(aux[0])
    if secondFileWrite.writable() and aux[1] != lastFile:
        secondFileWrite.close()
        secondFileWrite = open(generalPath + r'FEATURES\\' + aux[1] + r'.txt', 'w')
        lastFile = aux[1]

        fileNum = aux[1][aux[1].find('-')+1:aux[1].find('-')+3]
        gtFileRead = open(r'C:\Users\rausanto\Documents\TFG\MOT17\DATABASE\train\MOT17-' + fileNum + r'-DPM\gt\gt.txt', 'r')
        gtVariable = gtFileRead.read()
        gtVariable = gtVariable.splitlines()
        gtVariable = np.stack([np.array(line.split(',')) for line in gtVariable], axis=0)

    elif aux[1] != lastFile:
        secondFileWrite = open(generalPath + r'FEATURES\\' + aux[1] + r'.txt', 'w')
        lastFile = aux[1]

        fileNum = aux[1][aux[1].find('-') + 1:aux[1].find('-') + 3]
        gtFileRead = open(r'C:\Users\rausanto\Documents\TFG\MOT17\DATABASE\train\MOT17-' + fileNum + r'-DPM\gt\gt.txt','r')
        gtVariable = gtFileRead.read()
        gtVariable = np.stack([np.array(line.split(', ')) for line in gtVariable], axis=0)

    if os.path.isfile(generalPath+r'SAMPLES\train\\'+aux[0]) and aux[3] != lastID:
        # The following code will find the line of the Ground Truth file that corresponds to the detection and extracts
        # the information needed
        gtLineIndex = np.where(np.asarray(np.logical_and(gtVariable[:, 0] == aux[2], gtVariable[:, 1] == aux[3])))
        aux.append(np.squeeze(gtVariable[gtLineIndex, 2]).item())  # Bounding box left
        aux.append(np.squeeze(gtVariable[gtLineIndex, 3]).item())  # Bounding box top
        aux.append(np.squeeze(gtVariable[gtLineIndex, 4]).item())  # Bounding box width
        aux.append(np.squeeze(gtVariable[gtLineIndex, 5]).item())  # Bounding box height

        print(numberOfIdentifications, aux[1])
        numberOfIdentifications = 1
        lastID = aux[3]
        idCounter += 1
        aux[3] = idCounter.__str__()
        aux[0] = aux[0].replace(".png", ".txt")
        aux = '|'.join(aux)

        mainFileWrite.write(aux + '\n')
        secondFileWrite.write(aux + '\n')

    elif os.path.isfile(generalPath+r'SAMPLES\train\\'+aux[0]):
        # The following code will find the line of the Ground Truth file that corresponds to the detection and extracts
        # the information needed
        gtLineIndex = np.where(np.asarray(np.logical_and(gtVariable[:, 0] == aux[2], gtVariable[:, 1] == aux[3])))
        aux.append(np.squeeze(gtVariable[gtLineIndex, 2]).item())  # Bounding box left
        aux.append(np.squeeze(gtVariable[gtLineIndex, 3]).item())  # Bounding box top
        aux.append(np.squeeze(gtVariable[gtLineIndex, 4]).item())  # Bounding box width
        aux.append(np.squeeze(gtVariable[gtLineIndex, 5]).item())  # Bounding box height

        numberOfIdentifications += 1
        aux[3] = idCounter.__str__()
        aux[0] = aux[0].replace(".png", ".txt")
        aux = '|'.join(aux)

        mainFileWrite.write(aux + '\n')
        secondFileWrite.write(aux + '\n')


if secondFileWrite.writable():
    secondFileWrite.close()


mainFileWrite.close()
mainFileRead.close()