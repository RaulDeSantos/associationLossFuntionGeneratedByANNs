import numpy as np
import os
import shutil

generalPath = r'C:\Users\rausanto\Documents\TFG\MOT17'

movingFolders = ('MOT17-05-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM', 'MOT17-13-DPM')


file = open(generalPath + chr(92) + r'FEATURES' + chr(92) + movingFolders[0] + r'.txt', 'r')
fullVariable = file.read()
file.close()

file = open(generalPath + chr(92) + r'FEATURES' + chr(92) + movingFolders[1] + r'.txt', 'r')
fullVariable = fullVariable + file.read()
file.close()

file = open(generalPath + chr(92) + r'FEATURES' + chr(92) + movingFolders[2] + r'.txt', 'r')
fullVariable = fullVariable + file.read()
file.close()

file = open(generalPath + chr(92) + r'FEATURES' + chr(92) + movingFolders[3] + r'.txt', 'r')
fullVariable = fullVariable + file.read()
file.close()

fullVariable = fullVariable.splitlines()
fullVariable = np.stack([np.array(line.split('|')) for line in fullVariable], axis=0)

identifications = np.unique(fullVariable[:, 3])      # Gets all the identifications

np.random.shuffle(identifications)       # Shuffles all de identifications

testIdentifications = identifications[0:int(0.05*identifications.size)]     # Gets de first 5% of the identifications as test
valIdentifications = identifications[int(0.05*identifications.size):2*int(0.05*identifications.size)]     # Gets de second 5% of the identifications as validation
trainIdentifications = identifications[2*int(0.05*identifications.size)::]     # Gets de rest of the identifications as training

# Make LSTMTracklet for test
try:
    os.mkdir(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\test')
except FileExistsError as error:
    pass
except Exception as e:
    print(e)

for filename in os.listdir(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\test'):
    file_path = os.path.join(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\test', filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for identification in testIdentifications:
    wholeIDInformation = fullVariable[np.where(fullVariable[:, 3] == identification), :]
    wholeIDInformation = np.squeeze(wholeIDInformation, axis=0)
    idFile = open(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\test' + chr(92) + identification + r'.txt', 'w')
    for ii in range(wholeIDInformation.shape[0]):
        line = list(wholeIDInformation[ii, :])

        idFile.write('|'.join(line) + '\n')

    idFile.close()


# Make LSTMTracklet for validation
try:
    os.mkdir(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\val')
except FileExistsError as error:
    pass
except Exception as e:
    print(e)

for filename in os.listdir(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\val'):
    file_path = os.path.join(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\val', filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for identification in valIdentifications:
    wholeIDInformation = fullVariable[np.where(fullVariable[:, 3] == identification), :]
    wholeIDInformation = np.squeeze(wholeIDInformation, axis=0)
    idFile = open(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\val' + chr(92) + identification + r'.txt', 'w')
    for ii in range(wholeIDInformation.shape[0]):
        line = list(wholeIDInformation[ii, :])

        idFile.write('|'.join(line) + '\n')

    idFile.close()


# Make LSTMTracklet for train
try:
    os.mkdir(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\train')
except FileExistsError as error:
    pass
except Exception as e:
    print(e)

for filename in os.listdir(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\train'):
    file_path = os.path.join(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\train', filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for identification in trainIdentifications:
    wholeIDInformation = fullVariable[np.where(fullVariable[:, 3] == identification), :]
    wholeIDInformation = np.squeeze(wholeIDInformation, axis=0)
    idFile = open(generalPath + r'\DATA\LSTMTRACKLET\movingCamera\train' + chr(92) + identification + r'.txt', 'w')
    for ii in range(wholeIDInformation.shape[0]):
        line = list(wholeIDInformation[ii, :])

        idFile.write('|'.join(line) + '\n')

    idFile.close()
