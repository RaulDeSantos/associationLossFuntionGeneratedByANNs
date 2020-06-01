import os

firstNumber = 0

try:
    for ii in range(10):
        print("Generando datos: " + (ii+firstNumber).__str__())
        os.system("python generateMovingLSTMTracklets.py")
        print("Calculando la red: " + (ii+firstNumber).__str__())
        os.system("python movingLSTM.py " + (ii+firstNumber).__str__())
except Exception as e:
    print(e)