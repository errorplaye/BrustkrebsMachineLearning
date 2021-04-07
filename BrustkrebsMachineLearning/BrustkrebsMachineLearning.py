#Datensatz origin ist in Breastcancer dataset gespeichert
#Training datensätze etc sind in separatem ordner


from tensorflow import keras
import os 
import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

valPos = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\validate\krank'
valNeg = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\validate\gesund'

trainingData = []
trainingLabel = []
for img in os.listdir(valPos):
    pic = cv2.imread(os.path.join(path,img))
    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
    #pic = cv2.resize(pic,(80,80))
    trainingData.append([pic])
    trainingLabel.append([1])

print(trainingData)
print(trainingLabel)