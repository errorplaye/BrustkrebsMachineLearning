#Datensatz origin ist in Breastcancer dataset gespeichert
#Training datensätze etc sind in separatem ordner

#Multiple line comment ctrl k ctrl c oder ctrl k ctrl u


from tensorflow import keras
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

#valPos = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\validate\krank'
valPos = r'C:\Users\erics\Downloads\Datasets\validate\krank'
valNeg = r'C:\Users\erics\Downloads\Datasets\validate\gesund'

testPos = r'C:\Users\erics\Downloads\Datasets\test\krank'
testNeg = r'C:\Users\erics\Downloads\Datasets\test\gesund' 

trainPos = r'C:\Users\erics\Downloads\Datasets\train\krank'
trainNeg = r'C:\Users\erics\Downloads\Datasets\train\gesund'
 
######################################################################
#.  
#.  Für anderes Datenset verändere src vairable 2 mal und Speicherort !!
#.
#.
#.
######################################################################
#trainingData = []
#testData = []
#valData = []

#trainingDataP = []
#testDataP = []
#valDataP = []

#trainingDataN = []
#testDataN = []
#valDataN = []

Data = []
DataP = []
DataN = []

class_names = ['gesund', 'krank']

#i = 0
#src = trainPos
#for img in os.listdir(src):
#    pic = cv2.imread(os.path.join(src,img))
#    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
#    pic = cv2.resize(pic,(50,50))
#    DataP.append([pic])
#    i += 1
#    print(i)
###zwischenspeichern 
##np.save(r'C:\Users\erics\Downloads\Datasets\trainDataP',np.array(DataP))

i = 0
print("done with first")
src = trainNeg
for img in os.listdir(src):
    pic = cv2.imread(os.path.join(src,img))
    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
    pic = cv2.resize(pic,(50,50))
    DataN.append([pic])
    i += 1
    print(i)
 
#zwischenspeichern 
np.save(r'C:\Users\erics\Downloads\Datasets\trainDataN',np.array(DataN))
print("done with second")

##random combination
#m = len(DataN)
#n = len(DataP)

#print(m)
#print(n)
#permutation = np.array([0]*m + [1]*n)
#np.random.shuffle(permutation)

#n = 0;
#m = 0;
#for i in permutation:
#    print(m)
#    if(i == 0):
#        Data.append(DataN[m])
#        m += 1
#    else:
#        Data.append(DataP[n])
#        n += 1

#np.save(r'C:\Users\erics\Downloads\Datasets\trainData',np.array(Data))
#np.save(r'C:\Users\erics\Downloads\Datasets\trainLabel',np.array(permutation))

#print(len(Data))
#print(len(permutation))

