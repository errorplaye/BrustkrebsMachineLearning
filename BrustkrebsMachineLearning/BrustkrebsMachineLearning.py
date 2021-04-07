#Datensatz origin ist in Breastcancer dataset gespeichert
#Training datensätze etc sind in separatem ordner

#Multiple line comment ctrl k ctrl c oder ctrl k ctrl u


from tensorflow import keras
import os 
import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

#valPos = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\validate\krank'
#valPos = r'C:\Users\erics\Downloads\Datasets\validate\krank'
#valNeg = r'C:\Users\erics\Downloads\Datasets\validate\gesund'           

#trainingData = []
#trainingLabel = []
#class_names = ['gesund', 'krank']
#for img in os.listdir(valPos):
#    pic = cv2.imread(os.path.join(valPos,img))
#    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
#    pic = cv2.resize(pic,(50,50))
#    trainingData.append([pic])
#    trainingLabel.append([1])

#np.save(r'C:\Users\erics\Downloads\Datasets\features',np.array(trainingData))

##print(trainingData)
##print(trainingLabel)
#print(len(trainingData))

saved = np.load(r'C:\Users\erics\Downloads\Datasets\features.npy', allow_pickle=True)
print(len(saved))
print(saved.shape)

plt.imshow(saved[0].reshape(50,50,3))
plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(saved[i].reshape(50,50,3), cmap=plt.cm.binary)
    #plt.xlabel(class_names[saved[i]])
plt.show()