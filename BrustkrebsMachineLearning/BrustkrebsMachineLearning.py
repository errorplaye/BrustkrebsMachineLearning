#Datensatz origin ist in Breastcancer dataset gespeichert
#Training datensätze etc sind in separatem ordner

#Multiple line comment ctrl k ctrl c oder ctrl k ctrl u

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os 
import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

#valPos = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\validate\krank'
#valPos = r'C:\Users\erics\Downloads\Datasets\validate\krank'
#valNeg = r'C:\Users\erics\Downloads\Datasets\validate\gesund'           

class_names = ['gesund', 'krank']

trainData = np.load(r'C:\Users\erics\Downloads\Datasets\trainData.npy', allow_pickle=True)
trainLabel = np.load(r'C:\Users\erics\Downloads\Datasets\trainLabel.npy', allow_pickle=True)

testData = np.load(r'C:\Users\erics\Downloads\Datasets\testData.npy', allow_pickle=True)
testLabel = np.load(r'C:\Users\erics\Downloads\Datasets\testLabel.npy', allow_pickle=True)

#nur ein Teil nehmen
trainDataXS = trainData[:20000]
trainLabelXS = trainLabel[:20000]

testDataXS = trainData[:2000]
testLabelXS = trainLabel[:2000]

print("rescaled:")
print(len(trainDataXS))
print(len(testDataXS))
print(len(trainLabelXS))
print(len(testLabelXS))

print(trainDataXS.shape)
plt.imshow(trainDataXS[0].reshape(50,50,3))
plt.show()

trainDataXS = trainDataXS[:,0,:,:,:]
testDataXS = testDataXS[:,0,:,:,:]

print(trainDataXS.shape)
plt.imshow(trainDataXS[0].reshape(50,50,3))
plt.show()

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(saved[i].reshape(50,50,3), cmap=plt.cm.binary)
#    plt.xlabel(class_names[label[i]])
#plt.show()


num_classes = 2

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(50, 50, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(trainDataXS, trainLabelXS, epochs=20)

test_loss, test_acc = model.evaluate(testDataXS,  testLabelXS, verbose=2)

print('\nTest accuracy:', test_acc)

model.save(r'C:\Users\erics\Downloads\Datasets\Models')
