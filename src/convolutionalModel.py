import os
import datetime
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets, Input
from tensorflow.python.keras import activations
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam


def loadImageData():
   """ Returns normalized image data from a given day as a 4-D numpy array """
   MAX_PIXEL_VALUE = 255.0
   dirName = os.path.dirname(__file__)
   imageData = []
   imageNames = os.listdir(os.path.join(dirName, "../flatImages/5-19"))
   imageNames = sorted(imageNames, key=lambda imageName : int(imageName[3 : len(imageName) - 13]))
   # print(imageNames)
   for imageName in imageNames:
      image = cv2.imread(os.path.join(dirName, "../flatImages/5-19/" + imageName), cv2.IMREAD_COLOR)
      # Percent by which the image is resized
      scaleFactor = 6
      width = int(image.shape[1] * scaleFactor / 100)
      height = int(image.shape[0] * scaleFactor / 100)
      resizedImage = cv2.resize(image, (width, height))
      imageData.append(resizedImage)
   # Normalize pixel values to be in the range [0, 1] after converting to numpy array
   return np.array(imageData) / MAX_PIXEL_VALUE


def loadWeatherData():
   """ Returns normalized cloud coverage and wind data from a given day as 
       a Pandas Data Frame """
   MAX_CLOUD_COVER = 100.0
   # For now, using the highest wind speed recorded to normalize wind speed
   MAX_WIND_SPEED = 231.0
   dirName = os.path.dirname(__file__)
   excelPath = os.path.join(dirName, "../trainingData/2020-5-19-Wind.xlsx")
   dataFrame = pd.read_excel(excelPath)
   # For now, simply ignoring wind direction (weighting will come later)
   del dataFrame["Wind Direction"]
   dataFrame["Wind Speed"] /= MAX_WIND_SPEED
   for columnName in dataFrame.columns:
      if "Ring" in columnName:
         dataFrame[columnName] /= 100
   # columnIndices = {name: i for i, name in enumerate(dataFrame.columns)}
   # print(columnIndices)

   return dataFrame


def partitionData(timeAhead):
   """ Returns a dictionary containing numerical and image inputs (X) and outputs (Y) 
       for the training, testing, and validation dataset according to a 70% 20% 10%
       distribution respectively """
   IMAGE_INTERVAL = 5
   imageOffset = timeAhead // IMAGE_INTERVAL
   imageData = loadImageData()
   weatherData = loadWeatherData()

   print(imageData.shape)

   n = min(weatherData.shape[0], imageData.shape[0])
   imagesX = imageData[ : n - imageOffset]
   weatherX = weatherData.iloc[ : n - imageOffset]
   Y = weatherData.iloc[imageOffset : n]
   # for (column in Y.columns):
   #    if ("4" not in column):
   #       Y.pop(column)
   Y.pop("Wind Speed")

   # Now n - imageOffset data points after above correlation
   trainEnd = int(0.7 * (n - imageOffset))

   trainImagesX = imagesX[ : trainEnd]
   trainWeatherX = weatherX.iloc[ : trainEnd]
   trainY = Y.iloc[ : trainEnd]

   testImagesX = imagesX[trainEnd : ]
   testWeatherX = weatherX.iloc[trainEnd : ]
   testY = Y.iloc[trainEnd : ]

   return {
      "trainImagesX" : trainImagesX,
      "trainWeatherX" : trainWeatherX,
      "trainY" : trainY,
      "testImagesX" : testImagesX,
      "testWeatherX" : testWeatherX,
      "testY" : testY
   }


def createMLP(dimension, regress=False):
   # define our MLP network
   model = models.Sequential()
   model.add(layers.Dense(32, input_dim=dimension, activation="relu"))
   model.add(layers.Dense(64, activation="relu"))
   # check to see if the regression node should be added
   if regress:
      model.add(layers.Dense(1, activation="linear"))

   # model = models.Sequential()
   # model.add(layers.Embedding(input_dim=dimension, output_dim=64))
   # model.add(layers.GRU(256, return_sequences=True))
   # model.add(layers.SimpleRNN(128, input_shape=(dimension, 1), return_sequences=True))
   # model.add(layers.Dense(4, activation="relu"))

   model.summary()
   return model


# def createCNN(inputShape, filters=(64, 64, 192), regress=False):
#    # initialize the input shape and channel dimension, assuming
#    # TensorFlow/channels-last ordering
#    chanDim = -1
#    inputs = Input(shape=inputShape)
#    for (i, f) in enumerate(filters):
#       # if this is the first CONV layer then set the input
#       # appropriately
#       if i == 0:
#          x = inputs
#       # CONV => RELU => BN => POOL
#       x = layers.Conv2D(f, (3, 3), padding="same")(x)
#       x = layers.Activation("relu")(x)
#       x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#       x = layers.BatchNormalization(axis=chanDim)(x)

#    # flatten the volume, then FC => RELU => BN => DROPOUT
#    x = layers.Flatten()(x)
#    x = layers.Dense(16)(x)
#    x = layers.Activation("relu")(x)
#    # x = layers.BatchNormalization(axis=chanDim)(x)
#    x = layers.Dropout(0.5)(x)
#    # apply another FC layer, this one to match the number of nodes
#    # coming out of the MLP
#    x = layers.Dense(4)(x)
#    x = layers.Activation("relu")(x)
#    # check to see if the regression node should be added
#    if regress:
#       x = layers.Dense(1, activation="linear")(x)
#    model = models.Model(inputs, x)

#    model.summary()
#    return model


def createCNN(inputShape):
   chanDim = -1
   inputs = Input(shape=inputShape)
   x = layers.Conv2D(64, (11, 11), strides=(4, 4), padding="same")(inputs)
   x = layers.Activation("relu")(x)
   x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
   x = layers.Conv2D(192, (5, 5), strides=(1, 1), padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
   x = layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
   x = layers.Flatten()(x)
   x = layers.Dense(6, activation="relu")(x)
   x = layers.Dropout(0.8)(x)
   x = layers.Dense(7, activation="relu")(x)
   x = layers.Dropout(0.8)(x)
   x = layers.Dense(8, activation="relu")(x)
   x = layers.Dropout(0.8)(x)
   model = models.Model(inputs, x)

   model.summary()
   return model


def main():
   TIME_AHEAD = 5
   modelData = partitionData(TIME_AHEAD)
   trainWeatherX = modelData["trainWeatherX"]
   trainImagesX = modelData["trainImagesX"]
   trainY = modelData["trainY"]
   testWeatherX = modelData["testWeatherX"]
   testImagesX = modelData["testImagesX"]
   testY = modelData["testY"]
   mlp = createMLP(trainWeatherX.shape[1], regress=False)
   cnn = createCNN(trainImagesX[0].shape)
   combinedInput = concatenate([mlp.output, cnn.output])
   # our final FC layer head will have two dense layers, the final one
   # being our regression head
   x = layers.Dense(4, activation="relu")(combinedInput)
   x = layers.Dense(16, activation="sigmoid")(x)
   model = models.Model(inputs=[mlp.input, cnn.input], outputs=x)
   opt = Adam(lr=1e-3, decay=1e-3 / 200)
   model.compile(loss=tf.losses.MeanSquaredError(), 
         optimizer=opt, 
         metrics=[tf.keras.metrics.MeanAbsoluteError()])
   # train the model
   print("[INFO] training model...")
   model.fit(
      x=[trainWeatherX, trainImagesX], y=trainY,
      validation_data=([testWeatherX, testImagesX], testY),
      epochs=30, batch_size=25)

# def main():
#    TIME_AHEAD = 5
#    modelData = partitionData(TIME_AHEAD)
#    trainImagesX = modelData["trainImagesX"]
#    trainY = modelData["trainY"]
#    testImagesX = modelData["testImagesX"]
#    testY = modelData["testY"]
#    model = createCNN(trainImagesX[0].shape, regress=False)
#    opt = Adam(lr=1e-3, decay=1e-3 / 200)
#    model.compile(loss=tf.losses.MeanSquaredError(), 
#          optimizer=opt,
#          metrics=[tf.metrics.MeanAbsoluteError()])
#    print("[INFO] training model...")
#    model.fit(
#       x=trainImagesX, y=trainY,
#       validation_data=(testImagesX, testY),
#       epochs=30, batch_size=25)


if __name__ == "__main__":
   main()