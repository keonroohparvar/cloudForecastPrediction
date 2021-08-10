import os
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

def partitionData():
   trainingFile = "2020-5-19.xlsx"
   excelPath = os.path.abspath(os.getcwd()) + "/trainingData/" + trainingFile
   dataFrame = pd.read_excel(excelPath)

   # Again partitioning the data in the following ways:
   #   70% Training data 
   #   20% Validation Data
   #   10% Test Data

   n = len(dataFrame.columns)
   trainDataFrame = dataFrame[0 : int(n * 0.7)]
   valDataFrame = dataFrame[int(n * 0.7) : int(n * 0.9)]
   testDataFrame = dataFrame[int(n * 0.9) : n]

   # Normalize the Data
   trainDataFrame /= 100
   valDataFrame /= 100
   testDataFrame /= 100

def createMLP(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))
	# return our model
	return model

def createCNN(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

def createModel()
   # create the MLP and CNN models
   mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
   cnn = models.create_cnn(64, 64, 3, regress=False)
   # create the input to our final set of layers as the *output* of both
   # the MLP and CNN
   combinedInput = concatenate([mlp.output, cnn.output])
   # our final FC layer head will have two dense layers, the final one
   # being our regression head
   x = Dense(4, activation="relu")(combinedInput)
   x = Dense(1, activation="linear")(x)
   # our final model will accept categorical/numerical data on the MLP
   # input and images on the CNN input, outputting a single value (the
   # predicted price of the house)
   model = Model(inputs=[mlp.input, cnn.input], outputs=x)


def main():
   partitionData()

if __name__ == "__main__":
   main()