# This is the file that contains the neural network. Much of the information is taken from:
#   https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb

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
from tensorflow.python.keras.layers.recurrent import RNN


# Updating matplotlib to the correct figure sizes and properly label grids
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False



# Getting training data path
trainingFile = "2020-5-19.xlsx"
excelPath = os.path.abspath(os.getcwd()) + "/trainingData/" + trainingFile

# Read entire excel file as pandas dataframe
df = pd.read_excel(excelPath)


# Partitioning the data in the following ways:
    #   70% Training data 
    #   20% Validation Data
    #   10% Test Data
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
# print(f"num_features: {num_features}")
# print(f"shape: {df.shape}")
# exit()


# Normalize the Data
train_df = train_df / 100
val_df = val_df / 100
test_df = test_df / 100



# Function to format the input data for the model
def formatData():
    # Allow function to access the global variables 
    global df, n, train_df, val_df, test_df, wide_window
    

    n = len(df)
    print(df)
    print("----\n")
    train_df = df[:int(n * 0.6)]
    print(train_df.columns)
    
    val_df = df[int(n*0.6):int(n*0.9)]
    test_df = df[int(n*0.9):]
    print(train_df.shape)
    
    print("\nDONE W FORMATTING...\n\n")



# Class that creates windowed data
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df=train_df, val_df=val_df, test_df=test_df,
                label_columns=None):

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                                enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        print(f"label start: {self.label_start}")
        self.labels_slice = slice(self.label_start, None)
        print(f"label slice: {self.labels_slice}")
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        print(f"label indices: {self.label_indices}")

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        print(f"labels: {labels}")
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        # print("Making Dataset")
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=25,)

        # numElem = 0
        # for element in ds.as_numpy_iterator():
        #     if numElem == 0:
        #        print(element)
        #     numElem += 1
        
        # print(numElem)

        ds = ds.map(self.split_window)

        return ds
    

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            pass

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

        plt.xlabel('Time [h]')
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result




MAX_EPOCHS = 30

def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])


    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[])
    return history



def trainModel():
    rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(rnn_model, w2)

    # val_performance['Linear'] = linear.evaluate(single_step_window.val)
    # performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

    lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
    ])



if __name__ == "__main__":
    # formatData()
    w2 = WindowGenerator(
        input_width=4, label_width=1, shift=1,
        label_columns=['Ring 4 SW'])

    # example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
    #                        np.array(train_df[100:100+w2.total_window_size]),
    #                        np.array(train_df[200:200+w2.total_window_size])])


    # example_inputs, example_labels = w2.split_window(example_window)

    # print('All shapes are: (batch, time, features)')
    # print(f'Window shape: {example_window.shape}')
    # print(f'Inputs shape: {example_inputs.shape}')
    # print(f'labels shape: {example_labels.shape}')
    
    
    
    trainModel()
    exit()


