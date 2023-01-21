#Author: Timur Pulatovich Abdualimov
#Date code: 16.01.2023


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError, BinaryCrossentropy

class DatRetClassifier:

    '''
    Class to perform the classification task
    '''

    def __init__(self,
                    epoch=50,
                    optimizer=Adam(learning_rate=0.001),
                    loss=CategoricalCrossentropy(),
                    verbose=1,
                    number_neurons=500,
                    validation_split=0,
                    batch_size=10,
                    shuffle=True,
                    callback=[EarlyStopping(monitor='loss', mode='auto', patience=7, verbose=1),
                                ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)]
                ):
        '''
        Initialize class attributes
        '''
        self.epoch = epoch
        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self.number_neurons = number_neurons
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.callback = callback

    def fit(self, X_train, y_train, normalize=True):

        '''
        The class method for training the model.
        '''

        # one-hot encoding for y_train
        y_train = np.array(y_train)
        y_train_zero_matrix = np.zeros((y_train.size, y_train.max() + 1))
        y_train_zero_matrix[np.arange(y_train.size), y_train] = 1

        # normalize data
        self.normalize = normalize
        if normalize:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X_train)
            self.scaler = scaler
            X_train = scaler.transform(X_train)

        # compile model
        tf.keras.backend.clear_session() # clean session
        input_layer = Input(shape=(X_train.shape[1],)) # input layer
        # first dense
        # we will make a large neural network with 500 neurons if the number of features is less than 500
        if X_train.shape[1] < 500:
            x = Dense(self.number_neurons, activation="elu")(input_layer)
            count_layers = 0 # define the layer counter
            input_layer_count_neurons = self.number_neurons  # number of neurons in the first layer
        else:
            x = Dense(X_train.shape[1], activation="elu")(input_layer) # if there are more than 500 features, we make the first layer with the number of neurons equal to the number of features
            count_layers = 0 # define the layer counter
            input_layer_count_neurons = X_train.shape[1] # number of neurons in the first layer
        # we will reduce the number of neurons in the layer by half until the number of neurons becomes less than the number of predicted parameters
        # determine the number of layers
        cycle_input_layer_count_neurons = input_layer_count_neurons
        while cycle_input_layer_count_neurons > (y_train_zero_matrix.shape[1] * 2):
            cycle_input_layer_count_neurons //= 2
            count_layers += 1 # exact number of layers
        # second and others dense
        # cycle through the number of layers and reduce the number of neurons in the layer by half
        count_neurons = input_layer_count_neurons // 2
        for i in range(0, count_layers):
            x = Dense(count_neurons, activation="elu")(x)
            count_neurons //= 2
        # prediction layer
        predictions = Dense(y_train_zero_matrix.shape[1], activation ='softmax')(x)
        mod = Model(inputs=input_layer, outputs=predictions) # define input and output layer
        mod.compile(optimizer=self.optimizer, loss=self.loss) # compile model
        # fit model
        # start fit model
        mod.fit(X_train, y_train_zero_matrix,
                                validation_split = self.validation_split,
                                epochs=self.epoch,
                                verbose=self.verbose,
                                callbacks=self.callback,
                                steps_per_epoch = X_train.shape[0] // self.batch_size,
                                shuffle=self.shuffle,
                                use_multiprocessing=True)
        self.mod = mod


    def predict(self, X_test):

        '''
        Test sample prediction method
        '''
        # normalize X_test
        if self.normalize:
            X_test = self.scaler.transform(X_test)
        # normalize X_test
        predict_model = self.mod.predict(X_test)
        return np.argmax(predict_model, -1)

    def predict_proba(self, X_test):

        # normalize X_test
        if self.normalize:
            X_test = self.scaler.transform(X_test)
        # normalize X_test
        predict_model = self.mod.predict(X_test)
        return predict_model



class DatRetRegressor:

    '''
    Class to perform the classification task
    '''

    def __init__(self,
                    epoch=50,
                    optimizer=Adam(learning_rate=0.01),
                    loss=MeanSquaredError(),
                    verbose=1,
                    number_neurons=500,
                    validation_split=0,
                    batch_size=10,
                    shuffle=True,
                    callback = [EarlyStopping(monitor='loss', mode='auto', patience=7, verbose=1),
                                ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)]
                ):
        '''
        Initialize class attributes
        '''
        tf.keras.backend.clear_session() # clean session
        self.epoch = epoch
        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self.number_neurons = number_neurons
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.callback = callback


    def fit(self, X_train, y_train, normalize=True):

        '''
        The class method for training the model.
        '''

        # to array
        y_train = np.array(y_train)

        # normalize data
        self.normalize = normalize
        if normalize:
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            self.scaler = scaler
            X_train = scaler.transform(X_train)

        # compile model
        tf.keras.backend.clear_session() # clean session
        input_layer = Input(shape=(X_train.shape[1],)) # input layer
        # first dense
        # we will make a large neural network with 500 neurons if the number of features is less than 500
        if X_train.shape[1] < 500:
            x = Dense(self.number_neurons, activation="elu")(input_layer)
            count_layers = 0 # define the layer counter
            input_layer_count_neurons = self.number_neurons  # number of neurons in the first layer
        else:
            x = Dense(X_train.shape[1], activation="elu")(input_layer) # if there are more than 500 features, we make the first layer with the number of neurons equal to the number of features
            count_layers = 0 # define the layer counter
            input_layer_count_neurons = X_train.shape[1] # number of neurons in the first layer
        # we will reduce the number of neurons in the layer by half until the number of neurons becomes less than the number of predicted parameters
        # determine the number of layers
        cycle_input_layer_count_neurons = input_layer_count_neurons
        while cycle_input_layer_count_neurons > 20:
            cycle_input_layer_count_neurons //= 2
            count_layers += 1 # exact number of layers
        # second and others dense
        # cycle through the number of layers and reduce the number of neurons in the layer by half
        count_neurons = input_layer_count_neurons // 2
        for i in range(0, count_layers):
            x = Dense(count_neurons, activation="elu")(x)
            count_neurons //= 2
        # prediction layer
        predictions = Dense(1)(x)
        mod = Model(inputs=input_layer, outputs=predictions) # define input and output layer
        mod.compile(optimizer=self.optimizer, loss=self.loss) # compile model
        # fit model
        # start fit model
        mod.fit(X_train, y_train,
                                validation_split = self.validation_split,
                                epochs=self.epoch,
                                verbose=self.verbose,
                                callbacks=self.callback,
                                steps_per_epoch = X_train.shape[0] // self.batch_size,
                                shuffle=self.shuffle,
                                use_multiprocessing=True)
        self.mod = mod


    def predict(self, X_test):

        '''
        Test sample prediction method
        '''
        # normalize X_test
        if self.normalize:
            X_test = self.scaler.transform(X_test)
        # normalize X_test
        predict_model = self.mod.predict(X_test)
        return predict_model


class DatRetMultilabelClassifier:

    '''
    Class to perform the classification task
    '''

    def __init__(self,
                    epoch=50,
                    optimizer=Adam(learning_rate=0.01),
                    loss=MeanSquaredError(),
                    verbose=1,
                    number_neurons=500,
                    validation_split=0,
                    batch_size=10,
                    shuffle=True,
                    callback=[EarlyStopping(monitor='loss', mode='auto', patience=7, verbose=1),
                                ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)]

                ):
        '''
        Initialize class attributes
        '''
        tf.keras.backend.clear_session() # clean session
        self.epoch = epoch
        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self.number_neurons = number_neurons
        self.validation_split = validation_split
        self.callback = callback
        self.batch_size = batch_size
        self.shuffle = shuffle

    def fit(self, X_train, y_train, normalize=True):

        '''
        The class method for training the model.
        '''

        # one-hot encoding for y_train
        y_train = np.array(y_train)

        # normalize data
        self.normalize = normalize
        if normalize:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X_train)
            self.scaler = scaler
            X_train = scaler.transform(X_train)

        # compile model
        tf.keras.backend.clear_session() # clean session
        input_layer = Input(shape=(X_train.shape[1],)) # input layer
        # first dense
        # we will make a large neural network with 500 neurons if the number of features is less than 500
        if X_train.shape[1] < 500:
            x = Dense(self.number_neurons, activation="elu")(input_layer)
            count_layers = 0 # define the layer counter
            input_layer_count_neurons = self.number_neurons  # number of neurons in the first layer
        else:
            x = Dense(X_train.shape[1], activation="elu")(input_layer) # if there are more than 500 features, we make the first layer with the number of neurons equal to the number of features
            count_layers = 0 # define the layer counter
            input_layer_count_neurons = X_train.shape[1] # number of neurons in the first layer
        # we will reduce the number of neurons in the layer by half until the number of neurons becomes less than the number of predicted parameters
        # determine the number of layers
        cycle_input_layer_count_neurons = input_layer_count_neurons
        while cycle_input_layer_count_neurons > (y_train.shape[1] * 2):
            cycle_input_layer_count_neurons //= 2
            count_layers += 1 # exact number of layers
        # second and others dense
        # cycle through the number of layers and reduce the number of neurons in the layer by half
        count_neurons = input_layer_count_neurons // 2
        for i in range(0, count_layers):
            x = Dense(count_neurons, activation="elu")(x)
            count_neurons //= 2
        # prediction layer
        predictions = Dense(y_train.shape[1], activation ='sigmoid')(x)
        mod = Model(inputs=input_layer, outputs=predictions) # define input and output layer
        mod.compile(optimizer=self.optimizer, loss=self.loss) # compile model
        # fit model
        # start fit model
        mod.fit(X_train, y_train_zero_matrix,
                                validation_split = self.validation_split,
                                epochs=self.epoch,
                                verbose=self.verbose,
                                callbacks=self.callback,
                                steps_per_epoch = X_train.shape[0] // self.batch_size,
                                shuffle=self.shuffle,
                                use_multiprocessing=True)
        self.mod = mod


    def predict(self, X_test):

        '''
        Test sample prediction method
        '''
        # normalize X_test
        if self.normalize:
            X_test = self.scaler.transform(X_test)
        # normalize X_test
        predict_model = self.mod.predict(X_test)
        return redict_model

