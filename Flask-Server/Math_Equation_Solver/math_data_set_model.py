import pandas as pd
import numpy as np
import itertools
import keras
from keras.models import Model
from keras.layers import *
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
from math_data_set_csv import MathDataSet

# From deeplizards
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle


class MathDataSetModel:
    def __init__(self, csvFile):
        """
        :param csvFile: csvFile for training the model
        """
        self.__csvFile = csvFile
        self.__loadImageFiles = []
        self.__totalImagesInDataSet = 16762
        self.__totalClasses = 18

    def __dropLabelsColumn(self):
        df_train = pd.read_csv(self.__csvFile, index_col=False)
        # Extract the labels column
        labels = df_train["784"]
        # Drop the column from the df_train
        df_train.drop(df_train.columns[[784]], axis=1, inplace=True)
        return labels, df_train

    def setUpModel(self):
        labels, df_train = self.__dropLabelsColumn()
        # labels is in data frame format we need to convert it into an array
        labels = np.array(labels)
        labels, df_train = shuffle(labels, df_train)
        # Converting labels into categorical data
        categorizedData = to_categorical(labels, num_classes=self.__totalClasses)
        for i in range(self.__totalImagesInDataSet):
            self.__loadImageFiles.append(np.array(df_train[i:i + 1]).reshape((28, 28, 1)))
        return categorizedData

    def createModel(self):
        """ Create model using CNN """
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(self.__totalClasses, activation="softmax"))
        return model

    def saveModel(self, model):
        """
        :param model: Save our model in json format
        :return:
        """
        model.save("model_final.h5")

    def trainModelUsingCNN(self):
        """
        Train the model using Keras
        """
        categorizedData = self.setUpModel()
        # Create Model
        model = self.createModel()
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        train_batches = np.array(self.__loadImageFiles)
        model.fit(x=train_batches, y=categorizedData,
                  validation_split=0.3, shuffle=True,
                  epochs=200, batch_size=10, verbose=2)
        # Save the model for future use
        self.saveModel(model)


if __name__ == '__main__':
    mathDataSetModel = MathDataSetModel("train_final.csv")
    mathDataSetModel.trainModelUsingCNN()
