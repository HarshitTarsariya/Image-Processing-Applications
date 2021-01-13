import pandas as pd
import numpy as np

#For training dataset
import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json


class MathDataSetModel:
    def __init__(self, csvFile):
        """
        :param csvFile: csvFile for training the model
        """
        self.__csvFile = csvFile
        self.__loadImageFiles = []
        self.__totalImagesInDataSet = 1300
        self.__totalClasses = 13

    def __dropLabelsColumn(self):
        df_train = pd.read_csv(self.__csvFile, index_col=False)
        # Extract the labels column
        labels = df_train["784"]
        # Drop the column from the df_train
        df_train.drop(df_train.columns[[784]], axis=1, inplace=True)
        return labels, df_train

    def setUpModel(self):
        labels, df_train = self.__dropLabelsColumn()
        np.random.seed(1212)
        # labels is in data frame format we need to convert it into an array
        labels = np.array(labels)
        # Converting labels into categorical data
        categorizedData = to_categorical(labels, num_classes=self.__totalClasses)
        for i in range(self.__totalImagesInDataSet):
            self.__loadImageFiles.append(np.array(df_train[i:i + 1]).reshape((28, 28, 1)))
        return categorizedData

    def createModel(self):
        """ Create model using CNN """
        np.random.seed(7)
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(15, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(13, activation="softmax"))
        return model

    def saveModel(self, model):
        """
        :param model: Save our model in json format
        :return:
        """
        model_json = model.to_json()
        with open("model_final.json", "w") as json_file:
            json_file.write(model_json)
        # Serialize weights to HDF5
        model.save_weights("model_final.h5")

    def trainModelUsingCNN(self):
        """
        Train the model using Keras
        """
        categorizedData = self.setUpModel()
        # Create Model
        model = self.createModel()
        # Compile the model
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(np.array(self.__loadImageFiles), categorizedData, shuffle=True, epochs=10, batch_size=20)
        # Save the model for future use
        self.saveModel(model)

if __name__ == '__main__':
    mathDataSetModel = MathDataSetModel("train_final.csv")
    mathDataSetModel.trainModelUsingCNN()