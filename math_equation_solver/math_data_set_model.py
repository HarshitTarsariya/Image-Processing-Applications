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
from math_data_set import MathDataSet

# From deeplizards
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class MathDataSetModel:
    def __init__(self, csvFile):
        """
        :param csvFile: csvFile for training the model
        """
        self.__csvFile = csvFile
        self.__loadImageFiles = []
        self.__totalImagesInDataSet = 13000
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
        model.save("model_final.h5")

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def __getValidationDataset(self, valid_path):
        # valid_path = "F:/FUN_LEARN/AI/OpenCV/MathDataset/valid/"
        mathDataSet = MathDataSet()
        mathClass = list(mathDataSet.getMathDictionary().values())
        # valid_batches = mathDataSet.loadImagesFromFolder(valid_path)
        valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=valid_path, target_size=(28, 28), classes=mathClass, batch_size=10,
                                 shuffle=False)
        return valid_batches

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
        # valid_batches = self.__getValidationDataset()
        train_batches = np.array(self.__loadImageFiles)
        # train_batches = shuffle(train_batches)
        model.fit(x=train_batches, y=categorizedData,
                  # validation_data=valid_batches,
                  validation_split=.2,
                  shuffle=True, epochs=20, batch_size=10, verbose=2)
        # Save the model for future use
        self.saveModel(model)

    def doModelAnalysis(self):
        model = load_model('model_final.h5')
        test_path = 'F:/FUN_LEARN/AI/OpenCV/MathDataset/test/'
        cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', 'times']
        test_batches = self.__getValidationDataset(test_path)
        test_imgs, test_labels = next(test_batches)
        predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
        predictions = np.round(predictions)
        cm = confusion_matrix(y_true=test_labels, y_pred=np.argmax(predictions, axis=-1))
        self.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


if __name__ == '__main__':
    mathDataSetModel = MathDataSetModel("train_final.csv")
    mathDataSetModel.trainModelUsingCNN()
    # mathDataSetModel.doModelAnalysis()
