import numpy as np
import os
from sklearn.svm import SVC
import cv2
import joblib

'''Data other than present in Dataset Folder'''
from keras.datasets import mnist

'''Number classification using SVM'''''
class TrainingSVM:

    def __init__(self):
        model = SVC()

        '''Mnist Standard Data'''
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 784)
        y_train = y_train.reshape(X_train.shape[0])

        # X_train=X_train[:5000]
        # y_train=y_train[:5000]
        self.X_test=X_test.reshape(X_test.shape[0],784)
        self.y_test=y_test.reshape(X_test.shape[0])

        '''Extra Training Data in Dataset Folder'''
        TRAIN_PATH = "./Dataset/Train"
        list_folder = os.listdir(TRAIN_PATH)
        for folder in list_folder:
            flist = os.listdir(os.path.join(TRAIN_PATH, folder))
            for f in flist:
                im = cv2.imread(os.path.join(TRAIN_PATH, folder, f))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                # resizing of image
                im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
                im = im.flatten()
                X_train = np.vstack((X_train, im))

        for i in range(0, 10):
            temp = 500 * [i]
            for j in temp:
                y_train = np.append(y_train, j)

        X_train=X_train.astype('float')/255.0
        model.fit(X_train, y_train)

        '''Storing Model in Disk Storage'''
        joblib.dump(model, 'Number_model')

        self._model = model
        print(X_train.shape, y_train.shape)

    def score(self):
        return self._model.score(self._X_test,self._y_test)

if __name__=='__main__':
    model=TrainingSVM()
    print(model.score())

