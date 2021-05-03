# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as ply
import cv2
import os
from os import listdir
from os.path import isfile, join
from PIL import Image


class MathDataSet:
    def getMathDictionary(self):
        """
        :return: Returns the dictionary of data set elements containing digits and operators
        """
        mathDictionary = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
                          5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
                          10: "-", 11: "+", 12: "times"}
        return mathDictionary

    def loadImagesFromFolder(self, folder):
        """
        # load images from data set folder
        :param folder
        """
        train_data = []
        # Iterate over all the files in the folder
        for filename in listdir(folder):
            # Read the image and convert it into GrayScale
            img = cv2.imread(join(folder, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.bitwise_not(img)  # For inverting the image
            if img is not None:
                # Threshold the image
                ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                # Find the contours from the threshold image
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # Sort the contours based on the key of rectangles top left corner x dimension
                cnt = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
                # Find a rectangle of maximum width and height from the cnt list
                maximumArea = 0
                for c in cnt:
                    tl_x, tl_y, width, height = cv2.boundingRect(c)
                    maximumArea = max(width * height, maximumArea)
                    if maximumArea == width * height:
                        x_max = tl_x
                        y_max = tl_y
                        h_max = height
                        w_max = width
                im_crop = thresh[y_max:y_max + h_max + 10, x_max:x_max + w_max + 10]  # 10 is added extra
                im_resize = cv2.resize(im_crop, (28, 28))  # Resize rectangle to 28X28
                im_resize = np.reshape(im_resize, (784, 1))  # Reshape rectangle to 784X1

                train_data.append(im_resize)  # Add the rectangle to train_data array
        return train_data

    def storeDataSetIntoCsv(self, dataSetPath):
        """
        :param dataSetPath:
        """
        mathDictionary = self.getMathDictionary()

        # Not used within loop so as to initialize data array
        data = self.loadImagesFromFolder(dataSetPath + "0")
        for i in range(0, len(data)):
            data[i] = np.append(data[i], ["0"])
        # print(len(data))
        for key in range(1, len(mathDictionary)):
            data_aux = self.loadImagesFromFolder(dataSetPath + mathDictionary.get(key))
            for i in range(0, len(data_aux)):
                data_aux[i] = np.append(data_aux[i], [key])
            data = np.concatenate((data, data_aux))
            # print(len(data))
            data_aux.clear()
        # Store our data set into csv file
        df = pd.DataFrame(data, index=None)
        df.to_csv('train_final.csv', index=False)


if __name__ == '__main__':
    # Call storeDataSetIntoCsv() method from MathDataSet class
    mathDataSet = MathDataSet()
    mathDataSet.storeDataSetIntoCsv("F:/FUN_LEARN/AI/OpenCV/MathDataset/train/")
    # mathDataSet.getDataSet()