import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
from math_data_set import MathDataSet
from tensorflow.keras.models import load_model


class MathEqSolver:
    def getModel(self): 
        loaded_model = load_model('model_final.h5')
        return loaded_model

    def getModelPrediction(self, model, data_img):
        mathDataSet = MathDataSet()
        mathDictionary = mathDataSet.getMathDictionary()
        data_img = cv2.resize(data_img, (28, 28))
        data_img = np.array(data_img).reshape((1, 28, 28, 1))
        predictor = model.predict(data_img)
        result = np.argmax(predictor)
        value = mathDictionary.get(result)
        return value

class Utility:
    def resizeImg(self, img, w, h):
        im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        im_pil.thumbnail((w, h))
        img = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        return img

    def displayImgCV(self, imgArr=[]):
        if imgArr:
            for img in imgArr:
                cv2.imshow(img[0], img[1])
            cv2.waitKey(0)

    def displayImages(self, imags, mathEqObj, model,type='gray'):
        plt.figure(figsize=(10, 10), dpi=50, facecolor='w', edgecolor='k')
        cols = 3
        rows = math.ceil(len(imags) / cols)
        equString = ""
        for i, img in enumerate(imags, start=1):
            plt.subplot(rows, cols, i)
            value = mathEqObj.getModelPrediction(model, img)
            if value == "times":
                equString += '*'
            else:
                equString += value
            if type == 'gray':
                plt.imshow(img, type)
            else:
                plt.imshow(img)
            # For saving the tiny images
            # filename = value + str(i) + '.jpg'
            # cv2.imwrite(filename, cv2.bitwise_not(img))
            plt.xticks([]), plt.yticks([])
        print(equString, ' = ', eval(equString))
        plt.show()


class VanillaEqSolver:
    def __init__(self):
        self.util = Utility()
        self.mathEqObj = MathEqSolver()
        self.model = self.mathEqObj.getModel()

    def preprocess(self, img):
        img = self.util.resizeImg(img, w=1000, h=500)
        # color = cv2.GaussianBlur(color, (5, 5), 0)
        kernel = [
            [0.3, -0.1, 0],
            [-0.5, 4.5, -0.6],
            [0, -2, 0],
        ]
        color = cv2.bilateralFilter(img, 9, 75, 75)
        color = cv2.filter2D(color, -1, np.array(kernel))
        gray = ~cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # self.util.displayImgCV([["bin",binary],["gray",gray],["color",img]])
        return binary, gray

    def addMargin(self, shape, arr, margin=0):
        arr1 = []
        for x, y, w, h in arr:
            H, W = shape
            TLX = x if x - margin < 0 else x - margin
            TLY = y if y - margin < 0 else y - margin
            BRX = x + w if x + w + margin > W else x + w + margin
            BRY = y + h if y + h + margin > H else y + h + margin
            # print('TLY = ', TLY, ' BRY = ', BRY)
            if BRY - TLY <= 25:
                TLY -= 35
                BRY += 35
            arr1.append((TLX, TLY, BRX, BRY))
        return arr1

    def getRectangles(self, binary):
        contour, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for cnt in contour:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 or h > 10:
                rects.append((x, y, w, h))
        rects = sorted(rects)

        return rects

    def cutSymbols(self, gray, arr):
        symbols = []
        for TLX, TLY, BRX, BRY in arr:
            symbols.append(gray[TLY:BRY,TLX:BRX])
        return symbols

    def extractSymbols(self, img):
        binary, gray = self.preprocess(img)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 4)), iterations=2)
        rects = self.getRectangles(binary)
        rects = self.addMargin(binary.shape, rects, margin=5)
        symbols = self.cutSymbols(gray, rects)
        self.util.displayImages(symbols, self.mathEqObj, self.model)


if __name__ == '__main__':
    solver = VanillaEqSolver()
    solver.extractSymbols(cv2.imread(f'assets/images/paint/min.jpg'))
    # Paint Image
    # for i in range(0,9):
    #     solver.extractSymbols(cv2.imread(f'assets/images/paint/test{i}.png'))

    # Blank Image
    # for i in range(1,17):
    #     solver.extractSymbols(cv2.imread(f'assets/images/blank/blank{i}.png'))
