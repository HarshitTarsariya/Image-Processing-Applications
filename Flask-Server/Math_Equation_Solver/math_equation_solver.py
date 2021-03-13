import cv2
import numpy as np
from tensorflow.keras.models import load_model
from .math_data_set_csv import MathDataSet
from .image_utility import ImageUtility


class MathEquationSolver:
    def __init__(self):
        self.util = ImageUtility()
        self.mathDataSet = MathDataSet()
        self.mathDictionary = self.mathDataSet.getMathDictionary()
        self.mathModel = load_model('./Math_Equation_Solver/model_final.h5')

    def getModelPrediction(self, mathModel, data_img):
        data_img = cv2.resize(data_img, (28, 28))
        data_img = np.array(data_img).reshape((1, 28, 28, 1))
        predictor = mathModel.predict(data_img)
        result = np.argmax(predictor)
        value = self.mathDictionary.get(result)
        return value

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
            if BRY - TLY <= 30:
                mrgn = 5
                TLY += mrgn if TLY + mrgn < 0 else mrgn
                BRY += H-BRY-1 if BRY + mrgn > H else mrgn
                # print('After changing: TLY = ', TLY, ' BRY = ', BRY)
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

    def cutSymbols(self, gray, arr, margin=0):
        symbols = []
        for TLX, TLY, BRX, BRY in arr:
            symbols.append(gray[TLY:BRY+margin,TLX:BRX+margin])
        return symbols

    def extractSymbols(self, img):
        binary, gray = self.preprocess(img)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 4)), iterations=2)
        rects = self.getRectangles(binary)
        rects = self.addMargin(binary.shape, rects, margin=5)
        symbols = self.cutSymbols(gray, rects)
        return symbols

    def modifyEquationString(self, equString):
        """
        :param equString:
        :return: Modified equation string
        """
        sqrtStr = "sqrt"
        # Obtain starting index of "sqrt" in the equString
        sqrt_id = equString.find(sqrtStr)
        id = 0
        operators = ['+', '-', '*', '/']
        # Run the loop for all the appearances of "sqrt"
        while sqrt_id != -1:
            sqrt_end_id = sqrt_id + len(sqrtStr)
            symbol = equString[sqrt_end_id]
            # Check for opening parenthesis
            # If it contains opening parenthesis, then don't modify
            # We are assuming that the equation string will be correctly written
            if symbol != "(":
                # Add opening bracket next to "sqrt" in equString
                equString = equString[:sqrt_end_id] + "(" + equString[sqrt_end_id:]
                # symbol and sqrt_end_id is now pointing to content inside sqrt and not '('
                sqrt_end_id += 1
                for id in range(sqrt_end_id, len(equString)):
                    symbol = equString[id]
                    if symbol in operators:
                        # Add closing bracket if operator is encountered
                        equString = equString[:id] + ")" + equString[id:]
                        break
                # Add closing bracket if end of string is encountered
                if id == len(equString) - 1:
                    equString = equString[:id+1] + ")"
            # Obtain the next index of "sqrt" in the equString
            sqrt_id = equString.find(sqrtStr, sqrt_id + len(sqrtStr))
        return equString

    def getEquationStringFrom(self, symbols):
        equString = ""
        for i, img in enumerate(symbols, start=1):
            value = self.getModelPrediction(self.mathModel, img)
            # For saving the box image
            # self.util.saveBoxImage(img, value)
            if value == "times":
                equString += '*'
            elif value == "div":
                equString += "/"
            elif value == "dot":
                equString += "."
            else:
                equString += value
        if "sqrt" in equString:
            equString = self.modifyEquationString(equString)
        return str(equString)

    def solveEquation(self, img):
        """
        :param img:
        :return: answer of the given mathematical equation image
        """
        symbols = self.extractSymbols(img)
        equString = self.getEquationStringFrom(symbols)
        try:
            ans = eval(equString, self.mathDataSet.getMathOperatorDiary())
            ans=equString+' = '+str(ans)
            # self.util.displayImages(symbols)
        except ZeroDivisionError as divErr:
            ans = "You cannot perform division by zero!"
            print("Can't divide by zero : ", divErr)
        except Exception as e:
            ans = "Error :("
            print("Exception Caught : " , e)
        return ans


if __name__ == '__main__':
    solver = MathEquationSolver()

    # Paint Images
    for i in range(0, 19):
        solver.solveEquation(f'assets/images/paint/test{i}.png')

    # Handwritten Images
    # for i in range(1, 43):
    #     solver.solveEquation(f'assets/images/blank/blank{i}.png')

    # New Handwritten Images
    # for i in range(43, 64):
    #     solver.solveEquation(f'assets/images/blank/blank{i}.png')

    # Janak's handwritten images
    # for i in range(17, 20):
    #     solver.solveEquation(f'assets/images/janak/blank{i}.jpeg')
