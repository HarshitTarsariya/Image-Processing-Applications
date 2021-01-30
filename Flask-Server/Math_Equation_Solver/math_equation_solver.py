import cv2
import numpy as np

from .image_utility import EquationSolver
from .math_data_set import MathDataSet
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

class MathEquationSolver:
    def __init__(self, mathEquationImg):
        """
        :param mathEquationImg:
        """
        self.__mathEquationImg = mathEquationImg

    def __getModel(self):
        loaded_model = load_model('./Math_Equation_Solver/model_final.h5')
        return loaded_model

    def __showMathEquationImage(self, title, img):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __isRectangleValid(self, rects):
        # Whether the rectangle is valid or not
        bool_rect = []
        for r in rects:
            l = []
            for rec in rects:
                flag = 0
                if rec != r:
                    # Discard the smaller rectangles and add the maximum rects
                    if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (
                            rec[1] + rec[3] + 10) and rec[1] < (r[1] + r[3] + 10):
                        flag = 1
                    l.append(flag)
                if rec == r:
                    l.append(0)
            bool_rect.append(l)
        return bool_rect

    def __dumpRectangles(self, cnt, rects, bool_rect):
        # Dump the rectangles
        dump_rect = []
        for i in range(0, len(cnt)):
            for j in range(0, len(cnt)):
                if bool_rect[i][j] == 1:
                    area1 = rects[i][2] * rects[i][3]
                    area2 = rects[j][2] * rects[j][3]
                    if area1 == min(area1, area2):
                        dump_rect.append(rects[i])
        print("dump_rect", dump_rect)
        return dump_rect

    def __manageRectangle(self, cnt, thresh):
        # Array of rectangles from the contours
        rects = []
        for c in cnt:
            tl_x, tl_y, width, height = cv2.boundingRect(c)
            print(tl_x, tl_y, width, height)
            rect = [tl_x, tl_y, width, height]
            rects.append(rect)

        bool_rect = self.__isRectangleValid(rects)
        dump_rect = self.__dumpRectangles(cnt, rects, bool_rect)
        # Remove the rectangles of dump_rect from rects
        final_rect = [i for i in rects if i not in dump_rect]

        print("rects", rects)
        print("bool_rect ", bool_rect)
        print("dump_rect ", dump_rect)
        print("final_rect", final_rect)
        return final_rect

    def __getTrainData(self, thresh, final_rect):
        train_data = []
        for r in final_rect:
            x, y, w, h = r[0], r[1], r[2], r[3]
            im_crop = thresh[y:y + h + 10, x:x + w + 10]
            im_resize = cv2.resize(im_crop, (28, 28))
            train_data.append(im_resize)
        return train_data

    def __showDataExtracted(self, elements, final_rect, thresh):
        i = 1
        for r in final_rect:
            x, y, w, h = r[0], r[1], r[2], r[3]
            im_crop = thresh[y:y + h + 10, x:x + w + 10]
            im_resize = cv2.resize(im_crop, (28, 28))
            plt.subplot(3, 4, i), plt.imshow(im_resize, 'gray')
            plt.title(elements[i-1])
            plt.xticks([]), plt.yticks([])
            i += 1
        plt.show()

    def __processImage(self, img):
        if img is None:
            img = cv2.cvtColor(self.__mathEquationImg,cv2.COLOR_BGR2GRAY)
        equString = ""
        if img is not None:
            # self.__showMathEquationImage("Math Equation", img)
            img = cv2.bitwise_not(img)  # Invert the image
            ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
            final_rect = self.__manageRectangle(cnt, thresh)

            train_data = self.__getTrainData(thresh, final_rect)
            loaded_model = self.__getModel()
            mathDataSet = MathDataSet()
            mathDictionary = mathDataSet.getMathDictionary()
            elements = []
            for i in range(len(train_data)):
                train_data[i] = np.array(train_data[i])
                train_data[i] = train_data[i].reshape((1, 28, 28, 1))
                result = np.argmax(loaded_model.predict(train_data[i]), axis=-1)
                value = mathDictionary.get(result[0])
                elements.append(value)
                if value == "times":
                    equString += '*'
                else:
                    equString += value
            # self.__showDataExtracted(elements, final_rect, thresh)
            # print("String : ", equString)
        return equString

    def __preprocessImage(self):
        solver = EquationSolver()
        img = cv2.imread(self.__mathEquationImg)
        if img is not None:
            img = solver.solveEq(img)
            img = cv2.bitwise_not(img)
            return img

    def solveEquation(self):
        # img = self.__preprocessImage()
        equString = self.__processImage(img=None)
        if equString is not None or equString == "":
            value = eval(equString)
            return equString + " = " + str(value)
        else:
            return "Invalid Input"


if __name__ == '__main__':
    # For paint images
    for i in range(0, 9):
        mathEquation = MathEquationSolver(f"assets/images/paint/test{i}.png")
        print(mathEquation.solveEquation())

    # For handwritten blank background images
    # for i in range(1, 17):
    #     mathEquation = MathEquationSolver(f"assets/images/blank/blank{i}.png")
    #     print(mathEquation.solveEquation())