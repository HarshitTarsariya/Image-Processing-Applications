import cv2
import numpy as np
from numpy import bool_

from math_data_set import MathDataSet
from keras.models import model_from_json, load_model

class MathEquationSolver:
    def __init__(self, mathEquationImg):
        """
        :param mathEquationImg:
        """
        self.__mathEquationImg = mathEquationImg

    def getModel(self):
        # Get the model from the json file
        json_file = open('model_final.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # Load weights into new model
        loaded_model.load_weights('model_final.h5')
        print("Model loaded from disk")
        loaded_model.save('model_final.hdf5')
        loaded_model = load_model('model_final.hdf5')
        return loaded_model

    def showMathEquationImage(self, img):
        cv2.imshow("Math Equation", img)
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

    def getTrainData(self, thresh, final_rect):
        train_data = []
        for r in final_rect:
            x = r[0]
            y = r[1]
            w = r[2]
            h = r[3]
            im_crop = thresh[y:y + h + 10, x:x + w + 10]
            im_resize = cv2.resize(im_crop, (28, 28))
            cv2.imshow("Elements", im_resize)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # im_resize = np.reshape(im_resize, (28, 28, 1))
            train_data.append(im_resize)
        return train_data

    def processImage(self):
        img = cv2.imread(self.__mathEquationImg, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            self.showMathEquationImage(img)
            img = cv2.bitwise_not(img)  #Invert the image
            ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
            final_rect = self.__manageRectangle(cnt, thresh)

            train_data = self.getTrainData(thresh, final_rect)
            loaded_model = self.getModel()
            mathDataSet = MathDataSet()
            mathDictionary = mathDataSet.getMathDictionary()
            equString = ""
            for i in range(len(train_data)):
                train_data[i] = np.array(train_data[i])
                train_data[i] = train_data[i].reshape((1,28,28,1))
                # Deprecated method
                # result = loaded_model.predict_classes(train_data[i])
                result = np.argmax(loaded_model.predict(train_data[i]), axis=-1)
                value = mathDictionary.get(result[0])
                if value == "times":
                    equString += '*'
                else:
                    equString += value
            print("String : ",equString)
        return equString

    def solveEquation(self):
        equString = self.processImage()
        if equString is not None:
            print(eval(equString))
        else:
            print("Invalid Input")

if __name__ == '__main__':
    mathEquation = MathEquationSolver("assets/images/test0.png")
    mathEquation.solveEquation()