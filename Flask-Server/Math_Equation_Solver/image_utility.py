import cv2

class ImageUtil:
    def __init__(self):
        pass

    def colorToGray(self, img):
        """
        :param img: original image
        :return: grayscale image
        """
        img = cv2.GaussianBlur(img, (7, 7), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def displayImgCV(self, imges):
        for img in imges:
            cv2.imshow(img[0], img[1])
        cv2.waitKey(0)

class EquationSolver:
    def __init__(self):
        self.__util = ImageUtil()

    def solveEq(self, color):
        gray = self.__util.colorToGray(color)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 41, 15)
        # thresh = self.removeHorizontalLine(gray)
        # self.__util.displayImgCV([["img", thresh], ["color", color]])
        return thresh

# if __name__ == '__main__':
#     solver = EquationSolver()
#     for i in range(1, 17):
#         img = cv2.imread(f'assets/images/blank/blank{i}.png')
#         if img is not None:
#             solver.solveEq(img)