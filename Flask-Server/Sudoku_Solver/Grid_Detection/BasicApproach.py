import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage
from scipy import ndimage
from config import ROOT_DIR

class SudokuSolver:
    def __init__(self, image):
        """
        :param image: colored image read using cv2.imread
        """
        self.__originalImg = image

    def __applyThresolding(self, image):
        """
        :param image: colored image
        :return: grayscale image,binary image
        """
        grayscal = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(grayscal, 1, 195, 195)
        # img = cv2.medianBlur(img, 1)
        binary = cv2.adaptiveThreshold(grayscal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 19, 2)
        return 255-filtered, binary

    def __getSudokuBoundry(self, contours):
        """
        :param contours: array of contour found using cv2.findContours
        :return: counter of sudoku boundry , 4 endpoints
        """
        ans, max_area, endpoints = [], 0, np.array([])
        for curr in contours:
            currArea = cv2.contourArea(curr)
            if currArea > 1000:
                peri = cv2.arcLength(curr, True)
                approx = cv2.approxPolyDP(curr, 0.01 * peri, True)
                if approx.shape[0] == 4 and currArea > max_area:
                    ans = [curr]
                    max_area, endpoints = currArea, approx

        def reorderPoints(points):
            try:
                points = points.reshape((4, 2))
            except:
                return points
            points = points[points[:, 1].argsort()]
            points[:2] = points[:2][points[:2, 0].argsort()]
            points[2:4] = points[2:4][points[2:4, 0].argsort()]
            return points

        return ans, reorderPoints(endpoints)

    def __cutSudokuFromImg(self, img, points):
        """
        :param img: Binary Image of Sudoku
        :param points: 4 endpoints of Sudoku
        :return: cropped grid from image
        """
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [549, 0], [0, 549], [549, 549]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (549, 549))

        return dst

    def __displayBoxes(self, boxes):
        """
        :param boxes: numpy array of 81 boxes
        """
        img_digits=[]
        for i in range(9):
            for j in range(9):
                plt.subplot(9, 9, i * 9 + j + 1)
                box = boxes[i * 9 + j]

                a,box=cv2.threshold(box,130,255,cv2.THRESH_BINARY)
                # print(box)
                box = box[7:len(box) - 9, 10:len(box) - 10]
                box1 = np.zeros((box.shape[0], box.shape[1]), np.uint8)
                box2 = np.zeros((70, 70), np.uint8)

                box = cv2.bitwise_or(box1, box)
                box2[12:12 + box.shape[0], 17:17 + box.shape[1]] = box
                kernel = np.ones((2, 2), np.uint8)
                box2 = cv2.dilate(box2, kernel, iterations=1)
                img_digits.append(
                    cv2.resize(box2, (28, 28), interpolation=cv2.INTER_NEAREST).reshape(28, 28, 1))
                plt.imshow(box2, 'gray')
                plt.xticks([]), plt.yticks([])
        plt.show()

    def __gridToBoxes(self, grid):
        """
        :param grid: Binary Image of Sudoku grid
        :return: array of 81 boxes extracted from grid
        """

        def fillHolesContr(box1):
            box = box1.copy()
            contr, _ = cv2.findContours(box, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contr:
                if cv2.contourArea(cnt) < 100:
                    cv2.drawContours(box, [cnt], -1, 0, -1)
            return box

        def fillHolesMorpho(box):
            kernel = np.ones((3, 3), np.uint8)
            return cv2.morphologyEx(box, cv2.MORPH_OPEN, kernel)

        rows, boxes = np.vsplit(grid, 9), []
        for r in rows:
            cols = np.hsplit(r, 9)
            for c in cols:
                c = fillHolesContr(c)
                boxes.append(c)

        return np.array(boxes)

    def __displayImg(self, imges):
        for img in imges:
            cv2.imshow(img[0], img[1])
        cv2.waitKey(0)

    def solveSudoku(self):

        grayImg, binaryImg = self.__applyThresolding(self.__originalImg)
        contours, _ = cv2.findContours(binaryImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boundry, endpoints = self.__getSudokuBoundry(contours)
        grayImg = self.__cutSudokuFromImg(grayImg, endpoints)
        boxes = self.__gridToBoxes(grayImg)
        self.__displayBoxes(boxes)

if __name__ == '__main__':
    colorImg = cv2.imread(ROOT_DIR+'Flask-Server/Sudoku_Solver/Sudoku/sudoku_12.jpg')
    solver = SudokuSolver(colorImg)
    solver.solveSudoku()
