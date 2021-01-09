import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


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
        # brightImg = self.__adjustBrightness(grayscal)

        # img = cv2.medianBlur(img, 1)
        binary = cv2.adaptiveThreshold(grayscal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 19, 2)
        # self.__displayImg([["colorImg", colorImg], ["grayscal", grayscal],
        #                    ["filtered", filtered],["brightImg",brightImg]])
        return filtered, binary

    def __getSudokuBoundry(self, contours):
        """
        :param contours: array of contour found using cv2.findContours
        :return: counter of sudoku boundry , 4 endpoints
        """
        ans, max_area, endpoints = [], 0, np.array([])
        for curr in contours:
            currArea = cv2.contourArea(curr)
            # if currArea > 50:
            #     ans.append(curr)

            if currArea > 1000:
                peri = cv2.arcLength(curr, True)
                approx = cv2.approxPolyDP(curr, 0.01 * peri, True)
                # print(approx.shape)
                if approx.shape[0] == 4 and currArea > max_area:
                    ans = [curr]
                    max_area, endpoints = currArea, approx

        def reorderPoints(points):
            try:
                points = points.reshape((4, 2))
            except:
                return points
            points = sorted(points, key=lambda x: x[1])
            if points[0][0] > points[1][0]:
                points[0], points[1] = points[1], points[0]
            if points[2][0] > points[3][0]:
                points[2], points[3] = points[3], points[2]
            points = np.array(points)
            return points

        return ans, reorderPoints(endpoints)

    def __cutSudokuFromImg(self, img, points, contours):
        """
        :param img: Binary Image of Sudoku
        :param points: 4 endpoints of Sudoku
        :param contours: contours to be preserved
        :return: cropped grid from image
        """
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [549, 0], [0, 549], [549, 549]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (549, 549))

        # h, w = img.shape[:2]
        # mask = np.zeros((h + 2, w + 2), np.uint8)
        # cv2.floodFill(img, mask, (0, 0), 0)
        return dst

    def __adjustBrightness(self, img, gamma=1.5):
        invGamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)
        ],np.uint8)
        return cv2.LUT(img, table)

    def __displayBoxes(self, boxes):
        """
        :param boxes: numpy array of 81 boxes
        """
        for i in range(9):
            for j in range(9):
                plt.subplot(9, 9, i * 9 + j + 1)
                box = boxes[i * 9 + j]
                plt.imshow(box, 'gray')
                plt.xticks([]), plt.yticks([])
                cv2.imwrite(f"F:/pycharm/LearnAI/assets/images/sudoku/sudoku0/box{i*9+j}.jpg",boxes[i * 9 + j])

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
        # _start = time.time()

        colorImg = cv2.resize(self.__originalImg, (550, 550), interpolation=cv2.INTER_CUBIC)
        grayImg, binaryImg = self.__applyThresolding(colorImg)
        contours, _ = cv2.findContours(binaryImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boundry, endpoints = self.__getSudokuBoundry(contours)

        # cv2.drawContours(colorImg, boundry, -1, (0, 0, 255), 2)
        # self.__displayImg([["img", colorImg]])

        binaryImg = self.__cutSudokuFromImg(binaryImg, endpoints, boundry)
        # boxes = self.__gridToBoxes(binaryImg)

        # print(f"{time.time()-_start} seconds")

        self.__displayImg([["gray", grayImg], ["binaryImg", binaryImg]])
        # self.__displayBoxes(boxes)


if __name__ == '__main__':
    colorImg = cv2.imread('../assets/images/sudoku/sudoku0.jpg')
    solver = SudokuSolver(colorImg)
    solver.solveSudoku()
