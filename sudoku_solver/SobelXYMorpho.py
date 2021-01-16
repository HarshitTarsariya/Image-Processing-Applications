import cv2
import numpy as np
from matplotlib import pyplot as plt


class SudokuSolverPlay:
    def __init__(self, image):
        """
        :param image: colored image read using cv2.imread
        """
        self.__originalImg = image

    def __preProcess(self, img):
        """
        :param img: original image
        :return: grayscale image
        """
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
        div = np.float32(gray) / close
        res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
        # res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        return res
        # self.__displayImg([["original", img], ["gray", res], ["color", res2]])

    def __maskSudoku(self, img):
        """
        :param img: grayscale image
        :return: sudoku masked image
        """
        thresh = cv2.adaptiveThreshold(img, 255, 0, 1, 19, 2)
        mask = np.zeros(img.shape, np.uint8)

        contour, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area, best_cnt = 0, None
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area > 1000:
                if area > max_area:
                    max_area = area
                    best_cnt = cnt

        cv2.drawContours(mask, [best_cnt], 0, 255, -1)
        cv2.drawContours(mask, [best_cnt], 0, 0, 2)

        res = img & mask
        return res

    def __findGridPoints(self, img):
        """
        :param img: masked sudoku
        :return: centers of gridpoints
        """

        def findVerticalLines(img):
            """
            :param img: masked sudoku
            :return: binary image with vertical lines
            """
            kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

            dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            dx = cv2.convertScaleAbs(dx)
            # cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
            _, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations=1)

            contour, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                x, y, w, h = cv2.boundingRect(cnt)
                if h / w < 5 or h < 70:
                    cv2.drawContours(close, [cnt], 0, 0, -1)
                else:
                    cv2.drawContours(close, [cnt], 0, 255, -1)
            # close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations=2)
            return close

        def drawLines(img, func):
            lines = cv2.HoughLines(img, 1, np.pi / 180, 200)
            temp = np.full(img.shape, 0, np.uint8)
            for line in lines:
                for rho, theta in line:
                    if func(theta):
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(temp, (x1, y1), (x2, y2), 255, 2)
            return temp

        def findHorizotalLines(img):
            """
            :param img: masked sudoku
            :return: binary image with horizontal lines
            """
            kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
            dy = cv2.Sobel(img, cv2.CV_16S, 0, 1)
            dy = cv2.convertScaleAbs(dy)
            # cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)
            _, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely)

            contour, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                x, y, w, h = cv2.boundingRect(cnt)
                if w / h < 5 or cv2.contourArea(cnt) < 70:
                    cv2.drawContours(close, [cnt], 0, 0, -1)
                else:
                    cv2.drawContours(close, [cnt], 0, 255, -1)
            close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernely, iterations=2)

            return close

        def findCenters(img):
            """
            :param img: binary image with white centers and black background
            :return: (x,y) points of centers not sorted
            """
            contour, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            centroids = []
            for cnt in contour:
                mom = cv2.moments(cnt)
                (x, y) = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
                centroids.append((x, y))
            return centroids

        def sortCenters(centers):
            """
            :param centers: array of centers(x,y)
            :return: sorted array
            """
            centroids = np.array(centers, dtype=np.float32)
            c = centroids.reshape((100, 2))
            c2 = c[c[:, 1].argsort()]

            b = np.vstack([c2[i * 10:(i + 1) * 10][c2[i * 10:(i + 1) * 10, 0].argsort()] for i in range(10)])
            # bm = b.reshape((10, 10, 2))
            return b

        vertical = findVerticalLines(img)
        horizontal = findHorizotalLines(img)
        whitecenters = vertical & horizontal
        whitecenters = cv2.morphologyEx(whitecenters, cv2.MORPH_DILATE, None, iterations=1)

        centers = findCenters(whitecenters)
        centers = sortCenters(centers)
        # self.__displayImg([["img", img],
        #                    ["vertical", vertical],
        #                    ["horizontal", horizontal],
        #                    ["whitecenters", whitecenters]])
        return centers

    def __displayImg(self, imges):
        """
        :param imges: [["Label",image],...]
        """
        for img in imges:
            cv2.imshow(img[0], img[1])
        cv2.waitKey(0)

    def __extractImages(self, centers, img):
        for i in range(9):
            for j in range(9):
                id = i * 10 + j
                TL, TR = centers[id], centers[id + 1]
                BL, BR = centers[id + 10], centers[id + 11]
                pts1 = np.float32([TL, TR, BL, BR])

                pts2 = np.float32([[0, 0], [50, 0], [0, 50], [50, 50]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                box = cv2.warpPerspective(img, M, (50, 50))
                plt.subplot(9, 9, i * 9 + j + 1)
                box = 255 - box
                plt.imshow(box, 'gray')
                plt.xticks([]), plt.yticks([])
                # cv2.imwrite(f"F:/PlayGround/sudoku/sudoku0/box{i * 9 + j}.jpg", box)
        plt.show()

    def solveSudoku(self):
        gray = self.__preProcess(self.__originalImg)
        masked = self.__maskSudoku(gray)
        centers = self.__findGridPoints(masked)

        self.__extractImages(centers, masked)
        # for i, centr in enumerate(centers, start=0):
        #     cv2.circle(self.__originalImg, (centr[0], centr[1]), 3, (255, 0, 0), -1)
        #     cv2.putText(self.__originalImg, f'{i}', (centr[0], centr[1]), cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (255, 255, 255), 2, cv2.LINE_AA)
        # self.__displayImg([["original", self.__originalImg], ["gray", gray], ["masked", masked]])


if __name__ == '__main__':
    colorImg = cv2.imread('../assets/images/sudoku/sudoku0.jpg')
    solver = SudokuSolverPlay(colorImg)
    solver.solveSudoku()
