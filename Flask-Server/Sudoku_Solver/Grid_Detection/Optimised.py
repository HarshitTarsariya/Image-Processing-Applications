import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from ..SudokuSolve import SudokuSolve
from config import ROOT_DIR

class ImageUtil:
    def __init__(self):
        pass

    def colorToGray(self, img):
        """
        :param img: original image
        :return: grayscale image
        """
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
        div = np.float32(gray) / close
        gray = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

        # res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        return gray

    def displayImages(self, imags, type='gray'):
        """
        :param imags: array of images
        :return: display using matplotlib
        """
        plt.figure(figsize=(20, 30), dpi=50, facecolor='w', edgecolor='k')
        cols = 2
        rows = math.ceil(len(imags) / cols)
        for i, img in enumerate(imags, start=1):
            plt.subplot(rows, cols, i)
            if type == 'gray':
                plt.imshow(img, type)
            else:
                plt.imshow(img)
            plt.xticks([]), plt.yticks([])
        plt.show()

    def displayImgCV(self, imges):
        for img in imges:
            cv2.imshow(img[0], img[1])
        cv2.waitKey(0)
    def centering(self,gray):
        thresh = 128
        # threshold the image
        gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

        # Find contours
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            if (x < 2 or y < 2 or h < 2 or w < 2):
                continue
            ROI = gray[y:y + h, x:x + w]
            ROI = self.scale_and_centre(ROI, 28)
            return ROI

    def scale_and_centre(self,img, size, margin=2, background=0):
        """Scales and centres an image onto a new background square."""
        h, w = img.shape[:2]

        def centre_pad(length):
            if length % 2 == 0:
                side1 = int((size - length) / 2)
                side2 = side1
            else:
                side1 = int((size - length) / 2)
                side2 = side1 + 1
            return side1, side2

        def scale(r, x):
            return int(r * x)

        if h > w:
            t_pad = int(margin / 2)
            b_pad = t_pad
            ratio = (size - margin) / h
            w, h = scale(ratio, w), scale(ratio, h)
            l_pad, r_pad = centre_pad(w)
        else:
            l_pad = int(margin / 2)
            r_pad = l_pad
            ratio = (size - margin) / w
            w, h = scale(ratio, w), scale(ratio, h)
            t_pad, b_pad = centre_pad(h)

        img = cv2.resize(img, (w, h))
        img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)

    def norm_digit(self,im):
        h, w = im.shape
        if h > w:
            top, left = round(h * 0.1), round((1.2 * h - w) / 2)
        else:
            top, left = round(w * 0.1), round((1.2 * w - h) / 2)

        return cv2.resize(
            cv2.copyMakeBorder(im, top, top, left, left, cv2.BORDER_CONSTANT),
            (20, 20)
        )
    def display81Boxes(self, boxes):
        img_digits = []
        for i in range(9):
            for j in range(9):
                gray = boxes[i * 9 + j]
                _, opt = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                bordersize = 8
                opt = cv2.copyMakeBorder(
                    opt,
                    top=bordersize,
                    bottom=bordersize,
                    left=bordersize,
                    right=bordersize,
                    borderType=cv2.BORDER_CONSTANT,
                    value=255
                )
                h, w = opt.shape[:2]
                mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(opt, mask, (0, 0), 0)
                opt = opt[bordersize - 1: opt.shape[0] - bordersize,
                      bordersize - 1: opt.shape[1] - bordersize]
                opt = cv2.morphologyEx(opt, cv2.MORPH_CLOSE, np.ones((3, 3)))
                # opt=self.norm_digit(opt)
                opt=cv2.GaussianBlur(opt,(1,1),0)
                opt = cv2.resize(opt, (28,28),interpolation=cv2.INTER_AREA).reshape(28,28,1)
                # boxes[i * 9 + j] = opt
                img_digits.append(opt)

                # getNum(color, opt)
                # plt.subplot(9, 9, i * 9 + j + 1)
                # plt.imshow(opt, 'gray')
                # plt.xticks([]), plt.yticks([])
        # plt.suptitle(self.__fnm)
        # plt.show()
        img_digits = np.array(img_digits).astype(np.float32)
        img_digits_np = img_digits/255.0
        model = load_model(ROOT_DIR+'/Flask-Server/Sudoku_Solver/Models/digitalusingsmall2.h5')
        preds_proba = model.predict(img_digits_np)

        preds = []
        nbr_digits_extracted = 0
        # adapted_thresh_conf_cnn = 0.50
        for pred_proba in preds_proba:
            arg_max = np.argmax(pred_proba)
            preds.append(arg_max)
        preds = np.array(preds).reshape(9, 9)
        # print(preds)
        return preds

class SudokuSolver:

    def __init__(self, image):
        """
        :param image: colored image read using cv2.imread
        """
        self.__originalImg = image
        self.__util = ImageUtil()
        self.__BoxW = 50
        self.__BoxH = 60

    def __maskSudoku(self, image, margin=10):
        """
        :param img: grayscale image
        :return: height,width sudoku masked image
        """

        def reorderPoints(points):
            try:
                points = points.reshape((4, 2))
            except:
                return points

            points = points[points[:, 1].argsort()]
            points[:2] = points[:2][points[:2, 0].argsort()]
            points[2:4] = points[2:4][points[2:4, 0].argsort()]
            return points

        def resizeSudoku(points, image, margin=10):
            points = np.float32(reorderPoints(points))
            MaxH = float(int(max(np.linalg.norm(points[0] - points[2]),
                                 np.linalg.norm(points[1] - points[3]))))
            MaxW = float(int(max(np.linalg.norm(points[0] - points[1]),
                                 np.linalg.norm(points[2] - points[3]))))

            dst = np.float32([
                [margin, margin],
                [margin + MaxW - 1, margin],
                [margin, margin + MaxH - 1],
                [margin + MaxW - 1, margin + MaxH - 1]
            ])
            M = cv2.getPerspectiveTransform(points, dst)
            margin = cv2.warpPerspective(image, M, (int(MaxW + margin * 2)
                                                    , int(MaxH + margin * 2)))
            return MaxH, MaxW, margin

        def maskSudoku(img):
            thresh = cv2.adaptiveThreshold(img, 255, 0, 1, 19, 2)
            mask = np.zeros(img.shape, np.uint8)
            contour, _ = cv2.findContours(thresh,
                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_area, best_cnt, points = 0, None, []
            for cnt in contour:
                area = cv2.contourArea(cnt)
                if area > 1000:
                    peri = cv2.arcLength(cnt, True)
                    pnts = cv2.approxPolyDP(cnt, 0.01 * peri, True)
                    if area > max_area and len(pnts) == 4:
                        max_area, best_cnt = area, cnt
                        points = pnts
            points = points.reshape((4, 2))

            cv2.drawContours(mask, [best_cnt], 0, 255, -1)
            cv2.drawContours(mask, [best_cnt], 0, 0, 2)
            res = img & mask
            return reorderPoints(points), res

        points, masked = maskSudoku(image)
        points = np.float32(points)

        MaxH = float(int(max(np.linalg.norm(points[0] - points[2]),
                             np.linalg.norm(points[1] - points[3]))))
        MaxW = float(int(max(np.linalg.norm(points[0] - points[1]),
                             np.linalg.norm(points[2] - points[3]))))
        Hmod, Wmod = MaxH % 9, MaxW % 9
        MaxH0 = MaxH + (-Hmod if Hmod < 9 - Hmod else 9 - Hmod)
        MaxW0 = MaxW + (-Wmod if Wmod < 9 - Wmod else 9 - Wmod)

        dst = np.float32([
            [margin, margin],
            [margin + MaxW - 1, margin],
            [margin, margin + MaxH - 1],
            [margin + MaxW - 1, margin + MaxH - 1]
        ])

        dst2 = np.float32([
            [0, 0],
            [MaxW0 - 1, 0],
            [0, MaxH0 - 1],
            [MaxW0 - 1, MaxH0 - 1]
        ])

        M10 = cv2.getPerspectiveTransform(points, dst)
        margin10 = cv2.warpPerspective(masked, M10, (int(MaxW + margin * 2)
                                                     , int(MaxH + margin * 2)))

        M0 = cv2.getPerspectiveTransform(points, dst2)
        margin0 = cv2.warpPerspective(masked, M0, (int(MaxW0), int(MaxH0)))
        return int(MaxH), int(MaxW), margin10, margin0

    # ***Good Performance
    def solveMorphoXY(self, grayImg, H, W):

        def findVerticalLines(img):
            """
            :param img: masked sudoku
            :return: binary image with vertical lines
            """
            kernelx = np.ones((10, 2))

            dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            dx = cv2.convertScaleAbs(dx)
            _, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations=1)

            contour, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                x, y, w, h = cv2.boundingRect(cnt)
                if h / w < 5 or h < 70:
                    cv2.drawContours(close, [cnt], 0, 0, -1)
                else:
                    cv2.drawContours(close, [cnt], 0, 255, -1)
            return close

        def findHorizotalLines(img):
            """
            :param img: masked sudoku
            :return: binary image with horizontal lines
            """
            kernely = np.ones((1, 10))
            dy = cv2.Sobel(img, cv2.CV_16S, 0, 1)
            dy = cv2.convertScaleAbs(dy)
            _, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely)

            contour, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                x, y, w, h = cv2.boundingRect(cnt)
                if w / h < 8 or cv2.contourArea(cnt) < 70:
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
                if mom['m00'] != 0:
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
            c2 = c[np.argsort(c[:, 1])]

            b = np.vstack([c2[i * 10:(i + 1) * 10][np.argsort(c2[i * 10:(i + 1) * 10, 0])] for i in range(10)])
            return b

        def extractImages(centers, img):
            boxes = []
            h, w = self.__BoxH, self.__BoxW
            for i in range(9):
                for j in range(9):
                    id = i * 10 + j
                    TL, TR = centers[id], centers[id + 1]
                    BL, BR = centers[id + 10], centers[id + 11]
                    pts1 = np.float32([TL, TR, BL, BR])
                    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    box = cv2.warpPerspective(img, M, (w, h))
                    box = 255 - box
                    boxes.append(box)
            return boxes

        vertical = findVerticalLines(grayImg)
        horizontal = findHorizotalLines(grayImg)
        whitecenters = vertical & horizontal
        whitecenters = cv2.morphologyEx(whitecenters, cv2.MORPH_CLOSE, None,
                                        iterations=2)
        centers = findCenters(whitecenters)
        if len(centers) != 100:
            return None
        sorted = sortCenters(centers)

        sorted = sorted.astype(int)

        boxes = extractImages(sorted, grayImg)
        return boxes

    # Last Option
    def solveBasic(self, grayImg):
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

        grayImg = 255 - grayImg
        rows, boxes = np.vsplit(grayImg, 9), []
        for r in rows:
            cols = np.hsplit(r, 9)
            for c in cols:
                c = fillHolesContr(c)
                cv2.resize(c, (self.__BoxW, self.__BoxH), interpolation=cv2.INTER_NEAREST)
                boxes.append(c)

        return boxes


    def solveSudoku(self):
        gray = self.__util.colorToGray(self.__originalImg)
        H, W, margin20, margin0 = self.__maskSudoku(gray, margin=20)

        boxes = self.solveMorphoXY(margin20, H, W)
        if boxes == None:
            boxes = self.solveBasic(margin0)

        grid=self.__util.display81Boxes(boxes)

        return  SudokuSolve().solver(grid).tolist()


if __name__ == '__main__':
    for i in range(12,13):
        solver = SudokuSolver(cv2.imread(ROOT_DIR+f'Flask-Server/Sudoku_Solver/Sudoku/sudoku_{i}.jpg'))
        solver.solveSudoku()
