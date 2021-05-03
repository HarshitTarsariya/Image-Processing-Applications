import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import requests
from tensorflow.keras.models import load_model
from config import ROOT_DIR

class ImageUtil:
    def __init__(self):
        pass

    def centering(self,gray):
        thresh = 128
        # threshold the image
        gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

        # Find contours
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            if (x < 3 or y < 3 or h < 3 or w < 3):
                continue
            ROI = gray[y:y + h, x:x + w]
            ROI = self.scale_and_centre(ROI, 60)
            return ROI

    def scale_and_centre(self,img, size, margin=10, background=0):
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
        return cv2.resize(img, (size, size))
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

    def PILToBGR(self, img):
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image

    def getImageFromUrl(self, url):
        img = cv2.imread(url)
        return self.PILToBGR(img)

    def imageprepare(self,argv):
        """
        This function returns the pixel values.
        The imput is a png file location.
        """
        im = Image.open(argv).convert('L')
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = Image.new('L', (32, 32), (255))  # creates white canvas of 28x28 pixels

        if width > height:  # check which dimension is bigger
            # Width is bigger. Width becomes 20 pixels.
            nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
            if (nheight == 0):  # rare case but minimum is 1 pixel
                nheight = 1
                # resize and sharpen
            img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((32 - nheight) / 2), 0))  # calculate horizontal position
            newImage.paste(img, (4, wtop))  # paste resized image on white canvas
        else:
            # Height is bigger. Heigth becomes 20 pixels.
            nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
            if (nwidth == 0):  # rare case but minimum is 1 pixel
                nwidth = 1
                # resize and sharpen
            img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((32 - nwidth) / 2), 0))  # caculate vertical pozition
            newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

        # newImage.save("sample.png

        tv = list(newImage.getdata())  # get pixel values

        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
        print(tva)
        return tva
    def centering(self,gray):
        thresh = 128  # define a threshold, 128 is the middle of black and white in grey scale
        # threshold the image
        gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

        # Find contours
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            if (x < 3 or y < 3 or h < 3 or w < 3):
                # Note the number is always placed in the center
                # Since image is 28x28
                # the number will be in the center thus x >3 and y>3
                # Additionally any of the external lines of the sudoku will not be thicker than 3
                continue
            ROI = gray[y:y + h, x:x + w]
            # increasing the size of the number allws for better interpreation,
            # try adjusting the number and you will see the differnce
            ROI = self.scale_and_centre(ROI, 35)
            return ROI

    def scale_and_centre(self,img, size, margin=10, background=0):
        """Scales and centres an image onto a new background square."""
        h, w = img.shape[:2]

        def centre_pad(length):
            """Handles centering for a given length that may be odd or even."""
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
        return cv2.resize(img, (size, size))
    def display81Boxes(self, boxes):
        img_digits=[]
        for i in range(9):
            for j in range(9):
                plt.subplot(9, 9, i * 9 + j + 1)
                box = boxes[i * 9 + j][1:-1,1:-1]
                box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
                box = cv2.edgePreservingFilter(box, flags=1, sigma_s=60, sigma_r=0.4)
                box = cv2.detailEnhance(box, sigma_s=10, sigma_r=0.15)
                box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
                # # cv2.imwrite(f'./digits/{i * 9 + j}.jpg', box)
                box=self.centering(box)
                if not box is not None:
                    box=np.zeros((50,50))
                # box=255-box
                kernel=np.ones((4,4),dtype=float)

                # box=cv2.dilate(box,kernel)
                # box=255-box
                kernel_close = np.ones((2,2), np.uint8)
                closing = cv2.morphologyEx(box, cv2.MORPH_CLOSE, kernel_close)  # Delete space between line
                # cv2.imshow('closing', closing)
                # cv2.waitKey(0)
                box = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_close)
                box=cv2.GaussianBlur(box,(5,5),0)
                xx=cv2.resize(box, (32,32)).reshape(32,32, 1)
                # cv2.imshow('xx',xx)
                # cv2.waitKey(0)
                img_digits.append(xx)

                plt.imshow(box, 'gray')
                plt.xticks([]), plt.yticks([])
        plt.show()
        img_digits=np.array(img_digits).astype(np.float32)
        mean=img_digits.mean().astype(np.float32)
        std=img_digits.std().astype(np.float32)
        img_digits=(img_digits-mean)/std
        img_digits_np = img_digits
        model = load_model(ROOT_DIR+'/Flask-Server/Sudoku_Solver/Models/Number1.h5')
        preds_proba = model.predict(img_digits_np)

        preds = []
        nbr_digits_extracted = 0
        adapted_thresh_conf_cnn = 0.98
        for pred_proba in preds_proba:
            arg_max = np.argmax(pred_proba)
            if pred_proba[arg_max] > adapted_thresh_conf_cnn and arg_max < 9:
                preds.append(arg_max+1)
                nbr_digits_extracted += 1
            else:
                preds.append(-1)
        preds=np.array(preds).reshape(9,9)
        print(preds)

class SudokuSolver:

    def __init__(self, image):
        """
        :param image: colored image read using cv2.imread
        """
        self.__originalImg = image
        self.__util = ImageUtil()

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
        # self.__util.displayImages([margin10,margin0])
        return int(MaxH), int(MaxW), margin10, margin0

    def solveHough(self, grayImg):
        edges = cv2.Canny(grayImg, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        blackImg = np.zeros(grayImg.shape)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(blackImg, (x1, y1), (x2, y2), 255, 1)
        return blackImg

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
            for i in range(9):
                for j in range(9):
                    id = i * 10 + j
                    TL, TR = centers[id], centers[id + 1]
                    BL, BR = centers[id + 10], centers[id + 11]
                    pts1 = np.float32([TL, TR, BL, BR])
                    pts2 = np.float32([[0, 0], [50, 0], [0, 50], [50, 50]])
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    box = cv2.warpPerspective(img, M, (50, 50))
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
        # self.__util.displayImages([horizontal,whitecenters])
        return boxes

    def solveContour(self, grayImg):
        pass

    def solveConnectedCompAnalysis(self, grayImg):
        def removeNoise(img, minarea=5):
            contour, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            noise = []
            for cnt in contour:
                area = cv2.contourArea(cnt)
                if area < minarea:
                    noise.append(cnt)
            cv2.drawContours(img, noise, -1, 0, -1)

        def findVerticalLines(img, line_width=10, iteration=1):
            for _ in range(iteration):
                dil_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                kernal_v = np.ones((line_width, 1), np.uint8)
                imgY = cv2.dilate(img, dil_kernel, iterations=1)
                imgY = cv2.morphologyEx(imgY, cv2.MORPH_OPEN, kernal_v)
            return imgY

        def findHorizontalLines(img, line_width=10, iteration=1):
            for _ in range(iteration):
                dil_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                kernal_h = np.ones((1, line_width), np.uint8)
                imgX = cv2.dilate(img, dil_kernel, iterations=1)
                imgX = cv2.morphologyEx(imgX, cv2.MORPH_OPEN, kernal_h)
            return imgX

        edges = cv2.Canny(grayImg, 50, 100, apertureSize=3)
        sby = findVerticalLines(edges)
        sbx = findHorizontalLines(edges)
        removeNoise(sbx, minarea=50)
        removeNoise(sby, minarea=10)
        img = sby | sbx
        final_kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, final_kernel, iterations=1)
        # ret, labels, stats,centroids = cv2.connectedComponentsWithStats(img,
        #                                 connectivity=8, ltype=cv2.CV_32S)
        # colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # for x,y,w,h,area in stats[2:]:
        #     cv2.rectangle(colored,(x,y),(x+w,y+h),(0,255,0),2)

        self.__util.displayImages([img])

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

        # grid = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY_INV, 19, 2)
        grayImg = 255 - grayImg
        rows, boxes = np.vsplit(grayImg, 9), []
        for r in rows:
            cols = np.hsplit(r, 9)
            for c in cols:
                c = fillHolesContr(c)
                boxes.append(c)

        return boxes

    def solveSudoku(self):
        gray = self.__util.colorToGray(self.__originalImg)
        H, W, margin20, margin0 = self.__maskSudoku(gray, margin=20)
        # print(H,W)

        solvedMorphoXY = self.solveMorphoXY(margin20, H, W)
        if solvedMorphoXY != None:
            self.__util.display81Boxes(solvedMorphoXY)
            return

        # solvedContour = self.solveConnectedCompAnalysis(margin20)

        solvedBasic = self.solveBasic(margin0)
        if solvedBasic != None:
            self.__util.display81Boxes(solvedBasic)
            return



imgUrls=[
    ROOT_DIR+"Flask-Server/Sudoku/sudoku_1.jpg",
    "https://i.ibb.co/Kr4j1Ld/sudoku1.jpg",
    "https://i.ibb.co/Dg5NYxH/sudoku3.jpg",
    "https://i.ibb.co/9vHjQqg/sudoku4.jpg",
    "https://i.ibb.co/RzqYS7Y/sudoku2.jpg",
    "https://i.ibb.co/QjY3QrZ/sudoku5.jpg",
    "https://i.ibb.co/6r2wFxn/sudoku6.jpg",
    "https://i.ibb.co/qg1hd2r/sudoku7.jpg",
    "https://i.ibb.co/Fhrd9MK/sudoku8.jpg",
    "https://i.ibb.co/160Fhqq/sudoku9.jpg",
    "https://i.ibb.co/B3DX9ZF/004.jpg",
    "https://i.ibb.co/Xj25ft9/005.jpg",
    "https://i.ibb.co/mbJ8G1r/006.jpg",
    "https://i.ibb.co/K7sKP6C/007.jpg",
    "https://i.ibb.co/K6GPpCd/008.jpg",
    "https://i.ibb.co/XtsdGsG/009.jpg",
    "https://i.ibb.co/gZVKvXn/010.jpg",
    "https://i.ibb.co/Ms3vXRr/001.jpg",
    "https://i.ibb.co/8mrVvJN/002.jpg",
    "https://i.ibb.co/3zfd4nJ/003.jpg"
]
print(len(imgUrls))
solver = SudokuSolver(ImageUtil().getImageFromUrl(imgUrls[0]))
solver.solveSudoku()