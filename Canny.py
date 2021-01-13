import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def preProcess(img):
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

def maskSudoku(img):
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

    res = cv2.bitwise_and(img, mask)

    return res


def findVerticalLines(img):
    """
    :param img: masked sudoku
    :return: binary image with vertical lines
    """
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    dx = cv2.convertScaleAbs(dx)
    # self.__displayImg([["sobel",dx],["img",img]])
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
        if mom['m00']==0:
            mom['m00']=2
        (x, y) = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
        centroids.append((x, y))
    return centroids[:100]


def sortCenters(centers):
    """
    :param centers: array of centers(x,y)
    :return: sorted array
    """
    centroids = np.array(centers, dtype=np.float32)
    c = centroids.reshape((100, 2))
    c2 = c[np.argsort(c[:, 1])]

    b = np.vstack([c2[i * 10:(i + 1) * 10][np.argsort(c2[i * 10:(i + 1) * 10, 0])] for i in range(10)])
    # bm = b.reshape((10, 10, 2))
    return b

def extractImages(centers, img,k):
    for i in range(9):
        for j in range(9):
            id = i * 10 + j
            TL, TR = centers[id], centers[id + 1]
            BL, BR = centers[id + 10], centers[id + 11]
            pts1 = np.float32([TL, TR, BL, BR])

            pts2 = np.float32([[0, 0], [55, 0], [0, 55], [55, 55]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            box = cv2.warpPerspective(img, M, (50, 50))
            # plt.subplot(9, 9, i * 9 + j + 1)
            box = 255 - box
            # plt.imshow(box, 'gray')
            # plt.xticks([]), plt.yticks([])
            if not os.path.isdir(f'./sudoku1/sudoku{k}'):
                os.mkdir(f'./sudoku1/sudoku{k}')
            cv2.imwrite(f"./sudoku1/sudoku{k}/{i * 9 + j}.jpg", box)
    # plt.show()

if __name__=="__main__":

    PATH='./sudokus'
    k=0
    for filename in os.listdir(PATH):
        img = cv2.imread(os.path.join(PATH, filename))
        if img is None:
            k+=1
            continue
        gray=preProcess(img)
        gray=maskSudoku(gray)
        masked=gray.copy()
        edges = cv2.Canny(gray,50,150,apertureSize = 3)

        minLineLength=100
        lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=60)

        a,b,c = lines.shape
        for i in range(a):
            cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

        # cv2.imshow("a",gray)

        masked1=gray.copy()

        kernel = np.ones((3,3),np.uint8)
        vt=findVerticalLines(gray)

        ht=findHorizotalLines(gray)

        ol = cv2.bitwise_and(vt,ht)
        ol = cv2.dilate(ol,kernel,iterations = 4)
        ol = cv2.erode(ol,kernel,iterations = 4)

        centroids = findCenters(ol)
        if len(centroids)<100:
            k+=1
            continue
        sorted = sortCenters(centroids)
        # sorted = cv2.convertScaleAbs(sorted)
        extractImages(sorted,masked,k)

        # cv2.imshow("alll",ol)
        # cv2.waitKey(0)

