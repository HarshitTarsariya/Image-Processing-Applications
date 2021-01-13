import cv2
import numpy as np
import os

#Biggest Contour Finder
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
    return biggest,max_area

#Reordering the contour points as (top-lh corner, top-rh corner, bottom-lh corner, bottom-rh corner)
def reorder(contourPoints):
    contourPoints = contourPoints.reshape((4, 2))
    contourPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = contourPoints.sum(1)
    contourPointsNew[0] = contourPoints[np.argmin(add)]
    contourPointsNew[3] = contourPoints[np.argmax(add)]
    diff = np.diff(contourPoints, axis=1)
    contourPointsNew[1] = contourPoints[np.argmin(diff)]
    contourPointsNew[2] = contourPoints[np.argmax(diff)]
    return contourPointsNew

def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 3)
    imgThreshold = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    return(imgThreshold)

def splitImage(img):
  rows = np.vsplit(img, 9)
  cells = []
  for row in rows:
    boxes = np.hsplit(row, 9)
    for box in boxes:
      cells.append(box)
  return cells

def extractImage(boxes,k):
    i=0
    for image in boxes:
        img = image
        img = img[5:img.shape[0] - 5, 5:img.shape[1] -5]
        ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
        img = cv2.resize(img, (50,50),cv2.INTER_NEAREST)
        if not os.path.isdir(f'./sudoku/sudoku{k}'):
            os.mkdir(f'./sudoku/sudoku{k}')

        cv2.imwrite(f"./sudoku/sudoku{k}/{i}.jpg",img)
        i+=1

if __name__=="__main__":

    PATH='./sudokus'
    k=0
    for filename in os.listdir(PATH):
        img = cv2.imread(os.path.join(PATH, filename))
        imgHeight = 450
        imgWidth = 450
        img = cv2.resize(img, (imgWidth, imgHeight))

        blankImg = np.zeros_like(img)
        preprocessedImg = preprocessing(img)
        imgContour = img.copy()
        contours, hierarchy = cv2.findContours(preprocessedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContour, contours, -1, (0, 0, 255), 2)

        bigImgContour = img.copy()
        points, area = biggestContour(contours)
        if len(points) == 0:
            k+=1
            continue
        reorderedPoints = reorder(points)
        cv2.drawContours(bigImgContour, reorderedPoints, -1, (0, 0, 255), 10)

        p1 = np.float32(reorderedPoints)
        p2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
        matrix = cv2.getPerspectiveTransform(p1, p2)
        imgWarpCol = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))
        imgWarpGray = cv2.cvtColor(imgWarpCol, cv2.COLOR_BGR2GRAY)

        indivBoxes = splitImage(imgWarpGray)
        extractImage(indivBoxes,k)
        k+=1