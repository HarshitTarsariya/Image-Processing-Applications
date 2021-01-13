import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
def centering(gray):
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
        ROI = scale_and_centre(ROI, 60)
        return ROI


def scale_and_centre(img, size, margin=10, background=0):
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

img1=cv2.imread('./sudoku/sudoku5/19.jpg',0)
img=centering(img1)
if img is None:
    cv2.imshow('Image',img1)
else:
    cv2.imshow('Image',img)
cv2.waitKey(0)