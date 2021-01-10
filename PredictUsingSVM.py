import joblib
import cv2
import numpy as np

class PredictUsingSVM:
    def __init__(self):
        self._model = joblib.load('Number_model')

    def predict(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #     gray=Centering(gray)  '''For Centering the image and removing borders if necessary'''

        gray = gray.astype('float') / 255.0
        gray = cv2.resize(gray, (28, 28))
        gray = np.reshape(gray, (1, -1))
        pred = self._model.predict(gray)

        return pred[0]

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
            ROI = self.scale_and_centre(ROI, 120)
            return ROI

    def scale_and_centre(img, size, margin=10, background=0):
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