import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
import time


class ImageUtility:
    def resizeImg(self, img, w, h):
        im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        im_pil.thumbnail((w, h))
        img = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        return img

    def displayImgCV(self, imgArr=[]):
        if imgArr:
            for img in imgArr:
                cv2.imshow(img[0], img[1])
            cv2.waitKey(0)

    def saveBoxImage(self, img, value):
        filename = "cut_imgs/" + value + str(time.time()) + '.jpg'
        cv2.imwrite(filename, cv2.bitwise_not(img))

    def displayImages(self, imags, type='gray'):
        plt.figure(figsize=(10, 10), dpi=50, facecolor='w', edgecolor='k')
        cols = 3
        rows = math.ceil(len(imags) / cols)
        for i, img in enumerate(imags, start=1):
            plt.subplot(rows, cols, i)
            if type == 'gray':
                plt.imshow(img, type)
            else:
                plt.imshow(img)
            plt.xticks([]), plt.yticks([])
        plt.show()