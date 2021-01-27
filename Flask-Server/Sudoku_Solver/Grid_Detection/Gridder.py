import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from config import ROOT_DIR

class Gridder:

    def __init__(self, path):
        img = path
        self.__original_img = path

    def __preProcessing(self, img):
        """
        :param img: colored image
        :return:binary image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converting Image to Gray for better thresholding
        gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))
        blurred = cv2.GaussianBlur(gray_enhance, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 15)
        inversion = cv2.bitwise_not(thresh)

        kernel_close = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(inversion, cv2.MORPH_CLOSE, kernel_close)  # Removing spaces between line
        dilate = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_close)  # To make line clearer by dilation
        # cv2.imshow("dilate",dilate)
        # cv2.waitKey(0)
        return dilate

    def __findCorners(self, img):
        """
        :param img: binary image
        :return: boundry points of multiple sudoku
        """

        def __getCorners(contour):

            top_left = [10000, 10000]
            top_right = [0, 10000]
            bottom_right = [0, 0]
            bottom_left = [10000, 0]

            mean_x = np.mean(contour[:, :, 0])
            mean_y = np.mean(contour[:, :, 1])

            for j in range(len(contour)):
                x, y = contour[j][0]
                if x > mean_x:  # On right
                    if y > mean_y:  # On bottom
                        bottom_right = [x, y]
                    else:
                        top_right = [x, y]
                else:
                    if y > mean_y:  # On bottom
                        bottom_left = [x, y]
                    else:
                        top_left = [x, y]
            return [top_left, top_right, bottom_right, bottom_left]

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contours = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        biggest_area = cv2.contourArea(contours[0])

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 75000:
                break
            if area > biggest_area / 2:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
                if len(approx) == 4:
                    best_contours.append(approx)
        corners = []
        for best_contour in best_contours:
            corners.append(__getCorners(best_contour))

        return corners

    def __transformPerspective(self, img, allCorners):
        """
        :param img: original colored image
        :param allCorners: array of sudoku's corners
        :return: cropped sudoku's array
        """
        allSudokusInImage = []
        for corners in allCorners:
            plotting = np.array([[0, 0], [450 - 1, 0], [450 - 1, 450 - 1], [0, 450 - 1]], dtype=np.float32)
            transforming = cv2.getPerspectiveTransform(corners, plotting)
            transform = cv2.warpPerspective(img, transforming, (450, 450))
            allSudokusInImage.append(transform)
        return allSudokusInImage

    # def __verify_viable_grid(self, grid_tested):
    #     for y in range(9):
    #         for x in range(9):
    #             if grid_tested[y, x] == 0:
    #                 continue
    #             grid = grid_tested.copy()
    #             grid[y, x] = 0
    #             line = grid[y, :]
    #             column = grid[:, x]
    #             x1 = 3 * (x // 3)
    #             y1 = 3 * (y // 3)
    #             x2, y2 = x1 + 3, y1 + 3
    #             square = grid[y1:y2, x1:x2]
    #             val = grid_tested[y, x]
    #             if val in line or val in column or val in square:
    #                 # print("Unviable Grid")
    #                 return False
    #
    #     return True

    def __getAllBoxes(self, allSudokusInImage):
        """
        :param allSudokusInImage: all sudoku's cropped img array
        :return: sudoku grids
        """

        def fill_numeric_grid(preds, loc_digits, h_im, w_im):
            grid = np.zeros((9, 9), dtype=int)

            for pred, loc in zip(preds, loc_digits):
                if pred > 0:
                    y, x = loc
                    true_y = int(9 * y // h_im)
                    true_x = int(9 * x // w_im)
                    grid[true_y, true_x] = pred

            return grid

        def __preprocessing_im_grid(img, is_gray=False):
            if is_gray:
                gray = img
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))
            blurred = cv2.GaussianBlur(gray_enhance, (11, 11), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 29, 25)
            return thresh, gray_enhance

        black = np.ones((28, 28))
        grids = []
        if allSudokusInImage is not None:
            for img in allSudokusInImage:
                h_im, w_im = img.shape[:2]
                im_prepro, gray_enhance = __preprocessing_im_grid(img)
                cv2.imshow('im_prepro', im_prepro)
                cv2.waitKey(0)
                cv2.imshow('gray_enhance', gray_enhance)
                cv2.waitKey(0)
                contours, _ = cv2.findContours(im_prepro, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_digits = []
                loc_digits = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    y_true, x_true = y + h / 2, x + w / 2
                    if x_true < 10 or y_true < 10 or x_true > w_im - 10 or y_true > h_im - 10:
                        continue
                    if 15 < h < 50 and 210 < w * h < 900:
                        y1, y2 = y - 2, y + h + 2
                        border_x = max(1, int((y2 - y1 - w) / 2))
                        x1, x2 = x - border_x, x + w + border_x
                        digit_cut = gray_enhance[max(y1, 0):min(y2, h_im), max(x1, 0):min(x2, w_im)]
                        _, digit_thresh = cv2.threshold(digit_cut,
                                                        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        cv2.imshow("thresh",digit_thresh)
                        cv2.waitKey(0)
                        img_digits.append(
                            cv2.resize(digit_thresh, (28, 28), interpolation=cv2.INTER_NEAREST).reshape(28, 28, 1))
                        loc_digits.append([y_true, x_true])
                if not img_digits:
                    return None
                if not img_digits:
                    return None
                img_digits_np = np.array(img_digits) / 255.0
                model = load_model('../Dataset/my_model.h5')
                preds_proba = model.predict(img_digits_np)

                preds = []
                nbr_digits_extracted = 0
                adapted_thresh_conf_cnn = 0.98
                for pred_proba in preds_proba:
                    arg_max = np.argmax(pred_proba)
                    if pred_proba[arg_max] > adapted_thresh_conf_cnn and arg_max < 9:
                        preds.append(arg_max + 1)
                        nbr_digits_extracted += 1
                    else:
                        preds.append(-1)

                if nbr_digits_extracted < 13:
                    return None
                grid = fill_numeric_grid(preds, loc_digits, h_im, w_im)
                grids.append(grid)
        return grids

    def __showBoxes(self, boxes):
        for i in range(9):
            for j in range(9):
                plt.subplot(9, 9, i * 9 + j + 1)
                box = boxes[i * 9 + j]
                box = cv2.bitwise_not(box)
                plt.imshow(box, 'gray')
                plt.xticks([]), plt.yticks([])
        plt.show()

    def __displayImg(self, imges):
        for img in imges:
            cv2.imshow(img[0], img[1])
        cv2.waitKey(0)

    def Solver(self):
        dilate = self.__preProcessing(self.__original_img)
        cv2.imshow('dilate',dilate)
        cv2.waitKey(0)
        corners = self.__findCorners(dilate)

        allSudokusInImage = self.__transformPerspective(self.__original_img,
                                                        np.array(corners, dtype=np.float32))
        grids = self.__getAllBoxes(allSudokusInImage)
        print(grids)
        sdks = []
        for idx, sdk in enumerate(allSudokusInImage, start=1):
            sdks.append([f"sdk{idx}", sdk])
        self.__displayImg(sdks)


if __name__ == '__main__':
    img=cv2.imread(ROOT_DIR+'/Flask-Server/Sudoku_Solver/Sudoku/sudoku_7.jpg')
    ext = Gridder(img)
    ext.Solver()
