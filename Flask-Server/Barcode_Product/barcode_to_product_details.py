import cv2
# import numpy as np
from pyzbar.pyzbar import decode
import requests


class BarcodeToProductDetails():
    def __init__(self, img):
        """
        :param image: Image of Barcode Image
        """
        self.api_url = 'https://barcode.monster/api/'
        self.img = img

    def getProductInformation(self):
        """
        :return: Product information
        """

        for barcode in decode(self.img):
            barcodeNumber = barcode.data.decode('utf-8')
            print('Barcode detected : ', barcodeNumber)
            # Making api call
            r = requests.get(url=self.api_url + str(barcodeNumber))
            productData = r.json()
            if productData['status'] == 'active':
                return 'Product Details : ' + productData['description'] #.replace('(from barcode.monster)','')
            else:
                return 'Sorry, Product Not Found'

if __name__ == '__main__':
    # Loading image from the file system
    barcodeObj = BarcodeToProductDetails(cv2.imread('filename'))
    print(barcodeObj.getProductInformation())
