import cv2
import numpy as np
import pickle
import os
import datetime
import time
import multiprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from funzioni import *


listImage = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '/assets')

centerX = 930
centerY = 100
direction = 1  # 0 è lungo x 1 lungo Y
saveData = True
showProfile = False

if __name__ == "__main__":
    nameImg = listImage[0]
    image = cv2.imread('assets/'+nameImg, 0)
    h, w = image.shape
    imagePrint = np.zeros((h, w, 3), np.uint8)

    prof = []
    if direction == 0:
        prof = profile_dataX(image, centerY)
    else:
        prof = profile_dataY(image, centerX)

    for y in range(h):
        for x in range(w):
            if direction == 0:
                imagePrint[y, x] = (image[y, x], image[y, x],
                                    image[y, x]) if y != centerY else (0, 0, 255)
            else:
                imagePrint[y, x] = (image[y, x], image[y, x],
                                    image[y, x]) if x != centerX else (0, 0, 255)

    if saveData == True:
        cv2.imwrite('result/profile_select_cx_'+str(centerX)+'_cy_' + str(centerY) +'_image.jpg', imagePrint)
        with open('result/profile_select_cx_'+str(centerX)+'_cy_' + str(centerY) + '.dat', 'w') as file:
            if direction == 0:
                file.write(
                    'Profili in X passanti dal centro lungo CY: ' + str(centerY) + '\n')
            else:
                file.write(
                    'Profili in Y passanti dal centro lungo CX: ' + str(centerX) + '\n')
            if direction == 0:
                file.write('Y\tI\n')
            else:
                file.write('X\tI\n')

            for val in prof:
                if direction == 0:
                    file.write(str(val[2]) + '\t', str(val[0]) + '\n')
                else:
                    file.write(str(val[1]) + '\t' + str(val[0]) + '\n')

    if showProfile == True:
        cv2.imshow('Image', imagePrint)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
