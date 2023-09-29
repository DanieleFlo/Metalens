import cv2
import math
# import numpy as np

# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def media_vicinato(yc, xc, image, r):
    h, w = image.shape
    somma = 0
    N = 1
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            xi = xc-i
            yj = yc-j
            if xi >= 0 and yj >= 0 and xi < w and yj < h:
                d = math.sqrt((xc-xi) ** 2 + (yc-yj) ** 2)
                if d <= r:
                    somma += image[yj, xi]
                    N += 1
    media = somma/N
    return [media, xc, yc]


def cerca_massimo(path):
    image = cv2.imread(path, 0)
    h, w = image.shape
    Imax = [[0, 0, 0]]
    nImax = 0
    for y in range(h):
        for x in range(w):
            Itemp = image[y, x]
            if Itemp >= Imax[nImax][0] and Itemp > 0:
                nImax += 1
                Imax.append([Itemp, y, x])

    listMedia = []
    for n in range(len(Imax)-1):
        listMedia.append(media_vicinato(Imax[n][1], Imax[n][2], image, 5))

    lista_ordinata = sorted(listMedia, key=lambda x: x[0])
    maxL = lista_ordinata[(len(lista_ordinata)-1)]
    print(maxL)


cerca_massimo('assets/test.tif')
