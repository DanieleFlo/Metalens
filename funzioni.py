import math
import cv2
import numpy as np
import pickle
import os
import datetime

# Prende un punto in input e calcola la media pesata dalla distanza dei punti vicini compreso se stesso


def media_pesata_vicinato(yc, xc, image, r):
    h = len(image)  # Altezza della matrice
    w = len(image[0])  # Larghezza della matrice
    somma = 0
    N = 0
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            xi = xc-i
            yj = yc-j
            if xi >= 0 and yj >= 0 and xi < w and yj < h:
                d = math.sqrt((xc-xi) ** 2 + (yc-yj) ** 2)
                if d <= r:
                    Itemp = image[yj, xi]
                    if d != 0:
                        somma += Itemp*(1/d)
                        N += (1/d)
                    else:
                        somma += Itemp
                        N += 1
    media = somma/N
    return [image[yc, xc], media, yc, xc]


def pMax(dati, Imin, errore):  # Cerca i punti di massimo assoluti in una funzione all'interno dell'errore. I dati devono essere formati da una matrice HxW
    h = len(dati)  # Altezza della matrice
    w = len(dati[0])  # Larghezza della matrice

    dataMaxT = [[Imin, 0, 0]]
    Np = 0
    for y in range(h):
        for x in range(w):
            p = dati[y, x]
            if p > Imin and dataMaxT[Np][0] <= p:
                dataMaxT.append([p, y, x])
                Np += 1
    dataMax = []

    for k in range(len(dataMaxT) - 1, 0, -1):
        firstM = (len(dataMaxT) - 1)
        if dataMaxT[k][0] >= dataMaxT[firstM][0]-errore:
            dataMax.append(dataMaxT[k])
    return sorted(dataMax, key=lambda x: x[0])


# Prende in input un array di punti di massimo, allarga il vicinato finchè non trova quello ha in media il vicinato più iintenso
def filtra_vicinato(dati, image, r, pTot):
    h = len(image)  # Altezza della matrice
    w = len(image[0])  # Larghezza della matrice
    datiTemp = []
    for i in dati:
        datiTemp.append(media_pesata_vicinato(i[1], i[2], image, r))
    datiSMed = sorted(datiTemp, key=lambda x: x[1])

    newPmax = []
    for k in range(len(datiSMed) - 1, 0, -1):
        firstM = (len(datiSMed) - 1)
        if datiSMed[k][1] >= datiSMed[firstM][1]:
            newPmax.append(datiSMed[k])

    dataToSend = []
    for el in newPmax:
        dataToSend.append([el[0], el[2], el[3]])

    if len(dataToSend) > 1 and r < h/2 and r < w/2:
        if len(dati) != pTot:
            print('Ricerca massimo: ' + str(100 - int((len(dati)/pTot)*100)) + '%')
        return filtra_vicinato(dataToSend, image, (r+1), pTot)
    else:
        print('Ricerca massimo: 100%')
        return dataToSend


# Mostra un immagine in scala di grigio con un punto di colore diverso
def show_img_with_point(image, point, color):
    h = len(image)
    w = len(image[0])
    img = np.zeros((h, w, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            img[i, j] = (image[i, j], image[i, j], image[i, j])

    img[point[0], point[1]] = color
    cv2.imshow('Binary', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def profile_dataX(image, y):  # Restituisce il profilo lungo X
    profX = []
    w = len(image[0])  # Larghezza della matrice
    for x in range(w):
        profX.append([image[y, x], y, x])
    return profX


def profile_dataY(image, x):  # Restituisce il profilo lungo Y
    profY = []
    h = len(image)  # Altezza della matrice
    for y in range(h):
        profY.append([image[y, x], y, x])
    return profY


def prit_profile(nameFile, profile, info):  # Stampa su file un prfilo in I,X,Y
    with open('result/'+nameFile+'.dat', 'w') as file:
        file.write(info + '\n')
        file.write('I\tY\tX\n')
        for elemento in profile:
            file.write(str(elemento[0]) + '\t' +
                       str(elemento[1]) + '\t' + str(elemento[2]) + '\n')
