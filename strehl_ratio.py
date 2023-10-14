# Author: Eric Shore
# Purpose: To calculate the Strehl Ratio of an AO System

import cv2
import numpy as np
import pickle
import os
import datetime
import time
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from funzioni import *
from scipy.special import j1
from funzioni import *

img_avrg = []
img_norm = []
show = True  # Mostra l'immagine con il punto di massimo trovato in rosso
saveData = True  # Se su True salva i dati
pxToUm = 7.55e-8
lamb = 6.33e-7
diameter = .01
fuoco = .05
Imin = 0
errore = 3

Size = int(input('Dimensione area di ricerca (px): '))
avrgBreak = int(input('Numero di immagini da considerare: '))


def media_img():
    listNameImg = os.listdir(os.path.dirname(
        os.path.abspath(__file__)) + '/assets')
    h, w = cv2.imread('assets/'+listNameImg[0], 0).shape
    img_avrg = np.zeros((h, w), np.uint8)
    for k in range(len(listNameImg)):
        if k > avrgBreak:
            break
        name = listNameImg[k]
        print('Elaboro img: ', k, '->', name)
        image = cv2.imread('assets/'+name, 0)
        for y in range(h):
            for x in range(w):
                temp = (k*img_avrg[y, x]+image[y, x])/(k+1)
                img_avrg[y, x] = temp
    return img_avrg


def actual(data, vol_airy):
    dataTemp = np.array(data)
    h, w = dataTemp.shape
    centroX = round(w/2)
    centroY = round(h/2)
    minData = np.min(dataTemp)  # 0  # np.mean(dataTemp2)  # np.min(dataTemp)
    # N = 0
    # for y in range(h):
    #     for x in range(w):
    #         Y = y-centroY
    #         X = x-centroX
    #         r = (X**2+Y**2)**0.5
    #         if r > R:
    #             minData = minData + dataTemp[y, x]
    #             N = N + 1
    # minData = minData/N

    print('Minimo(V):\t', minData)

    img_norm = dataTemp-minData
    # img_norm = np.zeros((h, w), dtype=np.float32)
    # for y in range(h):
    #     for x in range(w):
    #         img_norm[y, x] = dataTemp[y, x]-minData
    # # img_norm = (dataTemp-minData)
    vol_data = float(np.sum(img_norm))
    img_norm = img_norm*vol_airy/vol_data
    print('Vol dati:\t', round(vol_data, 2))
    return [np.max(img_norm), img_norm]


def actual_profile(data, area_teoY, area_teoX):
    dataTemp = np.array(data)
    h, w = dataTemp.shape
    centroX = round(w/2)
    centroY = round(h/2)
    img_profileX = np.array(profile_dataX(data, centroY))
    img_profileY = np.array(profile_dataY(data, centroX))

    areaY = 0
    areaX = 0
    img_normX = []
    img_normY = []
    for y in range(h-1):
        areaY = areaY+float(img_profileY[y][0]+img_profileY[y+1][0])/2
        img_normY.append(img_profileY[y][0])

    for x in range(w-1):
        areaX = areaX+float(img_profileX[x][0]+img_profileX[x+1][0])/2
        img_normX.append(img_profileX[x][0])

    img_normX = np.array(img_normX)
    img_normY = np.array(img_normY)
    minX = np.min(img_normX)
    minY = np.min(img_normY)
    img_normX = (img_normX-minX)*area_teoX/areaX
    img_normY = (img_normY-minY)*area_teoY/areaY
    print('Minimo(X):\t', minX, '\nMinimo(Y):\t', minY)
    print('Vol dati(X):\t', areaX, '\nVol dati(Y):\t', areaY)
    return [np.max(img_normY), np.max(img_normX), img_normX, img_normY]


def ideal(data):
    dataTemp = np.array(data)
    h, w = dataTemp.shape
    centroX = round(w/2)
    centroY = round(h/2)
    airy_disc = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            Y = y-centroY
            X = x-centroX
            r = (X**2+Y**2)**0.5
            if r == 0:
                airy_disc[y, x] = 1
            else:
                r = r*pxToUm
                u = np.pi*diameter/lamb * (r/(r**2+fuoco**2)**0.5)
                airy_disc[y, x] = (2*j1(u)/(u))**2
    vol_airy = np.sum(airy_disc)
    return [vol_airy, airy_disc]


def ideal_profile(img_airy):
    dataTemp = np.array(img_airy)
    h, w = dataTemp.shape
    centroX = round(w/2)
    centroY = round(h/2)
    img_airyX = profile_dataX(img_airy, centroY)
    img_airyY = profile_dataY(img_airy, centroX)

    areaX = 0
    areaY = 0

    profX = []
    profY = []

    for y in range(h-1):
        areaY = areaY+float(img_airyY[y][0]+img_airyY[y+1][0])/2
        profY.append(img_airyY[y][0])

    for x in range(w-1):
        areaX = areaX+(img_airyX[x][0]+img_airyX[x+1][0])/2
        profX.append(img_airyX[x][0])

    return [profX, profY, areaY, areaX]


print('\n-----Inizio calcolo media sulla immagini-----')
img_avrg = media_img()


# Cerco se ci sono punti di massimo
print('\n-----Inizio ricerca massimo-----')
maxP = []
puntiMax = pMax(img_avrg, Imin, errore)
if len(puntiMax) > 1:
    print('Ho trovato ' +
          str(len(puntiMax)) + ' punti di massimo')
    print('Ricerca massimo: 0%')
    r = round(math.sqrt(len(puntiMax)/6.2931)/2)
    if r < 10:
        r = 10
    pMaxFiltrati = filtra_vicinato(
        puntiMax, img_avrg, r, len(puntiMax), 'Img media ')
    if len(pMaxFiltrati) == 1:
        maxP = pMaxFiltrati[0]
    else:
        print('Errore nella ricerca del massimo')
else:
    if len(puntiMax) == 1:
        print('Ho trovato ' +
              str(len(puntiMax)) + ' punto di massimo')
        print('Ricerca massimo: 0%')
        maxP = puntiMax[0]
        print('Ricerca massimo: 100%')
    else:
        print('Non ho trovato massimi')
CxMax = maxP[2]
CyMax = maxP[1]
print('\n-----Inizio calcolo parametri-----')
dataTemp = np.array(img_avrg)
h, w = dataTemp.shape

img_avrg_small = np.zeros((Size, Size), dtype=np.float32)
for y in range(Size):
    for x in range(Size):
        Y = CyMax-y+round(Size/2)
        X = CxMax-x+round(Size/2)
        img_avrg_small[y, x] = img_avrg[Y, X]

vol_airy, img_airy = ideal(img_avrg_small)

print('Vol airy:\t',  round(vol_airy))

img_airyX, img_airyY, areaY, areaX = ideal_profile(img_airy)

print('Area teo(Y):\t', round(areaY, 2),
      '\nArea teo(X)):\t', round(areaX, 2))

cp, img_norm = actual(img_avrg_small, vol_airy)
cpX, cpY, profX, profY = actual_profile(img_avrg_small, areaY, areaX)


print('S. ratio(V):\t', round(cp, 3), '\nS. ratio(X):\t',
      round(cpX, 3), '\nS. ratio(Y):\t', round(cpY, 3))


if saveData == True:
    print('\nSalvo i dati')
    profileX = profile_dataX(img_avrg, CyMax)
    profileY = profile_dataY(img_avrg, CxMax)
    midS = round(Size/2)
    with open('result/profile_fuoco_average_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(profileX)):
            file.write(str((i-midS)*pxToUm) + '\t' +
                       str(profileX[i][0]) + '\n')

    with open('result/profile_fuoco_average_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(profileY)):
            file.write(str((i-midS)*pxToUm) + '\t' +
                       str(profileY[i][0]) + '\n')

    with open('result/profile_norm_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(profY)):
            file.write(str((i-midS)*pxToUm) + '\t' + str(profY[i]) + '\n')

    with open('result/profile_norm_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(profX)):
            file.write(str((i-midS)*pxToUm) + '\t' + str(profX[i]) + '\n')

    with open('result/profile_teo_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(img_airyX)):
            file.write(str((i-midS)*pxToUm) + '\t' + str(img_airyX[i]) + '\n')

    with open('result/profile_teo_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(img_airyY)):
            file.write(str((i-midS)*pxToUm) + '\t' + str(img_airyY[i]) + '\n')

    with open('result/parametri.dat', 'w') as file:
        file.write('Strehl ratio:\n' + 'S. ratio(V):\t' + str(cp) + '\nS. ratio(X):\t' +
                   str(cpX) + '\nS. ratio(Y):\t' + str(cpY) + '\n')
        file.write('\nInfo:' + '\nDiametro(m):\t' + str(diameter) + '\npxToUm(m/px):\t' +
                   str(pxToUm) + '\nL. d\'onda(m):\t' + str(lamb) + '\nFuoco(m):\t' + str(fuoco))
        file.write('\nCentro X(px):\t' + str(CxMax) +
                   '\nCentro Y(px):\t' + str(CyMax))

if show == True:
    print('Mostro i dati')
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False)
    axs = axs.ravel()
    fig.suptitle('Grafici fuoco della metalente')
    midSize = pxToUm*(Size/2)
    x = np.linspace(-midSize, midSize, Size-1)

    axs[0].imshow(img_norm, vmin=0, vmax=1,
                  extent=(-midSize, midSize, - midSize, midSize))
    axs[0].set_xlabel('x(m)')
    axs[0].set_ylabel('y(m)')
    axs[0].title.set_text('Immagine fuoco sperimentale, normalizza')
    img0 = plt.imshow(img_norm, vmin=0, vmax=1,
                      extent=(-midSize, midSize, - midSize, midSize))
    cbar0 = fig.colorbar(img0, ax=axs[0])
    cbar0.set_label('I(au)')

    axs[1].imshow(img_airy, vmin=0, vmax=1,
                  extent=(-midSize, midSize, - midSize, midSize))
    axs[1].set_xlabel('x(m)')
    axs[1].set_ylabel('y(m)')
    img1 = plt.imshow(img_airy, vmin=0, vmax=1,
                      extent=(-midSize, midSize, - midSize, midSize))
    cbar1 = fig.colorbar(img1, ax=axs[1])
    cbar1.set_label('I(au)')
    axs[1].title.set_text('Immagine fuoco attesa')

    axs[2].plot(x, profX, color='blue')
    axs[2].set_xlabel('x(m)')
    axs[2].set_ylabel('I (au)')
    axs[2].title.set_text('Profilo fuoco lungo X, sperimentale')

    axs[3].plot(x, img_airyX, color='orange')
    axs[3].set_xlabel('x(m)')
    axs[3].set_ylabel('I(au)')
    axs[3].title.set_text('Profilo fuoco lungo X, atteso')

    axs[4].plot(x, profY, color='blue')
    axs[4].set_xlabel('y(m)')
    axs[4].set_ylabel('I(au)')
    axs[4].title.set_text('Profilo fuoco lungo Y, sperimentale')

    axs[5].plot(x, img_airyY, color='orange')
    axs[5].set_xlabel('y(m)')
    axs[5].set_ylabel('I(au)')
    axs[5].title.set_text('Profilo fuoco lungo Y, atteso')

    plt.show()

print('-----Fine-----')
