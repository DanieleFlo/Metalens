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
centroX = 790
centroY = 581
show = True  # Mostra l'immagine con il punto di massimo trovato in rosso
saveData = True  # Se su True salva i dati
pxToUm = 7.55e-8
lamb = 6.33e-7
diameter = .006
fuoco = .05
R = 100  # Raggio dello spot del fuoco
avrgBreak = 1000
Size = 350


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

    print('Min(V):', minData)

    img_norm = dataTemp-minData
    # img_norm = np.zeros((h, w), dtype=np.float32)
    # for y in range(h):
    #     for x in range(w):
    #         img_norm[y, x] = dataTemp[y, x]-minData
    # # img_norm = (dataTemp-minData)
    vol_data = float(np.sum(img_norm))
    img_norm = img_norm*vol_airy/vol_data
    print('Vol dati:', round(vol_data, 2))
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
    print('Min(X):', minX, '\nMin(Y):', minY)
    print('Vol dati(X):', areaX, '\nVol dati(Y):', areaY)
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


img_avrg = media_img()
print()
print('-----Inizio calcolo parametri-----')
dataTemp = np.array(img_avrg)
h, w = dataTemp.shape

img_avrg_small = np.zeros((Size, Size), dtype=np.float32)
for y in range(Size):
    for x in range(Size):
        Y = centroY-y+round(Size/2)
        X = centroX-x+round(Size/2)
        img_avrg_small[y, x] = img_avrg[Y, X]

vol_airy, img_airy = ideal(img_avrg_small)

print('Vol airy:',  round(vol_airy))

img_airyX, img_airyY, areaY, areaX = ideal_profile(img_airy)

print('Area teorica (Y):', round(areaY, 2),
      '\nArea teorica (X):', round(areaX, 2))

cp, img_norm = actual(img_avrg_small, vol_airy)
cpX, cpY, profX, profY = actual_profile(img_avrg_small, areaY, areaX)


print('S. ratio(V):', round(cp, 3), '\nS. ratio(X):',
      round(cpX, 3), '\nS. ratio(Y):', round(cpY, 3))


if saveData == True:
    print('Salvo i dati')
    profileX = profile_dataX(img_avrg, centroY)
    profileY = profile_dataY(img_avrg, centroX)
    with open('result/profile_fuoco_average_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(profileX)):
            file.write(str(i) + '\t' + str(profileX[i][0]) + '\n')

    with open('result/profile_fuoco_average_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(profileY)):
            file.write(str(i) + '\t' + str(profileY[i][0]) + '\n')

    with open('result/profile_norm_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(profY)):
            file.write(str(i) + '\t' + str(profY[i]) + '\n')

    with open('result/profile_norm_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(profX)):
            file.write(str(i) + '\t' + str(profX[i]) + '\n')

    with open('result/profile_teo_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(img_airyX)):
            file.write(str(i) + '\t' + str(img_airyX[i]) + '\n')

    with open('result/profile_teo_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(img_airyY)):
            file.write(str(i) + '\t' + str(img_airyY[i]) + '\n')

    with open('result/parametri.dat', 'w') as file:
        file.write('S. ratio(V):' + str(cp) + '\nS. ratio(X):' +
                   str(cpX) + '\nS. ratio(Y):' + str(cpY))
        file.write('\nDiametro(m):' + str(diameter) + '\npxToUm(m/px):' +
                   str(pxToUm) + '\nlamb(m):' + str(lamb) + '\nfuoco(m):' + str(fuoco))
        file.write('\ncentroX(px):' + str(centroX) +
                   '\ncentroY(px):' + str(centroY))

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
