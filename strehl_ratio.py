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
from scipy import integrate

# Variabili setup
show = True  # Mostra l'immagine con il punto di massimo trovato in rosso
saveData = True  # Se su True salva i dati
pxToUm = 7.55e-8
lamb = 5.32e-7
diameter = .01
fuoco = .0569
Imin = 0
errore = 3

# Variabili globali
img_avrg = []
img_norm = []
conv = False
sepDec = '.'
spectrum = []

f = open('sp500nm.txt', 'r')
riga = f.readline()
while riga != "":
    lineText = (riga.replace('\n', '')).split('\t')
    lambTemp = float(lineText[0])*1e-9
    IntTemp = float(lineText[1])
    spectrum.append([lambTemp, IntTemp])
    riga = f.readline()
f.close()

while True:
    convolution = input(
        'Il disco di airy è una convoluzione con uno spettro (y/n): ')
    if convolution == 'yes' or convolution == 'y' or convolution == 'si':
        conv = True
        break
    else:
        if convolution == 'n' or convolution == 'no' or convolution == 'si':
            conv = False
            break

while True:
    Size = int(input('Dimensione area di ricerca dispari (px): '))
    if Size > 10 and Size % 2 != 0:
        break

while True:
    avrgBreak = int(input('Numero di immagini da considerare: '))-1
    if avrgBreak >= 0:
        break

if saveData == True:
    while True:
        sepDec = input('Separazione decimale (. o ,): ')
        if sepDec == ',' or sepDec == '.':
            break


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
    minData = np.min(dataTemp)
    print('Minimo(V):\t', minData)
    img_norm = dataTemp-minData
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

    yy = np.zeros(h, dtype=np.float32)
    for y in range(h-1):
        img_normY.append(img_profileY[y][0])
        yy[y] = img_profileY[y][0]

    xy = np.arange(0, h)
    areaY = integrate.simpson(yy, xy)

    yx = np.zeros(w, dtype=np.float32)
    for x in range(w-1):
        img_normX.append(img_profileX[x][0])
        yx[x] = img_profileX[x][0]

    xx = np.arange(0, w)
    areaX = integrate.simpson(yx, xx)

    img_normX = np.array(img_normX)
    img_normY = np.array(img_normY)
    minX = np.min(img_normX)
    minY = np.min(img_normY)
    img_normX = (img_normX-minX)*area_teoX/areaX
    img_normY = (img_normY-minY)*area_teoY/areaY
    print('Minimo(X):\t', minX, '\nMinimo(Y):\t', minY)
    print('Area dati(X):\t', round(areaX, 2),
          '\nArea dati(Y):\t', round(areaY, 2))
    return [np.max(img_normY), np.max(img_normX), img_normX, img_normY]


def ideal(data, Lamb, norm):
    dataTemp = np.array(data)
    h, w = dataTemp.shape
    centroX = round((w-1)/2)
    centroY = round((h-1)/2)
    airy_disc = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            Y = y-centroY
            X = x-centroX
            r = (X**2+Y**2)**0.5
            if r == 0:
                airy_disc[y, x] = 1*norm
            else:
                r = r*pxToUm
                u = np.pi*diameter/Lamb * (r/(r**2+fuoco**2)**0.5)
                airy_disc[y, x] = norm*(2*j1(u)/(u))**2
    vol_airy = np.sum(airy_disc)
    return [vol_airy, airy_disc]


def convolutionAxS(data):
    dataTemp = np.array(data)
    h, w = dataTemp.shape
    airy_disc = np.zeros((h, w), dtype=np.float32)

    # Calcolo l'aria dello spettro
    aria_spettro = 0
    for line in spectrum:
        aria_spettro = aria_spettro + line[1]

    # Normalizzo lo spettro
    spectrum_norm = []
    for l in spectrum:
        spectrum_norm.append([l[0], l[1]/aria_spettro])

    # Calcolo la convoluzione del disco di airy rispetto allo spettro
    print('Calcolo convoluzione: 0%')
    for k in range(len(spectrum_norm)):
        line = spectrum_norm[k]
        vol_airy_temp, img_airy_temp = ideal(data, line[0], line[1])

        airy_disc = airy_disc + np.array(img_airy_temp)
        print(np.max(airy_disc))
        print('Calcolo convoluzione: ' +
              str(round(((k+1)/len(spectrum_norm))*100)) + '%')

    vol_airy = np.sum(airy_disc)
    print('-----Fine calcolo convoluzione-----')
    return [vol_airy, airy_disc]


def ideal_profile(img_airy):
    dataTemp = np.array(img_airy)
    h, w = dataTemp.shape
    centroX = round((w-1)/2)
    centroY = round((h-1)/2)
    img_airyX = profile_dataX(img_airy, centroY)
    img_airyY = profile_dataY(img_airy, centroX)

    areaX = 0
    areaY = 0

    profX = []
    profY = []

    yy = np.zeros(h, dtype=np.float32)
    for y in range(h-1):
        profY.append(img_airyY[y][0])
        yy[y] = img_airyY[y][0]

    xy = np.arange(0, h)
    areaY = integrate.simpson(yy, xy)

    yx = np.zeros(w, dtype=np.float32)
    for x in range(w-1):
        profX.append(img_airyX[x][0])
        yx[x] = img_airyX[x][0]

    xx = np.arange(0, w)
    areaX = integrate.simpson(yx, xx)

    return [profX, profY, areaY, areaX]


def HWHM(dataTemp):
    data = np.array(dataTemp)
    h2 = float((np.max(data)) - np.min(data))/2
    h, w = data.shape
    centroX = round((w-1)/2)
    centroY = round((h-1)/2)
    img_profileX = np.array(profile_dataX(data, centroY))
    img_profileY = np.array(profile_dataY(data, centroX))

    # Calcolo lungo X
    px1 = 0
    px2 = 0
    minPh2X = h2
    minNh2X = h2
    for k in range(len(img_profileX)-1):
        dX = (img_profileX[k+1][0]-img_profileX[k][0])
        dis = abs(img_profileX[k][0]-h2)
        if dX >= 0 and dis < minPh2X:
            minPh2X = dis
            px1 = k
        if dX < 0 and dis < minNh2X:
            minNh2X = dis
            px2 = k

    # Calcolo lungo Y
    py1 = 0
    py2 = 0
    minPh2Y = h2
    minNh2Y = h2
    for k in range(len(img_profileY)-1):
        dY = (img_profileY[k+1][0]-img_profileY[k][0])
        dis = abs(img_profileY[k][0]-h2)
        if dY >= 0 and dis < minPh2Y:
            minPh2Y = dis
            py1 = k
        if dY < 0 and dis < minNh2Y:
            minNh2Y = dis
            py2 = k

    return [abs(px1-px2)*pxToUm, abs(py1-py2)*pxToUm]


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
    if r < 1:
        r = 1
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
if conv == False:
    vol_airy, img_airy = ideal(img_avrg_small, lamb, 1)
else:
    vol_airy, img_airy = convolutionAxS(img_avrg_small)

print('\n-----Info-----')
print('Vol airy:\t',  round(vol_airy))

img_airyX, img_airyY, areaY, areaX = ideal_profile(img_airy)

print('Area teo(Y):\t', round(areaY, 2),
      '\nArea teo(X):\t', round(areaX, 2))

cp, img_norm = actual(img_avrg_small, vol_airy)
cpY, cpX, profX, profY = actual_profile(img_avrg_small, areaY, areaX)
print('\n-----Risultato-----')
hwx, hwy = HWHM(img_avrg_small)
print('HWHM (X):\t', hwx, '\nHWHM (Y):\t', hwy)

print('S. ratio(V):\t', round(cp, 3), '\nS. ratio(X):\t',
      round(cpX, 3), '\nS. ratio(Y):\t', round(cpY, 3))


if saveData == True:
    print('\nSalvo i dati')
    profileX = profile_dataX(img_avrg, CyMax)
    profileY = profile_dataY(img_avrg, CxMax)

    with open('result/profile_fuoco_average_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(profileX)):
            file.write(str((i-CxMax)*pxToUm).replace('.', sepDec) + '\t' +
                       str(profileX[i][0]).replace('.', sepDec) + '\n')

    with open('result/profile_fuoco_average_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(profileY)):
            file.write(str((i-CyMax)*pxToUm).replace('.', sepDec) + '\t' +
                       str(profileY[i][0]).replace('.', sepDec) + '\n')

    midS = round(Size/2)
    with open('result/profile_norm_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(profY)):
            file.write(str((i-midS)*pxToUm).replace('.', sepDec) +
                       '\t' + str(profY[i]).replace('.', sepDec) + '\n')

    with open('result/profile_norm_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(profX)):
            file.write(str((i-midS)*pxToUm).replace('.', sepDec) +
                       '\t' + str(profX[i]).replace('.', sepDec) + '\n')

    with open('result/profile_teo_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(img_airyX)):
            file.write(str((i-midS)*pxToUm).replace('.', sepDec) +
                       '\t' + str(img_airyX[i]).replace('.', sepDec) + '\n')

    with open('result/profile_teo_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(img_airyY)):
            file.write(str((i-midS)*pxToUm).replace('.', sepDec) +
                       '\t' + str(img_airyY[i]).replace('.', sepDec) + '\n')

    with open('result/parametri.dat', 'w') as file:
        file.write('Result:\n' + 'Strehl ratio(V):\t' + str(cp).replace('.', sepDec) + '\nStrehl ratio(X):\t' +
                   str(cpX).replace('.', sepDec) + '\nStrehl ratio(Y):\t' + str(cpY).replace('.', sepDec) + '\n')
        file.write('HWHM (X):\t' + str(hwx).replace('.', sepDec) +
                   '\nHWHM (Y):\t' + str(hwy).replace('.', sepDec) + '\n')
        file.write('\nInfo:' + '\nDiametro(m):\t' + str(diameter).replace('.', sepDec) + '\npxToUm(m/px):\t' +
                   str(pxToUm) + '\nL. d\'onda(m):\t' + str(lamb).replace('.', sepDec) + '\nFuoco(m):\t' + str(fuoco).replace('.', sepDec))
        file.write('\nCentro X(px):\t' + str(CxMax).replace('.', sepDec) +
                   '\nCentro Y(px):\t' + str(CyMax).replace('.', sepDec))

if show == True:
    print('Mostro i dati')
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)

    axs = axs.ravel()
    fig.suptitle('Grafici fuoco della metalente')
    midSize = pxToUm*(Size/2)
    x = np.linspace(-midSize, midSize, Size-1)

    im1 = axs[0].imshow(img_norm, vmin=0, vmax=1,
                        extent=(-midSize, midSize, - midSize, midSize))
    axs[0].set_xlabel('x(m)')
    axs[0].set_ylabel('y(m)')
    axs[0].title.set_text('Immagine del fuoco normalizzata')
    cbar0 = fig.colorbar(im1, ax=axs[0])
    cbar0.set_label('I(au)')

    im2 = axs[1].imshow(img_airy, vmin=0, vmax=1,
                        extent=(-midSize, midSize, - midSize, midSize))
    axs[1].set_xlabel('x(m)')
    axs[1].set_ylabel('y(m)')
    axs[1].title.set_text('Immagine disco di Airy attesa')
    cbar1 = fig.colorbar(im2, ax=axs[1])
    cbar1.set_label('I(au)')

    axs[2].plot(x, profX, color='blue', label='Profilo in X')
    axs[2].plot(x, img_airyX, color='orange', label='Disco di Airy')
    axs[2].set_xlabel('x(m)')
    axs[2].set_ylabel('I (au)')
    axs[2].title.set_text('Profilo fuoco lungo X')
    axs[2].legend()

    axs[3].plot(x, profY, color='blue', label='Profilo in Y')
    axs[3].plot(x, img_airyX, color='orange', label='Disco di Airy')
    axs[3].set_xlabel('x(m)')
    axs[3].set_ylabel('I(au)')
    axs[3].title.set_text('Profilo fuoco lungo Y')
    axs[3].legend()

    plt.show()

print('-----Fine-----')
