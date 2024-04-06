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

# Variabili globali da settare
show = False  # Mostra l'immagine con il punto di massimo trovato in rosso
saveData = True  # Se su True salva i dati
pxToUm = 7.55e-8
diameter = .01
Imin = 0
errore = 3
n_aria = 1.0002926
# Variable input
name_dataset = str(input('Nome dataset:'))
lamb = float(input('Lunghezza d\'onda(nm):'))*1e-9
fuoco = float(input('Distanza fuoco(mm):'))*1e-3

# Risoluzione immagini
width_px = 1280
height_px = 720

r_tp = diameter/2
NA = n_aria * r_tp / (math.sqrt((r_tp*r_tp)+(fuoco*fuoco)))
j_bessel2 = 7.016
Size = round(j_bessel2*lamb/NA*(1/pxToUm))
if (Size % 2 == 0):
    Size = Size+1
# Variabili globali
img_avrg = []
img_norm = []
conv = False
sepDec = '.'
spectrum = []
N_img_avrg = 0
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
    global N_img_avrg
    listNameImg = os.listdir(os.path.dirname(
        os.path.abspath(__file__)) + '/assets')
    h, w = cv2.imread('assets/'+listNameImg[0], 0).shape
    img_avrg = np.zeros((h, w), np.uint8)
    for k in range(len(listNameImg)):
        if k > avrgBreak:
            break
        N_img_avrg = N_img_avrg+1
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
    vol_airy = 0
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
            if (r <= h):
                vol_airy = vol_airy+airy_disc[y, x]

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
print('\n**---Inizio calcolo parametri---**')
dataTemp = np.array(img_avrg)
h, w = dataTemp.shape

img_avrg_small = np.zeros((Size, Size), dtype=np.float32)
for y in range(Size):
    for x in range(Size):
        Y = CyMax-y+round(Size/2)
        X = CxMax-x+round(Size/2)
        img_avrg_small[y, x] = img_avrg[Y, X]

vol_airy, img_airy = ideal(img_avrg_small, lamb, 1)


print('\n-----Info-----')
print('NA:\t\t',  round(NA, 3))
print('Size:\t\t',  Size)
print('Vol airy:\t',  round(vol_airy))

img_airyX, img_airyY, areaY, areaX = ideal_profile(img_airy)

print('Area teo(Y):\t', round(areaY, 2),
      '\nArea teo(X):\t', round(areaX, 2))

cp, img_norm = actual(img_avrg_small, vol_airy)
cpY, cpX, profX, profY = actual_profile(img_avrg_small, areaY, areaX)
print('\n-----Risultato-----')
hwx, hwy = HWHM(img_avrg_small)
print('HWHM (X):\t', hwx, '\nHWHM (Y):\t', hwy)
hwxT, hwyT = HWHM(img_airy)
print('HWHM teo(X):\t', hwxT, '\nHWHM teo(Y):\t', hwyT)

print('S. ratio(V):\t', round(cp, 3), '\nS. ratio(X):\t',
      round(cpX, 3), '\nS. ratio(Y):\t', round(cpY, 3))


if saveData == True:
    print('\nSalvo i dati')
    profileX = profile_dataX(img_avrg, CyMax)
    profileY = profile_dataY(img_avrg, CxMax)

    with open('result/'+name_dataset+'_profile_fuoco_average_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(profileX)):
            file.write(str((i-CxMax)*pxToUm).replace('.', sepDec) + '\t' +
                       str(profileX[i][0]).replace('.', sepDec) + '\n')

    with open('result/'+name_dataset+'_profile_fuoco_average_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(profileY)):
            file.write(str((i-CyMax)*pxToUm).replace('.', sepDec) + '\t' +
                       str(profileY[i][0]).replace('.', sepDec) + '\n')

    midS = round(Size/2)
    with open('result/'+name_dataset+'_profile_norm_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(profY)):
            file.write(str((i-midS)*pxToUm).replace('.', sepDec) +
                       '\t' + str(profY[i]).replace('.', sepDec) + '\n')

    with open('result/'+name_dataset+'_profile_norm_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(profX)):
            file.write(str((i-midS)*pxToUm).replace('.', sepDec) +
                       '\t' + str(profX[i]).replace('.', sepDec) + '\n')

    with open('result/'+name_dataset+'_profile_teo_X.dat', 'w') as file:
        file.write('X\tI\n')
        for i in range(len(img_airyX)):
            file.write(str((i-midS)*pxToUm).replace('.', sepDec) +
                       '\t' + str(img_airyX[i]).replace('.', sepDec) + '\n')

    with open('result/'+name_dataset+'_profile_teo_Y.dat', 'w') as file:
        file.write('Y\tI\n')
        for i in range(len(img_airyY)):
            file.write(str((i-midS)*pxToUm).replace('.', sepDec) +
                       '\t' + str(img_airyY[i]).replace('.', sepDec) + '\n')

    with open('result/'+name_dataset+'_parametri.dat', 'w') as file:
        file.write('Result:\n' + 'Strehl ratio(V):\t' + str(cp).replace('.', sepDec) + '\nStrehl ratio(X):\t' +
                   str(cpX).replace('.', sepDec) + '\nStrehl ratio(Y):\t' + str(cpY).replace('.', sepDec) + '\n')
        file.write('HWHM (X):\t' + str(hwx).replace('.', sepDec) +
                   '\nHWHM (Y):\t' + str(hwy).replace('.', sepDec) + '\n')
        file.write('\nInfo:' + '\nDiametro(m):\t' + str(diameter).replace('.', sepDec) + '\npxToUm(m/px):\t' +
                   str(pxToUm) + '\nL. d\'onda(m):\t' + str(lamb).replace('.', sepDec) + '\nFuoco(m):\t' + str(fuoco).replace('.', sepDec))
        file.write('\nCentro X(px):\t' + str(CxMax).replace('.', sepDec) +
                   '\nCentro Y(px):\t' + str(CyMax).replace('.', sepDec))
        file.write('\nNA:\t' + str(NA).replace('.', sepDec) +
                   '\nN img.:\t' + str(N_img_avrg))


if saveData == True or show == True:
    dpi = 100
    width_inch = width_px / dpi
    height_inch = height_px / dpi
    midSize = pxToUm*(Size/2)
    x = np.linspace(-midSize, midSize, Size-1)

    if show == True:
        print('Mostro i dati')
        fig, axs = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=False, figsize=(width_inch, height_inch))
        axs = axs.ravel()
        fig.suptitle('Grafici fuoco della metalente')

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
    if saveData == True:
        print('Salvo i grafici')
        fig, axs = plt.subplots(
            nrows=1, ncols=2, sharex=True, sharey=False, figsize=(width_inch, height_inch))
        axs = axs.ravel()
        # fig.suptitle('Grafici fuoco della metalente')

        im1 = axs[0].imshow(img_norm, vmin=0, vmax=1,
                            extent=(-midSize, midSize, - midSize, midSize))
        axs[0].set_xlabel('x(m)', fontsize=14)
        axs[0].set_ylabel('y(m)', fontsize=14)
        axs[0].set_title('Immagine fuoco normalizzato', fontsize=16)
        # Cambia dimensione testo numeri asse x
        axs[0].tick_params(axis='x', labelsize=12)
        # Cambia dimensione testo numeri asse y
        axs[0].tick_params(axis='y', labelsize=12)

        im2 = axs[1].imshow(img_airy, vmin=0, vmax=1,
                            extent=(-midSize, midSize, - midSize, midSize))
        axs[1].set_xlabel('x(m)', fontsize=14)
        axs[1].set_ylabel('y(m)', fontsize=14)
        axs[1].set_title('Disco di Airy atteso', fontsize=16)
        axs[1].tick_params(axis='both', labelsize=12)
        
        position=fig.add_axes([0.92,0.16,0.02,0.68])
        cbar = fig.colorbar(im1, ax=axs.ravel().tolist(),  cax=position)
        cbar.set_label('I(a.u.)', fontsize=14)

        plt.subplots_adjust(left=0.07, right=0.9, top=0.9, bottom=0.1)
        plt.savefig('result/'+name_dataset+'_disc_reale_norm.png', dpi=dpi)
        plt.close()

        # 2
        plt.figure(figsize=(width_inch, height_inch))
        plt.plot(x, profX, color='blue', label='Profilo in X')
        plt.plot(x, img_airyX, color='orange', label='Disco di Airy')
        step_axis = 11
        x_axis = np.zeros(step_axis)
        min_axis = np.min(x)
        max_axis = np.max(x)
        d_axis = (max_axis - min_axis)/(step_axis-1)
        for i in range(step_axis):
            x_axis[i] = np.min(x)+i*d_axis
        plt.xticks(x_axis)
        plt.tick_params(axis='both', labelsize=12)
        plt.xlabel('x(m)', fontsize=14)
        plt.ylabel('I(a.u.)', fontsize=14)
        plt.title('Profilo fuoco lungo X', fontsize=16)
        plt.savefig('result/'+name_dataset+'_profilo_x.png', dpi=dpi)
        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
        plt.close()

        # 3
        plt.figure(figsize=(width_inch, height_inch))
        plt.plot(x, profY, color='blue', label='Profilo in Y')
        plt.plot(x, img_airyY, color='orange', label='Disco di Airy')
        step_axis = 11
        x_axis = np.zeros(step_axis)
        min_axis = np.min(x)
        max_axis = np.max(x)
        d_axis = (max_axis - min_axis)/(step_axis-1)
        for i in range(step_axis):
            x_axis[i] = np.min(x)+i*d_axis
        plt.xticks(x_axis)
        plt.xlabel('y(m)', fontsize=14)
        plt.ylabel('I(a.u.)', fontsize=14)
        plt.title('Profilo fuoco lungo Y', fontsize=16)
        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
        plt.savefig('result/'+name_dataset+'_profilo_y.png', dpi=dpi)
        plt.close()

print('-----Fine-----')
