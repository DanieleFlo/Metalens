import cv2
import numpy as np
import pickle
import os
import datetime
import time
import multiprocessing
import matplotlib.pyplot as plt
import re
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from funzioni import *

# Inizializzo le varili principali
dateNow = (datetime.datetime.now()).strftime("%d-%m-%Y_%H-%M-%S")
tempo_inizio = time.time()

# ----------setup----------
Imax = 254  # Intensità massima dopo il quale ha saturato
Imin = 0  # Intensità minima
errore = 3  # Errore sull'intensiità, i punti di massimo vengo cercati considerando l'errore
showM = False  # Mostra l'immagine con il punto di massimo trovato in rosso
saveData = True  # Se su True salva i dati
color_light = 'red'

pxToUm = 7.55e-8  # Fattore di conversio da px a metri
# -------Fine setup-------

stepZu = float(input('Step Z(mm):'))
while True:
    sepDec = input('Separazione decimale (. o ,): ')
    if sepDec == ',' or sepDec == '.':
        break

# Cerco la lista di dutte le immagini da analizzare
listImage = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '/assets')
listImage = sorted(listImage, key=lambda s: [int(
    x) if x.isdigit() else x for x in re.split('([0-9]+)', s)])

allProfileX = []
allProfileY = []
allCenter = []
Isaturo = 180
stepZ = stepZu/1000
pxToStep = stepZ/pxToUm

if __name__ == "__main__":
    def mainF(name, index):
        # Carico le immagini
        idTh = str(index) + 'th '
        print(name + ' -> ' + idTh)
        tempC_start = time.time()
        image = cv2.imread('assets/'+name, 0)

        # Cerco se ci sono punti di massimo
        maxP = []
        puntiMax = pMax(image, Imin, errore)
        if len(puntiMax) > 1:
            print('In ' + idTh + ' ho trovato ' +
                  str(len(puntiMax)) + ' punti di massimo')
            print(idTh + '-> Ricerca massimo: 0%')
            r = round(math.sqrt(len(puntiMax)/6.2931)/2)
            if r < 1:
                r = 1
            pMaxFiltrati = filtra_vicinato(
                puntiMax, image, r, len(puntiMax), idTh)
            if len(pMaxFiltrati) == 1:
                maxP = pMaxFiltrati[0]
            else:
                print('Errore nella ricerca del massimo')
        else:
            if len(puntiMax) == 1:
                print('In ' + name + ' ho trovato ' +
                      str(len(puntiMax)) + ' punto di massimo')
                print(idTh + '-> Ricerca massimo: 0%')
                maxP = puntiMax[0]
                print(idTh + '-> Ricerca massimo: 100%')
            else:
                print('Non ho trovato massimi')
        CxMax = maxP[2]
        CyMax = maxP[1]
        print(idTh + '-> Massimo trovato ->  I:' +
              str(maxP[0]) + ', Y:' + str(CyMax) + ', X:' + str(CxMax))

        if showM == True:
            red = (0, 0, 255)
            point = [CyMax, CxMax]
            show_img_with_point(image, point, red)

        return [profile_dataX(image, CyMax), profile_dataY(image, CxMax), [CyMax, CxMax], tempC_start, idTh]

    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(lambda args: mainF(
            *args), [(item, index) for index, item in enumerate(listImage)])

        for result in results:
            allProfileX.append(result[0])
            allProfileY.append(result[1])
            allCenter.append(result[2])
            tempC_end = time.time()
            print(result[4] + '-> Tempo analisi: ' +
                  str(round((tempC_end-result[3])/60, 2)) + 'm')

    print()
    print('-----Fine ricerca massimi------')
    print('---------Salvo i dati----------')

    Cx = []  # Lista centri lungo X
    Cy = []  # Lista centri lungo Y
    mx = 0  # Coefficiente angolare dell'andamento dei centri lungo X
    qx = 0  # Intercetta andamento dei centri lungo X
    my = 0  # Coefficiente angolare dei centri lungo Y
    qy = 0  # Intercetta andamento dei centri lungo Y

    if saveData == True:  # Salvo su un file le coordinate di tutti i centri
        with open('result/'+dateNow+'_center.dat', 'w') as file:
            print(
                'Salvo le coordinate di utti i centri e l\'interpolazione del movimento')
            for elemento in allCenter:
                Cx.append(elemento[1])
                Cy.append(elemento[0])
            Xx = np.arange(len(listImage)).reshape(-1, 1)
            Yx = np.array(Cx)
            model = LinearRegression()
            model.fit(Xx, Yx)
            mx = model.coef_[0]
            qx = model.intercept_
            # Sostituisci 'model' con il tuo modello e 'X' con i dati utilizzati per l'addestramento
            y_predetti_X = model.predict(Xx)
            coefficiente_r2_X = r2_score(Yx, y_predetti_X)

            Xy = np.arange(len(listImage)).reshape(-1, 1)
            Yy = np.array(Cy)
            model.fit(Xy, Yy)
            my = model.coef_[0]
            qy = model.intercept_
            # Sostituisci 'model' con il tuo modello e 'X' con i dati utilizzati per l'addestramento
            y_predetti_Y = model.predict(Xy)
            coefficiente_r2_Y = r2_score(Yy, y_predetti_Y)

            file.write('Centri delle immagini' + '\n')
            file.write(f"m(Cx): {round(mx, 4)}\t")
            file.write(f"q(Cx): {round(qx, 4)}\t")
            file.write(f"R^2(Cx): {round(coefficiente_r2_X, 4)}\n")
            file.write(f"m(Cy): {round(my, 4)}\t")
            file.write(f"q(Cy): {round(qy, 4)}\t")
            file.write(f"R^2(Cy): {round(coefficiente_r2_Y, 4)}\n")
            file.write('X\tY\n')
            for elemento in allCenter:
                file.write(str(elemento[1]) + '\t' + str(elemento[0]) + '\n')

    # Disegno l'imaggine costruirta da tutti i profili
    hx = len(listImage)
    wx = len(allProfileX[0])
    hy = len(allProfileY[0])
    wy = len(listImage)

    imgX = np.zeros((hx, wx, 3), np.uint8)
    imgX_shift = np.zeros((hx, wx, 3), np.uint8)
    imgY = np.zeros((hy, wy, 3), np.uint8)
    imgY_shift = np.zeros((hy, wy, 3), np.uint8)
    print('ImmagineX: ' + str(hx) + 'x' + str(wx))
    print('ImmagineY: ' + str(hy) + 'x' + str(wy))
    print('Rapporto px/step: ' + str(round(pxToStep, 3)))

    print('Creo l\'immagine dei profili lungo X')
    for y in range(hx):  # Lungo X
        for x in range(wx):
            I = allProfileX[y][x][0]
            imgX[y, x] = (Isaturo, Isaturo, I) if I >= Imax else (0, 0, I)

    print('Creo l\'immagine dei profili lungo Y')
    for y in range(hy):  # Lungo y
        for x in range(wy):
            I = allProfileY[x][y][0]
            imgY[y, x] = (Isaturo, Isaturo, I) if I >= Imax else (0, 0, I)
            cv2.flip(imgY, 0)

    # Creo le immagini corrette e shiftate
    print('Creo l\'immagine dei profili lungo X,Y corretta con i centri secondo le eq. e shiftati')
    CxMed = round(mx*(len(listImage)/2) + qx)
    CyMed = round(my*(len(listImage)/2) + qy)
    for n in range(len(listImage)):  # Lungo X
        name = listImage[n]
        image = cv2.imread('assets/'+name, 0)
        # Calcolo il punto medio in x presente sulla retta che passa per i centri
        xs = round(mx*n + qx)
        ys = round(my*n + qy)
        px = profile_dataX(image, ys)
        py = profile_dataY(image, xs)

        # Inserisco l'intensità per creare l'immagine dei profili in X e li shifto
        for x in range(wx):
            Is = px[x][0]
            x_shift = x-(xs-CxMed)
            if x_shift < 0:
                x_shift = 0
                Is = 0
            if x_shift > wx-1:
                x_shift = wx-1
                Is = 0
            imgX_shift[n, x_shift] = (
                Isaturo, Isaturo, Is) if Is >= Imax else (0, 0, Is)

        # Inserisco l'intensità per creare l'immagine dei profili in Y e li shifto
        for y in range(hy):
            Is = py[y][0]
            y_shift = y-(ys-CyMed)
            if y_shift < 0:
                y_shift = 0
                Is = 0
            if y_shift > hy-1:
                y_shift = hy-1
                Is = 0
            imgY_shift[y_shift, n] = (
                Isaturo, Isaturo, Is) if Is >= Imax else (0, 0, Is)
            cv2.flip(imgY_shift, 0)

    if saveData == True:
        print('Salvo i dati dei profili grezzi in X')
        with open('result/'+dateNow+'_imgX.dat', 'w') as file:
            file.write('Profili passanti dal centro lungo X' + '\n')
            for i in range(hx):
                file.write('X'+str(i+1)+'\tI'+str(i+1) + '\t')
            file.write('\n')

            for j in range(wx):
                for i in range(hx):
                    file.write(str(pxToUm*(j-allCenter[i][1])).replace('.', sepDec) + '\t' +
                               str(imgX[i, j, 2]) + '\t')

                file.write('\n')

        print('Salvo i dati dei profili shiftati in X')
        with open('result/'+dateNow+'_imgX_shift.dat', 'w') as file:
            file.write('Profili passanti dal centro lungo X' + '\n')
            for i in range(hx):
                file.write('X'+str(i+1)+'\tI'+str(i+1) + '\t')
            file.write('\n')

            for j in range(wx):
                for i in range(hx):
                    file.write(str(pxToUm*(j-CxMed)).replace('.', sepDec) + '\t' +
                               str(imgX_shift[i, j, 2]) + '\t')
                file.write('\n')

        print('Salvo i dati dei profili grezzi in Y')
        with open('result/'+dateNow+'_imgY.dat', 'w') as file:
            file.write('Profili passanti dal centro lungo Y' + '\n')
            for i in range(wy):
                file.write('Y'+str(i+1)+'\tI'+str(i+1) + '\t')
            file.write('\n')

            for i in range(hy):
                for j in range(wy):
                    file.write(str(pxToUm*(i-allCenter[j][0])).replace('.', sepDec) + '\t' +
                               str(imgY[i, j, 2]) + '\t')
                file.write('\n')

        print('Salvo i dati dei profili shiftati in Y')
        with open('result/'+dateNow+'_imgY_shift.dat', 'w') as file:
            file.write('Profili passanti dal centro lungo Y' + '\n')
            for i in range(wy):
                file.write('Y'+str(i+1)+'\tI'+str(i+1) + '\t')
            file.write('\n')

            for i in range(hy):
                for j in range(wy):
                    file.write(str(pxToUm*(i-CyMed)).replace('.', sepDec) + '\t' +
                               str(imgY_shift[i, j, 2]) + '\t')
                file.write('\n')

    if saveData == True:  # Salvo l'immagine formata da tutti i profili
        print('Salvo tutte le immagini')

        dpi = 100
        s_inch = 720 / dpi
        l_inch = 1920 / dpi

        colors = []
        for i in range(255):
            if (i != 254):
                if (color_light == 'red'):
                    colors.append((i/255, 0, 0))
                if (color_light == 'green'):
                    colors.append((0, i/255, 0))
                if (color_light == 'blue'):
                    colors.append((0, 0, i/255))
            else:
                colors.append((1, 1, 1))

        # Creazione della mappa di colori personalizzata
        custom_cmap = ListedColormap(colors)
        # ----------------------------------------------------------------

        imgX_real = cv2.resize(imgX_shift, (wx, round(hx*pxToStep)))
        imgX_real_rot90 = np.rot90(imgX_real)

        # Calcolo delle dimensioni dell'immagine in metri
        width_meters = imgX_real.shape[1] * pxToUm
        height_meters = imgX_real.shape[0] * pxToUm

        maxPxz = []
        imageXZ = imgX_shift[:, :, 2]
        puntiMaxXZ = pMax(imageXZ, Imin, errore)
        if len(puntiMaxXZ) > 1:
            print('In X-Z ho trovato ' +
                  str(len(puntiMaxXZ)) + ' punti di massimo')
            print('X-Z -> Ricerca massimo: 0%')
            r = round(math.sqrt(len(puntiMaxXZ)/6.2931)/2)
            if r < 1:
                r = 1
            pMaxFiltratiXZ = filtra_vicinato(
                puntiMaxXZ, imageXZ, r, len(puntiMaxXZ), 'X-Z')
            if len(pMaxFiltratiXZ) == 1:
                maxPxz = pMaxFiltratiXZ[0]
            else:
                print('Errore nella ricerca del massimo')
        else:
            if len(puntiMaxXZ) == 1:
                print('In X-Z ho trovato ' +
                      str(len(puntiMaxXZ)) + ' punto di massimo')
                print('X-Z-> Ricerca massimo: 0%')
                maxPxz = puntiMaxXZ[0]
                print('X-Z -> Ricerca massimo: 100%')
            else:
                print('Non ho trovato massimi')
        CxMaxXZ = maxPxz[2]*pxToUm
        CyMaxXZ = (hx-maxPxz[1])*stepZ

        plt.figure(figsize=(l_inch, s_inch))
        plt.imshow(imgX_real_rot90[:, :, 2] / 255, vmax=1, vmin=0, extent=[height_meters-CyMaxXZ, -CyMaxXZ, -CxMaxXZ, width_meters -
                   CxMaxXZ,], cmap=custom_cmap)
        plt.xlabel('Z(m)')
        plt.ylabel('X(m)')
        plt.title('Pofilo dell\'intesità lungo X-Z')
        plt.colorbar(label='I(a.u.)', orientation='horizontal')
        plt.grid(False)
        plt.savefig('result/'+dateNow+'_profilo_XZ.png', dpi=dpi)
        plt.close()

        # ****************Y-Axis***********************
        imgY_real = cv2.resize(imgY_shift, (round(wy*pxToStep), hy))
        width_meters = imgY_real.shape[1] * pxToUm
        height_meters = imgY_real.shape[0] * pxToUm

        maxPyz = []
        imageYZ = imgY_shift[:, :, 2]
        puntiMaxYZ = pMax(imageYZ, Imin, errore)
        if len(puntiMaxYZ) > 1:
            print('In Y-Z ho trovato ' +
                  str(len(puntiMaxYZ)) + ' punti di massimo')
            print('Y-Z -> Ricerca massimo: 0%')
            r = round(math.sqrt(len(puntiMaxYZ)/6.2931)/2)
            if r < 1:
                r = 1
            pMaxFiltratiYZ = filtra_vicinato(
                puntiMaxYZ, imageYZ, r, len(puntiMaxYZ), 'Y-Z')
            if len(pMaxFiltratiYZ) == 1:
                maxPyz = pMaxFiltratiYZ[0]
            else:
                print('Errore nella ricerca del massimo')
        else:
            if len(puntiMaxYZ) == 1:
                print('In Y-Z ho trovato ' +
                      str(len(puntiMaxYZ)) + ' punto di massimo')
                print('Y-Z-> Ricerca massimo: 0%')
                maxPyz = puntiMaxYZ[0]
                print('Y-Z -> Ricerca massimo: 100%')
            else:
                print('Non ho trovato massimi')
        CxMaxYZ = (wy-maxPyz[2])*stepZ
        CyMaxYZ = maxPyz[1]*pxToUm

        plt.figure(figsize=(l_inch, s_inch))
        plt.imshow(imgY_real[:, :, 2] / 255, vmax=1, vmin=0, extent=[width_meters -
                   CxMaxYZ, -CxMaxYZ, -CyMaxYZ, height_meters-CyMaxYZ], cmap=custom_cmap)
        plt.xlabel('Z(m)')
        plt.ylabel('Y(m)')
        plt.title('Pofilo dell\'intesità lungo Y-Z')
        plt.colorbar(label='I(a.u.)', orientation='horizontal')
        plt.grid(False)
        plt.savefig('result/'+dateNow+'_profilo_YZ.png', dpi=dpi)
        plt.close()

    tempo_fine = time.time()
    tempo_trascorso = tempo_fine - tempo_inizio
    print('Tempo impiegato:' + str(round((tempo_trascorso/60), 2)) + 'm')
    print('Ho finito!')


# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
