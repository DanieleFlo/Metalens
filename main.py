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

# Inizializzo le varili principali
dateNow = (datetime.datetime.now()).strftime("%d-%m-%Y_%H-%M-%S")
tempo_inizio = time.time()

# ----------setup----------
Imax = 254  # Intensità massima dopo il quale ha saturato
Imin = 0  # Intensità minima
errore = 3  # Errore sull'intensiità, i punti di massimo vengo cercati considerando l'errore
showM = False  # Mostra l'immagine con il punto di massimo trovato in rosso
saveData = True  # Se su True salva i dati
# -------Fine setup-------

# Cerco la lista di dutte le immagini da analizzare
listImage = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '/assets')

allProfileX = []
allProfileY = []
allCenter = []
Isaturo = 180

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
            if r < 10:
                r = 10
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
        allCenter.append([CyMax, CxMax])

        if showM == True:
            red = (0, 0, 255)
            point = [CyMax, CxMax]
            show_img_with_point(image, point, red)

        return [profile_dataX(image, CyMax), profile_dataY(image, CxMax), tempC_start, idTh]

    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(lambda args: mainF(
            *args), [(item, index) for index, item in enumerate(listImage)])

        for result in results:
            allProfileX.append(result[0])
            allProfileY.append(result[1])
            tempC_end = time.time()
            print(result[3] + '-> Tempo analisi: ' +
                  str(round((tempC_end-result[2])/60, 2)) + 'm')
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
    for n in range(len(listImage)):  # Lungo X
        name = listImage[n]
        image = cv2.imread('assets/'+name, 0)
        # Calcolo il punto medio in x presente sulla retta che passa per i centri
        CxMed = round(mx*(len(listImage)/2) + qx)
        CyMed = round(my*(len(listImage)/2) + qy)
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
                    file.write(str(j) + '\t' +
                               str(imgX[i, j][0]) + '\t')
                file.write('\n')

        print('Salvo i dati dei profili shiftati in X')
        with open('result/'+dateNow+'_imgX_shift.dat', 'w') as file:
            file.write('Profili passanti dal centro lungo X' + '\n')
            for i in range(hx):
                file.write('X'+str(i+1)+'\tI'+str(i+1) + '\t')
            file.write('\n')

            for j in range(wx):
                for i in range(hx):
                    file.write(str(j) + '\t' +
                               str(imgX_shift[i, j][0]) + '\t')
                file.write('\n')

        print('Salvo i dati dei profili grezzi in Y')
        with open('result/'+dateNow+'_imgY.dat', 'w') as file:
            file.write('Profili passanti dal centro lungo Y' + '\n')
            for i in range(wy):
                file.write('Y'+str(i+1)+'\tI'+str(i+1) + '\t')
            file.write('\n')

            for i in range(hy):
                for j in range(wy):
                    file.write(str(i) + '\t' +
                               str(imgY[i, j][0]) + '\t')
                file.write('\n')

        print('Salvo i dati dei profili shiftati in Y')
        with open('result/'+dateNow+'_imgY_shift.dat', 'w') as file:
            file.write('Profili passanti dal centro lungo Y' + '\n')
            for i in range(wy):
                file.write('Y'+str(i+1)+'\tI'+str(i+1) + '\t')
            file.write('\n')

            for i in range(hy):
                for j in range(wy):
                    file.write(str(i) + '\t' +
                               str(imgY_shift[i, j][0]) + '\t')
                file.write('\n')

    if saveData == True:  # Salvo l'immagine formata da tutti i profili
        print('Salvo tutte le immagini')
        cv2.imwrite('result/'+dateNow+'_imageX_shift.jpg', imgX_shift)
        cv2.imwrite('result/'+dateNow+'_imageX.jpg', imgX)
        cv2.imwrite('result/'+dateNow+'_imageY_shift.jpg', imgY_shift)
        cv2.imwrite('result/'+dateNow+'_imageY.jpg', imgY)

    tempo_fine = time.time()
    tempo_trascorso = tempo_fine - tempo_inizio
    print('Tempo impiegato:' + str(round((tempo_trascorso/60), 2)) + 'm')
    print('Ho finito zi!')


# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
