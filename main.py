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

# Imax = 1023
Imin = 0  # ntensità miinima
errore = 3  # Errore sull'intensiità, i punti di massimo vengo cercati considerando l'errore
showM = False
saveData = True

# Cerco la lista di dutte le immagini da analizzare
listImage = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '/assets')

allProfileX = []
allProfileY = []
allCenter = []

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

        print(idTh + '-> Massimo trovato ->  I:' +
              str(maxP[0]) + ', Y:' + str(maxP[1]) + ', X:' + str(maxP[2]))
        allCenter.append([maxP[1], maxP[2]])

        if showM == True:
            red = (0, 0, 255)
            point = [maxP[1], maxP[2]]
            show_img_with_point(image, point, red)

        return [profile_dataX(image, maxP[1]), profile_dataY(image, maxP[2]), tempC_start, idTh]

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

    print('-------------Fine--------------')
    print('---------Salvo i dati----------')

    # Salvo su un file le coordinate di tutti i centri
    CxMed = 0
    Cx = []
    Cy = []
    CyMed = 0
    if saveData == True:
        with open('result/'+dateNow+'_center.dat', 'w') as file:
            for elemento in allCenter:
                Cx.append(elemento[1])
                Cy.append(elemento[0])
                CxMed += elemento[1]
                CyMed += elemento[0]
            Xx = np.arange(len(listImage)).reshape(-1, 1)
            Yx = np.array(Cx)
            model = LinearRegression()
            model.fit(Xx, Yx)
            slopeX = model.coef_[0]
            interceptX = model.intercept_
            # Sostituisci 'model' con il tuo modello e 'X' con i dati utilizzati per l'addestramento
            y_predetti_X = model.predict(Xx)
            coefficiente_r2_X = r2_score(Yx, y_predetti_X)

            Xy = np.arange(len(listImage)).reshape(-1, 1)
            Yy = np.array(Cy)
            model.fit(Xy, Yy)
            slopeY = model.coef_[0]
            interceptY = model.intercept_
            # Sostituisci 'model' con il tuo modello e 'X' con i dati utilizzati per l'addestramento
            y_predetti_Y = model.predict(Xy)
            coefficiente_r2_Y = r2_score(Yy, y_predetti_Y)

            file.write('Centri delle immagini' + '\n')
            file.write(f"m(Cx): {round(slopeX, 4)}\t")
            file.write(f"q(Cx): {round(interceptX, 4)}\t")
            file.write(f"R^2(Cx): {round(coefficiente_r2_X, 4)}\n")
            file.write(f"m(Cy): {round(slopeY, 4)}\t")
            file.write(f"q(Cy): {round(interceptY, 4)}\t")
            file.write(f"R^2(Cy): {round(coefficiente_r2_Y, 4)}\n")
            file.write('X\tY\n')
            for elemento in allCenter:
                file.write(str(elemento[1]) + '\t' + str(elemento[0]) + '\n')

    # Calcolo i valori medi
    CxMed = round(CxMed/len(allCenter))
    CyMed = round(CyMed/len(allCenter))

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

    for y in range(hx):  # Lungo X
        for x in range(wx):
            I = allProfileX[y][x][0]
            imgX[y, x] = (0, 0, I)  # (I, I, I)
            x_shift = x-(allCenter[y][1]-CxMed)
            if x_shift < 0:
                x_shift = 0
            if x_shift > wx-1:
                x_shift = wx-1
            imgX_shift[y, x_shift] = (0, 0, I)

    for y in range(hy):  # Lungo y
        for x in range(wy):
            I = allProfileY[x][y][0]
            imgY[y, x] = (0, 0, I)  # (I, I, I)
            cv2.flip(imgY, 0)
            y_shift = y-(allCenter[x][0]-CyMed)
            if y_shift < 0:
                y_shift = 0
            if y_shift > hy-1:
                y_shift = hy-1
            imgY_shift[y_shift, x] = (0, 0, I)
            cv2.flip(imgY_shift, 0)

    if saveData == True:
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

    if saveData == True:  # Salvo l'immagine formata da tutti i profili
        cv2.imwrite('result/'+dateNow+'_imageX_shift.jpg', imgX_shift)
        cv2.imwrite('result/'+dateNow+'_imageX.jpg', imgX)
        cv2.imwrite('result/'+dateNow+'_imageY_shift.jpg', imgY_shift)
        cv2.imwrite('result/'+dateNow+'_imageY.jpg', imgY)

    tempo_fine = time.time()
    tempo_trascorso = tempo_fine - tempo_inizio
    print('Tempo impiegato:' + str(round((tempo_trascorso/60), 2)) + 'm')


# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
