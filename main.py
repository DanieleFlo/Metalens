import cv2
import numpy as np
import pickle
import os
import datetime
import time
import multiprocessing

from funzioni import *

# Inizializzo le varili principali
dateNow = (datetime.datetime.now()).strftime("%d-%m-%Y_%H-%M-%S")
tempo_inizio = time.time()

# Imax = 1023
Imin = 0  # ntensità miinima
errore = 1  # Errore sull'intensiità, i punti di massimo vengo cercati considerando l'errore
showM = False
saveData = True

# Cerco la lista di dutte le immagini da analizzare
listImage = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '/assets')

allProfileX = []
allProfileY = []
allCenter = []

if __name__ == "__main__":
    for name in listImage:
        tempC_start = time.time()
        image = cv2.imread('assets/'+name, 0)

        # Cerco se ci sono punti di massimo
        maxP = []
        puntiMax = pMax(image, Imin, errore)
        if len(puntiMax) > 1:
            print('In ' + name + ' ho trovato ' +
                str(len(puntiMax)) + ' punti di massimo')
            print('Ricerca massimo: 0%')
            r = round(math.sqrt(len(puntiMax)/6.2931)/2)
            if r < 1:
                r = 1
            pMaxFiltrati = filtra_vicinato(puntiMax, image, r, len(puntiMax))
            if len(pMaxFiltrati) == 1:
                maxP = pMaxFiltrati[0]
            else:
                print('Errore nella ricerca del massimo')
        else:
            if len(puntiMax) == 1:
                print('In ' + name + ' ho trovato ' +
                    str(len(puntiMax)) + ' punto di massimo')
                print('Ricerca massimo: 0%')
                maxP = puntiMax[0]
                print('Ricerca massimo: 100%')
            else:
                print('Non ho trovato massimi')

        print('Massimo trovato ->  I:' +
            str(maxP[0]) + ', Y:' + str(maxP[1]) + ', X:' + str(maxP[2]))
        allCenter.append([maxP[1], maxP[2]])

        if showM == True:
            red = (0, 0, 255)
            point = [maxP[1], maxP[2]]
            show_img_with_point(image, point, red)

        profX = profile_dataX(image, maxP[1])
        profY = profile_dataY(image, maxP[2])
        allProfileX.append(profX)
        allProfileY.append(profY)
        tempC_end = time.time()
        print('Tempo interazione: ' + str(tempC_end-tempC_start) + 's')
        print()

    print('-------------Fine--------------')
    print('---------Salvo i dati----------')

    # Salvo su un file le coordinate di tutti i centri
    if saveData == True:
        with open('result/'+dateNow+'_center.dat', 'w') as file:
            file.write('Centri delle immagini' + '\n')
            file.write('X\tY\n')
            for elemento in allCenter:
                file.write(str(elemento[1]) + '\t' + str(elemento[0]) + '\n')

    # Disegno l'imaggine costruirta da tutti i profili
    hx = len(listImage)
    wx = len(allProfileX[0])
    hy = len(allProfileY[0])
    wy = len(listImage)
    imgX = np.zeros((hx, wx, 3), np.uint8)
    imgY = np.zeros((hy, wy, 3), np.uint8)
    print('ImmagineX: ' + str(hx) + 'x' + str(wx))
    print('ImmagineY: ' + str(hy) + 'x' + str(wy))

    for y in range(hx):  # Lungo X
        for x in range(wx):
            I = allProfileX[y][x][0]
            imgX[y, x] = (0, 0, I)#(I, I, I)

    for y in range(hy):  # Lungo y
        for x in range(wy):
            I = allProfileY[x][y][0]
            imgY[y, x] = (0, 0, I)#(I, I, I)
            cv2.flip(imgY, 0)


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
        cv2.imwrite('result/'+dateNow+'_imageX.jpg', imgX)
        cv2.imwrite('result/'+dateNow+'_imageY.jpg', imgY)

    tempo_fine = time.time()
    tempo_trascorso = tempo_fine - tempo_inizio
    print('Tempo impiegato:' + str(round(tempo_trascorso, 2)) + 's')


# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()