import cv2
import numpy as np
import pickle
import os
import datetime

from funzioni import *

# Inizializzo le variilii principali
dateNow = (datetime.datetime.now()).strftime("%d-%m-%Y_%H-%M-%S")

# Imax = 1023
Imin = 0  # ntensità miinima
errore = 1  # Errore sull'intensiità, i punti di massimo vengo cercati considerando l'errore
showM = False

# Cerco la lista di dutte le immagini da analizzare
listImage = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '/assets')

allProfileX = []
allCenter = []
for name in listImage:
    image = cv2.imread('assets/'+name, 0)

    # Cerco se ci sono punti di massimo
    maxP = []
    puntiMax = pMax(image, Imin, errore)
    if len(puntiMax) > 1:
        print('In ' + name + ' ho trovato ' +
              str(len(puntiMax)) + ' punti di massimo')
        print('Ricerca massimo: 0%')
        pMaxFiltrati = filtra_vicinato(puntiMax, image, 1, len(puntiMax))
        if len(pMaxFiltrati) == 1:
            maxP = pMaxFiltrati[0]
        else:
            print('Errore nella ricerca del massimo')
    else:
        if len(puntiMax) == 1:
            print('In ' + name + ' ho trovato ' +
                  str(len(puntiMax)) + ' come massimo')
            print('Ricerca massimo: 0%')
            maxP = puntiMax[0]
        else:
            print('Non ho trovato massimi')

    print('Massimo trovato=  I:' +
          str(maxP[0]) + ', Y:' + str(maxP[1]) + ', X:' + str(maxP[2]))
    print()
    allCenter.append([maxP[1], maxP[2]])

    if showM == True:
        red = (0, 0, 255)
        point = [maxP[1], maxP[2]]
        show_img_with_point(image, point, red)

    profX = profile_dataX(image, maxP[1])
    profY = profile_dataY(image, maxP[2])
    allProfileX.append(profX)


print('----------------------------')

# Salvol su un file le coordinate di tutti i centri
with open('result/'+dateNow+'_center.dat', 'w') as file:
    file.write('Centri delle immagini' + '\n')
    file.write('Y\tX\n')
    for elemento in allCenter:
        file.write(str(elemento[0]) + '\t' + str(elemento[1]) + '\n')

# Disegno l'imaggine costruirta da tutti i profili
h = len(listImage)
w = len(allProfileX[0])
img = np.zeros((h, w, 3), np.uint8)
print('Immagine: ' + str(h) + 'x' + str(w))
for y in range(h):
    for x in range(w):
        I = allProfileX[y][x][0]
        # print('I:' + str(I)+'; Y:' + str(y)+'; X:' + str(x))
        img[y, x] = (I, I, I)
# Salvo l'immagine formata da tutti i profili

cv2.imwrite('result/'+dateNow+'_image.jpg', img)
# cv2.imshow('Binary', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# prit_profile('porfX_'+name, profX,
#              'Profilo che passa dal centro lungo l asse X, nome immagine: ' + name)
# prit_profile('porfY_'+name, profY,
#              'Profilo che passa dal centro lungo l asse Y, nome immagine: ' + name)


# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
