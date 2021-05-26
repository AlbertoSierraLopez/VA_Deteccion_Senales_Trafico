import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from Clase_Senal import Clase_Senal
from Enmascarar import Enmascarar


class Entrenar:

    def __init__(self, vers, directory, dimensiones=(25, 25)):
        self.vers = vers
        self.directory = directory
        self.dimensiones = dimensiones

        self.enmascararador = Enmascarar()

        self.prohibicion = Clase_Senal(['00', '01', '02', '03', '04', '05', '07', '08', '09', '10', '15', '16'], 'prohibicion', 1)
        self.peligro     = Clase_Senal(['11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'], 'peligro', 2)
        self.stop        = Clase_Senal(['14'], 'stop', 3)
        self.ceda        = Clase_Senal(['13'], 'ceda_el_paso', 4)
        self.paso        = Clase_Senal(['17'], 'prohibido_el_paso', 5)

        if vers > 0:
            self.senales = [self.prohibicion, self.peligro, self.stop, self.ceda, self.paso]
        else:
            self.senales = [self.prohibicion, self.peligro, self.stop]

        self.get_avg_images()


    # Crear imágenes medias
    def get_avg_images(self):

        for clase_senal in self.senales:
            images = []
            for folder in clase_senal.directories:
                # Iterar sobre todas las carpetas de un tipo de señal
                for file in os.listdir(self.directory + folder):
                    # Iterar sobre todas las imágenes de una carpeta
                    path = self.directory + str(folder) + "/" + file
                    src = cv.imread(path)
                    img = cv.cvtColor(src, cv.COLOR_BGR2RGB)

                    # Cada imagen se redimensiona a 30x30 y se guarda en images
                    img_resize = cv.resize(img, self.dimensiones, interpolation=cv.INTER_LINEAR)  # Interpolación linear
                    images.append(img_resize)

            # Calcular la imagen media
            avg_img = np.zeros((self.dimensiones[0], self.dimensiones[1], 3), dtype=np.float64)  # tipo 'float64' (o 'float') para que no se desborde al hacer la suma
            for img in images:
                avg_img += img
            avg_img = avg_img / len(images)  # Esto deja decimales en la matriz
            avg_img = avg_img.astype(np.uint8)  # Como los decimales no son compatibles con la imagen, hay que pasarlos a 'int8' (o 'int', pero 'int8' solo deja numeros de 0 a 255, es mejor)

            if self.vers > 0:   # Versión mejorada
                avg_mask = self.enmascararador.enmascarar(src_img=avg_img)
            else:               # Versión básica
                avg_mask = self.enmascararador.enmascarar_sin_dilatar(src_img=avg_img)

            # plt.imshow(avg_mask, 'gray')
            # plt.show()

            clase_senal.mask = avg_mask
            clase_senal.pixel_count = cv.sumElems(avg_mask)[0] / 255

    def get_clases(self):
        return self.senales
