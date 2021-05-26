import os

import cv2 as cv
import matplotlib.pyplot as plt


class Representar:
    def __init__(self, path, color=(255, 255, 255)):
        self.color = color
        self.path = path

        self.font = cv.FONT_HERSHEY_COMPLEX

        os.makedirs(path, exist_ok=True)

    def representar(self, detecciones):
        i = 0
        path = detecciones[0].path
        src = cv.imread(path)
        while i < len(detecciones):
            deteccion = detecciones[i]

            if deteccion.path != path:
                path_name = path.split('/')
                cv.imwrite(self.path + path_name[len(path_name)-1], src)

                path = deteccion.path
                src = cv.imread(path)

            (x, y, w, h) = deteccion.rectangle
            # Poner un rectángulo alrededor de la detección
            cv.rectangle(src, (x, y), (x + w, y + h), self.color, 2)
            # Poner un texto que expresa el tipo de detección sobre el rectángulo
            #                                                                                           posición                tamaño fuente            grosor fuente
            cv.putText(src, deteccion.name + " (" + str(int(deteccion.sub_type[0])) + ")", (int(x - w * 0.20), y - 4), self.font, w * 0.009, (0, 0, 0),       2, cv.LINE_AA)
            cv.putText(src, deteccion.name + " (" + str(int(deteccion.sub_type[0])) + ")", (int(x - w * 0.20), y - 4), self.font, w * 0.009, (255, 255, 255), 1, cv.LINE_AA)

            i += 1

        path_name = path.split('/')
        cv.imwrite(self.path + path_name[len(path_name) - 1], src)
