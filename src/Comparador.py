import cv2 as cv
import numpy as np

from Enmascarar import Enmascarar
from skimage.feature import local_binary_pattern


class Comparador:

    def __init__(self, vers, clases_senal, dimensiones=(25, 25), descriptor_type='hog'):
        self.vers = vers
        self.clases_senal = clases_senal
        self.dimensiones = dimensiones

        self.descriptor_type = descriptor_type
        self.hog = self.get_hog()

        self.enmascararador = Enmascarar()


    def get_hog(self):
        winSize = (30, 30)
        blockSize = (10, 10)
        blockStride = (5, 5)
        cellSize = (5, 5)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        # gammaCorrection = 1
        gammaCorrection = False
        nlevels = 64
        signedGradient = True

        return cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)


    def detectar_Hough(self, img):
        cnts, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            epsilon = 0.01 * cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, epsilon, True)
            return (len(approx) == 3) or (len(approx) == 8) or (len(approx) > 10)


    def saturar(self, img):
        # Aumentar saturación para facilitar la detección del rojo
        (H, S, V) = cv.split(img)
        S = cv.multiply(S, 1.2)
        V = cv.multiply(V, 1.2)

        return cv.merge([H, S, V])


    def calculo_score_enunciado(self, clase_senal, deteccion):
        mask_sum = clase_senal.pixel_count
        det_sum = cv.sumElems(deteccion.mask)[0] / 255

        multiplication = cv.multiply(clase_senal.mask, deteccion.mask)
        mult_sum = cv.sumElems(multiplication)[0] / 255

        score = round(mult_sum / mask_sum * 100, 2)
        return score


    def calculo_score_doble_for(self, clase_senal, deteccion):
        mask_sum = clase_senal.pixel_count
        mask_signal = clase_senal.mask

        w_count = 0
        b_count = 0
        row = deteccion.mask.shape[0]
        col = deteccion.mask.shape[1]
        total = row * col
        for i in range(row):
            for j in range(col):
                if deteccion.mask[i, j] == mask_signal[i, j]:
                    if deteccion.mask[i, j] == 0:
                        b_count += 1
                    else:
                        w_count += 1

        score = 0.0
        if w_count >= mask_sum / 2 and b_count >= (total - mask_sum) / 2:
            score = ((w_count + b_count) / total) * 100

        return score


    def calculo_score_ands(self, clase_senal, deteccion):
        mask_signal = clase_senal.mask
        mask_sum = clase_senal.pixel_count
        row = deteccion.mask.shape[0]
        col = deteccion.mask.shape[1]
        total = row * col

        img_and = cv.bitwise_and(deteccion.mask, mask_signal)
        sum_white = cv.sumElems(img_and)[0] / 255

        img_and_not = cv.bitwise_and(~deteccion.mask, ~mask_signal)
        sum_black = cv.sumElems(img_and_not)[0] / 255

        score = 0.0
        if (sum_white >= mask_sum / 2) and (sum_black >= (total - mask_sum) / 2):
            score = (sum_white + sum_black) / total * 100
        return score


    # Puntuar detecciones
    def score(self, detecciones):
        scored_detecciones = []

        for deteccion in detecciones:
            if self.vers > 0:
                deteccion.mask = self.enmascararador.enmascarar(deteccion.get_image())
            else:
                deteccion.mask = self.enmascararador.enmascarar_sin_dilatar(deteccion.get_image())

            if self.vers == 2:
                if not self.detectar_Hough(deteccion.mask):
                    continue

            result_array = []
            for clase_senal in self.clases_senal:
                if self.vers > 0:
                    score = self.calculo_score_ands(clase_senal, deteccion)
                else:
                    score = self.calculo_score_enunciado(clase_senal, deteccion)
                result_array.append(score)

            deteccion.score = np.max(result_array)
            deteccion.type = np.argmax(result_array) + 1
            deteccion.name = self.clases_senal[np.argmax(result_array)].name

            if (self.vers == 0 and deteccion.score >= 50.0) or (self.vers > 0 and deteccion.score >= 70.0):
                scored_detecciones.append(deteccion)

        return scored_detecciones


    def score_y_clasificar_no_senales(self, detecciones):
        scored_detecciones = []

        X_train = np.array([])
        y_train = np.array([])

        X_test = np.array([])

        i = 0
        for deteccion in detecciones:
            if self.vers > 0:
                deteccion.mask = self.enmascararador.enmascarar(deteccion.get_image())
            else:
                deteccion.mask = self.enmascararador.enmascarar_sin_dilatar(deteccion.get_image())

            if self.vers == 2:
                if not self.detectar_Hough(deteccion.mask):
                    continue

            result_array = []
            for clase_senal in self.clases_senal:
                if self.vers > 0:
                    score = self.calculo_score_ands(clase_senal, deteccion)
                else:
                    score = self.calculo_score_enunciado(clase_senal, deteccion)
                result_array.append(score)

            deteccion.score = np.max(result_array)
            deteccion.type = np.argmax(result_array) + 1
            deteccion.name = self.clases_senal[np.argmax(result_array)].name

            ####

            img = deteccion.get_image()
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            img = cv.equalizeHist(img)
            img = cv.resize(img, self.dimensiones, interpolation=cv.INTER_LINEAR)

            descriptor = None
            if self.descriptor_type == 'hog':
                descriptor = self.hog.compute(img)
            elif self.descriptor_type == 'lbp':
                radius = 3
                n_points = 8 * radius
                descriptor = local_binary_pattern(img, n_points, radius, method='uniform')

            if descriptor is None:
                raise Exception("No descriptor available")
            else:
                descriptor = np.reshape(descriptor, (1, self.dimensiones[0] * self.dimensiones[1]))

                if (self.vers == 0 and deteccion.score >= 50.0) or (self.vers > 0 and deteccion.score >= 60.0):
                    scored_detecciones.append(deteccion)
                    deteccion.caracteristicas = descriptor

                    if X_test.shape[0] < 1:
                        X_test = np.array(descriptor)

                    else:
                        X_test = np.vstack((X_test, descriptor))

                else:
                    if X_train.shape[0] < 1:
                        X_train = np.array(descriptor)
                        y_train = np.zeros((1,))
                    else:
                        X_train = np.vstack((X_train, descriptor))
                        tmp = np.zeros((1,))
                        y_train = np.hstack((y_train, tmp))

            ####
            if i == 100:
                print('.', end=' ')
                i = 0
            i += 1
        return scored_detecciones, X_train, y_train, X_test
