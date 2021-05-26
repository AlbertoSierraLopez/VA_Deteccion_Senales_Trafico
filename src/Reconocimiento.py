import os
import cv2 as cv
import numpy as np

from skimage.feature import local_binary_pattern


class Reconocimiento:

    def __init__(self):
        self.hog = self.get_hog()

        self.X_test = None
        self.y_test = None


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


    def load_data(self, test_dir, descriptor_type='hog', dimensions=(30, 30)):
        list_files = os.listdir(test_dir)
        list_files.pop(0)
        X = np.zeros((len(list_files), dimensions[0] * dimensions[1]))
        y = np.ones((len(list_files), 1))

        for i in range(len(list_files)):
            file = list_files[i]
            file_name = file.split('-')
            # Procesar imagen
            img = cv.imread(test_dir + "/" + file, 0)
            img = cv.equalizeHist(img)
            img = cv.resize(img, (30, 30), interpolation=cv.INTER_LINEAR)

            descriptor = None
            if descriptor_type == 'hog':
                descriptor = self.hog.compute(img)
            elif descriptor_type == 'lbp':
                radius = 3
                n_points = 8 * radius
                descriptor = local_binary_pattern(img, n_points, radius, method='uniform')

            if descriptor is None:
                raise Exception("No descriptor available")
            else:
                X[i] = np.reshape(descriptor, (1, dimensions[0] * dimensions[1]))
                y[i] = int(file_name[0]) * y[i]

        self.X_test = X
        self.y_test = np.reshape(y, (y.shape[0],))


    def clasificar_LDA(self, lda):
        y_predicted = lda.predict(self.X_test)
        n_aciertos = np.sum(y_predicted == self.y_test)
        # print("Tasa de acierto:", round(n_aciertos / len(y_predicted) * 100, 2), '%')
        return y_predicted


    def clasificar_KNN(self, X, knn):
        y_predicted = knn.predict(X)
        n_aciertos = np.sum(self.y_test == y_predicted)
        return y_predicted