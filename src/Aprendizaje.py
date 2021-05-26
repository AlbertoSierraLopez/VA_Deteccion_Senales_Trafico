import os
import cv2 as cv
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import neighbors
from skimage.feature import local_binary_pattern


class Aprendizaje:

    def __init__(self):
        self.hog = self.get_hog()

        self.X_train = None
        self.y_train = None

        self.pca = None
        self.lda = LinearDiscriminantAnalysis()
        self.knn = None


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


    def load_data(self, train_dir, descriptor_type='hog', dimensions=(30, 30)):
        X = np.array([])
        y = np.array([])
        # Iterar sobre todas las sub-carpetas de train
        for sub_dir in os.listdir(train_dir):
            list_sub_dir = os.listdir(train_dir + "/" + sub_dir)
            X_sub = np.zeros((len(list_sub_dir), dimensions[0] * dimensions[1]))
            y_sub = np.ones((len(list_sub_dir), 1))
            # Iterar sobre todas las imágenes de una carpeta
            for i in range(len(list_sub_dir)):
                file = list_sub_dir[i]

                # Procesar imagen
                img = cv.imread(train_dir + "/" + sub_dir + "/" + file, 0)
                img = cv.equalizeHist(img)
                img = cv.resize(img, dimensions, interpolation=cv.INTER_LINEAR)

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
                    X_sub[i] = np.reshape(descriptor, (1, dimensions[0] * dimensions[1]))
                    y_sub[i] = int(sub_dir) * y_sub[i]

            # Apilar arrays
            if X.shape[0] < 1:
                X = X_sub
                y = y_sub
            else:
                X = np.vstack((X, X_sub))
                y = np.vstack((y, y_sub))

        self.X_train = X
        self.y_train = np.reshape(y, (y.shape[0],))


    def load_senales(self, train_dir, X_train_no, y_train_no, descriptor_type='hog', dimensions=(30, 30)):
        X = np.array([])
        y = np.array([])
        # Iterar sobre todas las sub-carpetas de train
        for sub_dir in os.listdir(train_dir):
            list_sub_dir = os.listdir(train_dir + "/" + sub_dir)
            X_sub = np.zeros((len(list_sub_dir), dimensions[0] * dimensions[1]))
            y_sub = np.zeros((len(list_sub_dir), 1))
            # Iterar sobre todas las imágenes de una carpeta
            for i in range(len(list_sub_dir)):
                file = list_sub_dir[i]

                # Procesar imagen
                img = cv.imread(train_dir + "/" + sub_dir + "/" + file, 0)
                img = cv.equalizeHist(img)
                img = cv.resize(img, dimensions, interpolation=cv.INTER_LINEAR)

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
                    X_sub[i] = np.reshape(descriptor, (1, dimensions[0] * dimensions[1]))
                    y_sub[i] = 1

            # Apilar arrays
            if X.shape[0] < 1:
                X = X_sub
                y = y_sub
            else:
                X = np.vstack((X, X_sub))
                y = np.vstack((y, y_sub))

        self.X_train = np.vstack((X, X_train_no))
        y = np.reshape(y, (y.shape[0], ))
        self.y_train = np.concatenate([y, y_train_no])


    def entrenar_PCA(self):
        self.pca = PCA(len(np.unique(self.y_train)))
        self.pca.fit(self.X_train)


    def reducir_PCA(self, X):
        Z = self.pca.transform(X)
        # print("Reducción de la Dimensionalidad PCA:", X.shape[1], '-->', Z.shape[1])

        return Z


    def reducir_LDA(self, X):
        self.lda.fit(self.X_train, self.y_train)
        Z = self.lda.transform(X)
        # print("Reducción de la Dimensionalidad LDA:", X.shape[1], '-->', Z.shape[1])

        return Z


    def entrenar_LDA(self):
        self.lda.fit(self.X_train, self.y_train)

        return self.lda


    def entrenar_KNN(self, X, k=5):
        self.knn = neighbors.KNeighborsClassifier(k)
        self.knn.fit(X, self.y_train)

        return self.knn
