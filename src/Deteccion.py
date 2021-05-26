import cv2 as cv
import numpy as np

class Deteccion:

    def __init__(self, path, rectangle, dimensiones=(25, 25)):
        self.path = path
        self.rectangle = rectangle
        self.mask = None
        self.score = None
        self.type = None
        self.sub_type = None
        self.name = ''

        self.caracteristicas = None

        self.dimensiones = dimensiones


    def get_image(self):
        src = cv.imread(self.path)
        img = cv.cvtColor(src, cv.COLOR_BGR2RGB)

        # Pasar un filtro a la imagen
        ## Suavizado
        # img = cv.GaussianBlur(img, (3, 3), 1)

        # Recortar imagen
        (x, y, w, h) = self.rectangle
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv.resize(crop_img, self.dimensiones, interpolation=cv.INTER_LINEAR)

        return crop_img


    def get_centro(self):
        (x, y, w, h) = self.rectangle
        return np.array([x+w/2, y+h/2])


    def get_coord(self):
        (x, y, w, h) = self.rectangle
        return [np.array([x, y]), np.array([x+w, y+h])]


    def to_string(self):
        split_path = self.path.split('/')
        img_path = split_path[len(split_path)-1]
        return img_path+';'+str(self.rectangle[0])+';'+str(self.rectangle[1])+';'+str(self.rectangle[0]+self.rectangle[2])+';'+str(self.rectangle[1]+self.rectangle[3])+';'+str(self.type)+';'+str(round(self.score, 2))
