import cv2 as cv

from Deteccion import Deteccion


class Detector_MSER:

    def __init__(self, vers, delta=7, max_var=0.30, dimensiones=(25, 25)):
        self.vers = vers
        self.mser = cv.MSER_create(_delta=delta, _max_variation=max_var, _min_area=128)
        self.detecciones = []
        self.dimensiones = dimensiones


    def ecualizar(self, img, var=7200):
        if self.vers == 0 or img.var() < var:
            img = cv.equalizeHist(img)

        return img


    def detectar(self, path):
        src = cv.imread(path)
        img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        # Ecualizar el histograma (solo si es necesario)
        img = self.ecualizar(img)

        polygons = self.mser.detectRegions(img)
        rectangles_img = set()

        for polygon in polygons[0]:
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
            # https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
            x, y, w, h = cv.boundingRect(polygon)
            if (w / h > 0.85) and (w / h < 1.15):  # Sólo se admiten los rectángulos con el aspect ratio es parecido a 1:1
                ## Rectángulo pequeño (señal stop)
                off = int(w * 0.1)
                if (x - off > 0 and y - off > 0 and x + w + off < img.shape[1] and y + h + off < img.shape[0]):
                    rectangles_img.add((x - off, y - off, w + off * 2, h + off * 2))

                ## Rectángulo grande (señal peligro y prohibición)
                off = int(w * 0.35)  # Ampliar el rectángulo en caso de que sea señal de prohibición o peligro (sólo se ha detectado el interior blanco)
                if (x - off > 0 and y - off > 0 and x + w + off < img.shape[1] and y + h + off < img.shape[0]):
                    rectangles_img.add((x - off, y - off, w + off * 2, h + off * 2))

        for rectangle in rectangles_img:
            deteccion = Deteccion(path, rectangle, self.dimensiones)
            self.detecciones.append(deteccion)
