import cv2 as cv


class Enmascarar():

    # Transforma una imagen en una máscara binaria del color rojo
    def enmascarar_sin_dilatar(self, src_img):
        # https://medium.com/@gastonace1/detecci%C3%B3n-de-objetos-por-colores-en-im%C3%A1genes-con-python-y-opencv-c8d9b6768ff
        img_hsv = cv.cvtColor(src_img, cv.COLOR_RGB2HSV)

        # img_hsv = saturar(img_hsv)

        # Elegimos el umbral de rojo en HSV
        umbral_bajo1 = (165, 40, 30)
        umbral_alto1 = (180, 255, 255)

        # Elegimos el segundo umbral de rojo en HSV
        umbral_bajo2 = (0, 40, 30)
        umbral_alto2 = (10, 255, 255)

        # hacemos la mask y filtramos en la original
        mask1 = cv.inRange(img_hsv, umbral_bajo1, umbral_alto1)
        mask2 = cv.inRange(img_hsv, umbral_bajo2, umbral_alto2)
        mask = mask1 + mask2
        # La máscara es una matriz 25x25, res es una imagen 25x25x3 que sólo tiene los pixeles rojos
        res = cv.bitwise_and(src_img, src_img, mask=mask)

        # Umbralizar
        g_img = cv.cvtColor(res, cv.COLOR_RGB2GRAY)

        # Pasar un filtro a la imagen
        ## Suavizado
        # g_img = cv.GaussianBlur(g_img, (3, 3), 1)

        ret, bin_img = cv.threshold(g_img, 5, 255, cv.THRESH_BINARY)

        return bin_img

    def enmascarar(self, src_img):
        bin_img = self.enmascarar_sin_dilatar(src_img)

        # Pasar un filtro a la máscara
        ## Dilatación
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))  # creamos primero el elemento estructurante
        bin_img = cv.dilate(bin_img, kernel)  # dilatamos con la máscara creada

        return bin_img