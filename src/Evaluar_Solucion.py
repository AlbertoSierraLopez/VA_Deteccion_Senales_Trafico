

class Evaluar_Solucion:

    def __init__(self):
        self.training_list = []
        self.file_ok = False

        self.recall = 0
        self.precision = 0


    def set_file(self, file):
        op_file = open(file, 'r')
        lines = op_file.readlines()
        for txt_line in lines:
            line = txt_line.split(';')
            self.training_list.append(line)
        self.file_ok = True


    def son_iguales(self, deteccion, training):
        img_d, x1_d, y1_d, x2_d, y2_d, type_d, score = deteccion.to_string().split(';')
        img_t, x1_t, y1_t, x2_t, y2_t, type_t = training

        dif_x1 = abs(int(x1_d) - int(x1_t))
        dif_y1 = abs(int(y1_d) - int(y1_t))
        dif_x2 = abs(int(x2_d) - int(x2_t))
        dif_y2 = abs(int(y2_d) - int(y2_t))

        return (dif_x1 < 20) and (dif_y1 < 20) and (dif_x2 < 20) and (dif_y2 < 20)


    def evaluar(self, detecciones):
        n_aciertos = 0  # n_aciertos = True Positives

        for deteccion in detecciones:
            img = deteccion.to_string().split(';')[0]

            i = 0
            checked = False
            stop = False
            while (i < len(self.training_list)) and (not stop):
                if (img != self.training_list[i][0]) and checked:
                    stop = True
                elif img == self.training_list[i][0]:
                    checked = True
                    if self.son_iguales(deteccion, self.training_list[i]):
                        n_aciertos += 1
                        stop = True
                i += 1

        self.recall    = round(n_aciertos / len(detecciones) * 100, 2)
        self.precision = round(n_aciertos / len(self.training_list) * 100, 2)
