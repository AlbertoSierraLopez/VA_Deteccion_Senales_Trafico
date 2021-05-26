import numpy as np


class Limpieza:

    def cercanos(self, det1, det2):
        d = np.linalg.norm(det1.get_centro() - det2.get_centro())
        return d < det1.rectangle[2]*0.2


    def limpiar(self, detecciones):
        detecciones_buenas_buenisimas = []

        last_path = ''
        path_index = 0
        for i in range(len(detecciones)):
            current_path = detecciones[i].path
            aux = detecciones[i]

            if current_path != last_path:
                path_index = i

            j = path_index
            while j < len(detecciones) and detecciones[j].path == current_path:
                if self.cercanos(aux, detecciones[j]):
                    if detecciones[j].score >= aux.score:
                        aux = detecciones[j]
                j += 1

            if not detecciones_buenas_buenisimas.__contains__(aux):
                detecciones_buenas_buenisimas.append(aux)
            last_path = current_path

        return detecciones_buenas_buenisimas
