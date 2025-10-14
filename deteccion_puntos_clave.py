import cv2
import dlib
import numpy as np
import os

# --- Configuraciones ---
RUTA_MODELO_LANDMARKS = os.path.join("modelos", "shape_predictor_68_face_landmarks.dat")
NOMBRE_IMAGEN_ENTRADA = os.path.join("input_fotos", "rostro_prueba.jpg")  # ¡Ajustar por su foto!

# Inicializar detectores
predictor_landmarks = dlib.shape_predictor(RUTA_MODELO_LANDMARKS)
detector_rostro = dlib.get_frontal_face_detector()

print(f"--- 1. Detectando Rostro y Puntos Clave (Usando Dlib Predictor): {NOMBRE_IMAGEN_ENTRADA} ---")

# 1. Cargar imagen con OpenCV
imagen = cv2.imread(NOMBRE_IMAGEN_ENTRADA) 

if imagen is None:
    print(f"ERROR: No se pudo cargar la imagen {NOMBRE_IMAGEN_ENTRADA}")
    print("VERIFICAR: Asegúrese que la ruta y el nombre del archivo son correctos.")
    exit()

gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 2. Detección de rostros
rectangulos = detector_rostro(gris, 1)

if len(rectangulos) == 0:
    print("❌ No se detectó ningún rostro. Verifique la imagen.")
    exit()

# Tomar el primer rostro y obtener los 68 puntos clave
rect = rectangulos[0]
puntos_clave_dlib = predictor_landmarks(gris, rect)

# Convertir a array NumPy para uso en OpenCV
puntos_clave_array = np.zeros((68, 2), dtype="int")
for i in range(0, 68):
    puntos_clave_array[i] = (puntos_clave_dlib.part(i).x, puntos_clave_dlib.part(i).y)

print(f"✔ 68 Puntos Clave (Landmarks) detectados.")


# --- NUEVA SECCIÓN: VISUALIZACIÓN DE LANDMARKS CON OPENCV ---
# Creamos una copia de la imagen original para dibujar sobre ella.
imagen_visualizacion = imagen.copy()

# 3. Dibujar cada landmark como un círculo azul
for (x, y) in puntos_clave_array:
    # cv2.circle(imagen, centro, radio, color, grosor)
    cv2.circle(imagen_visualizacion, (x, y), 2, (255, 0, 0), -1) # Círculo azul de 2 píxeles

# También dibujamos el bounding box (opcional pero útil)
(x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
cv2.rectangle(imagen_visualizacion, (x, y), (x + w, y + h), (0, 255, 0), 2)


# 4. Guardar los archivos de salida
np.save("temp_puntos_clave.npy", puntos_clave_array)
cv2.imwrite("temp_imagen_original.jpg", imagen)

# ¡Guardamos la imagen con los dibujos!
cv2.imwrite("1_deteccion_puntos_clave.jpg", imagen_visualizacion) 

print("Archivos temporales guardados. ✔")
print("Imagen de visualización con landmarks guardada en: 1_deteccion_puntos_clave.jpg")