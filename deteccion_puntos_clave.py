import cv2  # Importación de la librería OpenCV para procesamiento de imágenes.
import dlib  # Importación de dlib, usado aquí para detección de rostros y puntos clave (landmarks).
import numpy as np  # Importación de NumPy para manejo eficiente de arrays numéricos, especialmente para los puntos clave.
import os  # Importación de la librería 'os' para manejo de rutas de archivos, manteniendo el código portable.

# PEP 8: Constantes (como las rutas) suelen ir en MAYÚSCULAS con guiones bajos.
# Aunque las rutas se usan como variables de configuración, el código las mantiene con ese nombre.
# --- Configuraciones ---
RUTA_MODELO_LANDMARKS = os.path.join("modelos", "shape_predictor_68_face_landmarks.dat")
NOMBRE_IMAGEN_ENTRADA = os.path.join("input_fotos", "rostro_prueba.jpg") 

# PEP 8: Se dejan dos líneas en blanco después de las importaciones (ya está correcto).
# Inicializar detectores
# Carga el predictor de 68 puntos clave de dlib usando la ruta especificada.
predictor_landmarks = dlib.shape_predictor(RUTA_MODELO_LANDMARKS)
# Inicializa el detector de rostros predeterminado de dlib (HOG + SVM).
detector_rostro = dlib.get_frontal_face_detector()

# Mensaje informativo para el usuario.
print(f"--- 1. Detectando Rostro y Puntos Clave (Usando Dlib Predictor): {NOMBRE_IMAGEN_ENTRADA} ---")

# 1. Cargar imagen con OpenCV
# Lee la imagen desde la ruta de entrada.
imagen = cv2.imread(NOMBRE_IMAGEN_ENTRADA) 

# Verificación de carga de imagen. (Buena práctica para evitar errores de ejecución)
if imagen is None:
    print(f"ERROR: No se pudo cargar la imagen {NOMBRE_IMAGEN_ENTRADA}")
    print("VERIFICAR: Asegúrese que la ruta y el nombre del archivo son correctos.")
    exit()

# Convierte la imagen a escala de grises. La mayoría de los algoritmos de dlib prefieren la imagen en escala de grises.
gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)

# 2. Detección de rostros
# PEP 8: Se prefiere dejar una línea en blanco antes y después de bloques lógicos o bucles.
# Ejecuta el detector de rostros. El '1' indica que se submuestree la imagen 1 vez (más detección, más lento).
rectangulos = detector_rostro(gris, 1)

# Verificación si se encontró al menos un rostro.
if len(rectangulos) == 0:
    print("No se detectó ningún rostro. Verifique la imagen.")
    exit()

# Tomar el primer rostro y obtener los 68 puntos clave
# Selecciona el primer rectángulo de rostro detectado.
rect = rectangulos[0]
# Usa el predictor para obtener los 68 puntos clave (landmarks) en el rostro detectado.
puntos_clave_dlib = predictor_landmarks(gris, rect)

# Convertir a array NumPy para uso en OpenCV
# Inicializa un array NumPy de 68x2 para almacenar las coordenadas (x, y) de los puntos.
puntos_clave_array = np.zeros((68, 2), dtype="int")
# Itera sobre los 68 puntos clave devueltos por dlib.
for i in range(0, 68):
    # Almacena las coordenadas x e y del punto 'i' en el array NumPy.
    puntos_clave_array[i] = (puntos_clave_dlib.part(i).x, puntos_clave_dlib.part(i).y)

# Mensaje de confirmación.
print(f"68 Puntos Clave (Landmarks) detectados.")


# PEP 8: Separador de sección lógica con comentarios.
# --- NUEVA SECCIÓN: VISUALIZACIÓN DE LANDMARKS CON OPENCV ---
# Creamos una copia de la imagen original para dibujar sobre ella.
imagen_visualizacion = imagen.copy()

# 3. Dibujar cada landmark como un círculo azul
# Itera sobre las coordenadas (x, y) de cada punto clave ya convertido a NumPy.
for (x, y) in puntos_clave_array:
    # Comentario útil sobre la sintaxis de cv2.circle.
    # cv2.circle(imagen, centro, radio, color, grosor)
    # Dibuja un círculo azul (255, 0, 0 en BGR) de radio 2, relleno (-1).
    cv2.circle(imagen_visualizacion, (x, y), 2, (255, 0, 0), -1) 

# También dibujamos el bounding box (opcional pero útil)
# Desempaqueta las coordenadas y dimensiones del rectángulo (bounding box) de dlib.
(x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
# Dibuja el rectángulo verde (0, 255, 0 en BGR) con grosor de 2 píxeles.
cv2.rectangle(imagen_visualizacion, (x, y), (x + w, y + h), (0, 255, 0), 2)


# 4. Guardar los archivos de salida
# Guarda el array de puntos clave en formato binario NumPy (.npy).
np.save("temp_puntos_clave.npy", puntos_clave_array)
# Guarda la imagen original sin modificaciones.
cv2.imwrite("temp_imagen_original.jpg", imagen)

# ¡Guardamos la imagen con los dibujos!
# Guarda la imagen final con los puntos clave y el rectángulo dibujados.
cv2.imwrite("1_deteccion_puntos_clave.jpg", imagen_visualizacion) 

# Mensajes de salida.
print("Archivos temporales guardados.")
print("Imagen de visualización con landmarks guardada en: 1_deteccion_puntos_clave.jpg")