import cv2  # Importación de OpenCV para operaciones de imagen y transformaciones afines.
import numpy as np  # Importación de NumPy para manejo de arrays y cálculos vectoriales/matriciales.
import os  # Importación de 'os' (aunque no se usa directamente en este script, se mantiene por convención).

# PEP 8: Se dejan dos líneas en blanco después de las importaciones.
# --- Configuraciones de la Pose Canónica Deseada ---
# Dimensiones de la imagen de salida normalizada. (Constantes en MAYÚSCULAS).
ANCHO_ROSTRO_DESEADO = 160
ALTO_ROSTRO_DESEADO = 160
# Definición de la posición canónica (para el centrado del rostro)
# Posición relativa donde se desea que caiga el centro del ojo izquierdo (ej. 35% del ancho y 35% del alto).
CENTRO_OJO_IZQUIERDO_X = 0.35 # Queremos el ojo izquierdo al 35% del ancho
CENTRO_OJO_IZQUIERDO_Y = 0.35 # Queremos el ojo izquierdo al 35% del alto

# Mensaje informativo.
print("--- 2. Aplicando Transformación Afín (OpenCV): Alineación y Normalización Geométrica ---")

# 1. Cargar imagen y puntos clave
# Bloque try-except para manejar la dependencia del script anterior.
try:
    # Carga la imagen original guardada temporalmente.
    imagen = cv2.imread("temp_imagen_original.jpg") 
    # Carga los 68 puntos clave detectados en el script anterior.
    puntos_clave = np.load("temp_puntos_clave.npy")
except FileNotFoundError:
    # Manejo de error si los archivos temporales no existen.
    print("ERROR: No se encontraron los archivos temporales. Ejecute el Script 1 primero.")
    exit()

# LÍNEA DE DEPURACIÓN: Asegurar que la imagen se cargó
# Verificación robusta de la carga de la imagen.
if imagen is None or imagen.size == 0:
    print("ERROR FATAL: La imagen 'temp_imagen_original.jpg' no pudo cargarse o está vacía. Verifique el archivo.")
    exit()

# 2. Calcular Centros de Ojos (Puntos 36-47)
# Se usa np.mean para obtener las coordenadas flotantes precisas de los centros
# Ojo Izquierdo (puntos 36 a 41 de los 68 landmarks).
centro_ojo_izquierdo = puntos_clave[36:42].mean(axis=0) # array de float
# Ojo Derecho (puntos 42 a 47 de los 68 landmarks).
centro_ojo_derecho = puntos_clave[42:48].mean(axis=0)  # array de float

# El centro de rotación (punto medio entre los ojos) debe ser float
# Se calcula el punto medio entre los dos centros de ojos para usarlo como pivote de rotación.
centro_ojos_flotante = ((centro_ojo_izquierdo + centro_ojo_derecho) / 2.0).astype(float)


# 3. Calcular Ángulo (Rotación) y Escala (Zoom)
# Diferencia en el eje Y y X entre los centros de los ojos.
dY = centro_ojo_derecho[1] - centro_ojo_izquierdo[1]
dX = centro_ojo_derecho[0] - centro_ojo_izquierdo[0]
# Cálculo del ángulo de inclinación del rostro usando atan2 (más robusto).
angulo = np.degrees(np.arctan2(dY, dX)) 

# Normalización de tamaño (Escala)
# Distancia euclidiana entre los ojos en la imagen actual.
distancia_interocular_actual = np.sqrt((dX ** 2) + (dY ** 2))
# Distancia que *debería* tener la distancia interocular en la imagen normalizada.
distancia_interocular_deseada = (1.0 - 2 * CENTRO_OJO_IZQUIERDO_X) * ANCHO_ROSTRO_DESEADO
# Factor de escala (cuánto debe ampliarse/reducirse la imagen).
escala = distancia_interocular_deseada / distancia_interocular_actual

print(f"DEBUG: Distancia interocular real: {distancia_interocular_actual:.2f} px")
print(f"DEBUG: Factor de Escala aplicado: {escala:.2f}")


# 4. Construir Matriz de Transformación M (OpenCV)
# Obtener matriz M para Rotación y Escala
# Matriz 2x3 para la rotación alrededor del centro de los ojos y aplicación de la escala.
M = cv2.getRotationMatrix2D(tuple(centro_ojos_flotante), angulo, escala)

# Calcular el punto objetivo donde queremos que caiga el ojo izquierdo
# Coordenadas absolutas donde queremos que el ojo izquierdo se ubique en la imagen de salida.
punto_objetivo_izquierdo = (CENTRO_OJO_IZQUIERDO_X * ANCHO_ROSTRO_DESEADO, 
                             CENTRO_OJO_IZQUIERDO_Y * ALTO_ROSTRO_DESEADO)

# Ajustar la Traslación (Centrado) en la matriz M para llevar el centro de rotación al centro deseado
# M[0, 2] = Traslación en X (corrección)
# M[1, 2] = Traslación en Y (corrección)
# Se ajustan los términos de traslación (M[0,2] y M[1,2]) para alinear el ojo izquierdo al punto objetivo.
M[0, 2] += (punto_objetivo_izquierdo[0] - centro_ojo_izquierdo[0] * M[0, 0] - centro_ojo_izquierdo[1] * M[0, 1])
M[1, 2] += (punto_objetivo_izquierdo[1] - centro_ojo_izquierdo[0] * M[1, 0] - centro_ojo_izquierdo[1] * M[1, 1])


# 5. Aplicar la Transformación Afín con OpenCV
# Aplica la matriz de transformación M a la imagen original.
# Se usa cv2.INTER_CUBIC para una interpolación de alta calidad.
rostro_alineado = cv2.warpAffine(imagen, M, (ANCHO_ROSTRO_DESEADO, ALTO_ROSTRO_DESEADO), 
                                 flags=cv2.INTER_CUBIC)

# 6. Guardar el resultado (Solución definitiva al problema de imagen negra)
# La conversión a np.uint8 es esencial para que los visores puedan interpretar el archivo JPG.
# Se asegura que los tipos de datos sean enteros de 8 bits (estándar para imágenes).
rostro_alineado_final = rostro_alineado.astype(np.uint8)

# *** SOLUCIÓN ADICIONAL PARA IMAGEN NEGRA: Conversión a BGR de 3 canales ***
# Si por alguna razón la imagen alineada solo tiene 1 canal, la convertimos a 3.
# Se verifica si el array tiene solo dos dimensiones (escala de grises) o un solo canal.
if len(rostro_alineado_final.shape) == 2 or rostro_alineado_final.shape[2] == 1:
    # Se convierte la imagen de escala de grises a BGR (3 canales).
    rostro_alineado_final = cv2.cvtColor(rostro_alineado_final, cv2.COLOR_GRAY2BGR)


# Nombres de archivo de salida.
output_filename = "2_rostro_alineado.jpg"
# Guarda la imagen final alineada.
cv2.imwrite(output_filename, rostro_alineado_final)
# Guarda una copia temporal.
cv2.imwrite("temp_rostro_alineado.jpg", rostro_alineado_final)
# Mensaje de confirmación.
print(f"Transformación Afín aplicada con éxito. Rostro normalizado y guardado en: {output_filename}")