import cv2  # Importación de OpenCV para operaciones de imagen y transformaciones afines.
import numpy as np  # Importación de NumPy para manejo de arrays y cálculos vectoriales/matriciales.
import os

# --- Configuraciones de la Pose Canónica Deseada ---
# Dimensiones de la imagen de salida normalizada.
ANCHO_ROSTRO_DESEADO = 160
ALTO_ROSTRO_DESEADO = 160
# Definición de la posición canónica (para el centrado del rostro)
# Posición relativa donde se desea que caiga el centro del ojo izquierdo
CENTRO_OJO_IZQUIERDO_X = 0.35 # Queremos el ojo izquierdo al 35% del ancho
CENTRO_OJO_IZQUIERDO_Y = 0.35 # Queremos el ojo izquierdo al 35% del alto

print("--- 2. Aplicando Transformación Afín (OpenCV): Alineación y Normalización Geométrica ---")

# 1. Cargar imagen y puntos clave
try:
    imagen = cv2.imread("temp_imagen_original.jpg") 
    puntos_clave = np.load("temp_puntos_clave.npy")
except FileNotFoundError:
    print("ERROR: No se encontraron los archivos temporales. Ejecute el Script 1 primero.")
    exit()

if imagen is None or imagen.size == 0:
    print("ERROR FATAL: La imagen 'temp_imagen_original.jpg' no pudo cargarse o está vacía. Verifique el archivo.")
    exit()

# 2. Calcular Centros de Ojos (Puntos 36-47)
centro_ojo_izquierdo = puntos_clave[36:42].mean(axis=0) # array de float (36-41)
centro_ojo_derecho = puntos_clave[42:48].mean(axis=0)  # array de float (42-47)
centro_ojos_flotante = ((centro_ojo_izquierdo + centro_ojo_derecho) / 2.0).astype(float)


# 3. Calcular Ángulo (Rotación) y Escala (Zoom)
dY = centro_ojo_derecho[1] - centro_ojo_izquierdo[1]
dX = centro_ojo_derecho[0] - centro_ojo_izquierdo[0]
angulo = np.degrees(np.arctan2(dY, dX)) 

distancia_interocular_actual = np.sqrt((dX ** 2) + (dY ** 2))
distancia_interocular_deseada = (1.0 - 2 * CENTRO_OJO_IZQUIERDO_X) * ANCHO_ROSTRO_DESEADO
escala = distancia_interocular_deseada / distancia_interocular_actual

print(f"DEBUG: Distancia interocular real: {distancia_interocular_actual:.2f} px")
print(f"DEBUG: Factor de Escala aplicado: {escala:.2f}")


# 4. Construir Matriz de Transformación M (OpenCV)
# Obtener matriz M inicial para Rotación y Escala
M = cv2.getRotationMatrix2D(tuple(centro_ojos_flotante), angulo, escala)

# Calcular el punto objetivo (posición absoluta) donde queremos que caiga el ojo izquierdo
punto_objetivo_izquierdo = (CENTRO_OJO_IZQUIERDO_X * ANCHO_ROSTRO_DESEADO, 
                            CENTRO_OJO_IZQUIERDO_Y * ALTO_ROSTRO_DESEADO)

# -------------------------------------------------------------
# *** CORRECCIÓN CRÍTICA DE TRASLACIÓN ***
# -------------------------------------------------------------
# Se ajustan los términos de traslación (M[0,2] y M[1,2]) para alinear el ojo izquierdo al punto objetivo.
# La traslación se debe hacer en las coordenadas del ojo ya escaladas y rotadas.

# Posición del ojo izquierdo después de la rotación y el escalado (ignorando traslación inicial)
x_transformado = centro_ojo_izquierdo[0] * M[0, 0] + centro_ojo_izquierdo[1] * M[0, 1]
y_transformado = centro_ojo_izquierdo[0] * M[1, 0] + centro_ojo_izquierdo[1] * M[1, 1]

# La traslación necesaria es la diferencia entre el objetivo y la posición transformada.
traslacion_x_ajustada = punto_objetivo_izquierdo[0] - x_transformado
traslacion_y_ajustada = punto_objetivo_izquierdo[1] - y_transformado

# Asignar la traslación corregida a la matriz M:
M[0, 2] = traslacion_x_ajustada
M[1, 2] = traslacion_y_ajustada

print(f"DEBUG: Traslación Final X (M[0, 2]): {M[0, 2]:.2f}")
print(f"DEBUG: Traslación Final Y (M[1, 2]): {M[1, 2]:.2f}")


# 5. Aplicar la Transformación Afín con OpenCV
rostro_alineado = cv2.warpAffine(imagen, M, (ANCHO_ROSTRO_DESEADO, ALTO_ROSTRO_DESEADO), 
                                 flags=cv2.INTER_CUBIC)

# 6. Guardar el resultado (CORRECCIÓN FINAL DE PIXEL RANGE Y DTYPE)
# 1. RECORTAR LOS VALORES: Forzar que estén en el rango [0, 255].
#    Esto resuelve el problema de los valores flotantes fuera de rango que causan la imagen negra.
rostro_alineado_clipped = np.clip(rostro_alineado, 0, 255)

# 2. CONVERTIR EL TIPO DE DATO: Cambiar de flotante a entero sin signo de 8 bits (uint8).
rostro_alineado_final = rostro_alineado_clipped.astype(np.uint8)

# 3. Conversión a BGR de 3 canales (Si la imagen se perdió a escala de grises en el proceso)
if len(rostro_alineado_final.shape) == 2 or rostro_alineado_final.shape[2] == 1:
    rostro_alineado_final = cv2.cvtColor(rostro_alineado_final, cv2.COLOR_GRAY2BGR)


# Nombres de archivo de salida.
output_filename = "2_rostro_alineado.jpg"
cv2.imwrite(output_filename, rostro_alineado_final)
cv2.imwrite("temp_rostro_alineado.jpg", rostro_alineado_final)

print(f"Transformación Afín aplicada con éxito. Rostro normalizado y guardado en: {output_filename}")
