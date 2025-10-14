import cv2
import numpy as np
import os

# --- Configuraciones de la Pose Canónica Deseada ---
ANCHO_ROSTRO_DESEADO = 160
ALTO_ROSTRO_DESEADO = 160
# Definición de la posición canónica (para el centrado del rostro)
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

# LÍNEA DE DEPURACIÓN: Asegurar que la imagen se cargó
if imagen is None or imagen.size == 0:
    print("ERROR FATAL: La imagen 'temp_imagen_original.jpg' no pudo cargarse o está vacía. Verifique el archivo.")
    exit()

# 2. Calcular Centros de Ojos (Puntos 36-47)
# Se usa np.mean para obtener las coordenadas flotantes precisas de los centros
centro_ojo_izquierdo = puntos_clave[36:42].mean(axis=0) # array de float
centro_ojo_derecho = puntos_clave[42:48].mean(axis=0)  # array de float

# El centro de rotación (punto medio entre los ojos) debe ser float
centro_ojos_flotante = ((centro_ojo_izquierdo + centro_ojo_derecho) / 2.0).astype(float)


# 3. Calcular Ángulo (Rotación) y Escala (Zoom)
dY = centro_ojo_derecho[1] - centro_ojo_izquierdo[1]
dX = centro_ojo_derecho[0] - centro_ojo_izquierdo[0]
angulo = np.degrees(np.arctan2(dY, dX)) 

# Normalización de tamaño (Escala)
distancia_interocular_actual = np.sqrt((dX ** 2) + (dY ** 2))
distancia_interocular_deseada = (1.0 - 2 * CENTRO_OJO_IZQUIERDO_X) * ANCHO_ROSTRO_DESEADO
escala = distancia_interocular_deseada / distancia_interocular_actual

print(f"DEBUG: Distancia interocular real: {distancia_interocular_actual:.2f} px")
print(f"DEBUG: Factor de Escala aplicado: {escala:.2f}")


# 4. Construir Matriz de Transformación M (OpenCV)
# Obtener matriz M para Rotación y Escala
M = cv2.getRotationMatrix2D(tuple(centro_ojos_flotante), angulo, escala)

# Calcular el punto objetivo donde queremos que caiga el ojo izquierdo
punto_objetivo_izquierdo = (CENTRO_OJO_IZQUIERDO_X * ANCHO_ROSTRO_DESEADO, 
                            CENTRO_OJO_IZQUIERDO_Y * ALTO_ROSTRO_DESEADO)

# Ajustar la Traslación (Centrado) en la matriz M para llevar el centro de rotación al centro deseado
# M[0, 2] = Traslación en X (corrección)
# M[1, 2] = Traslación en Y (corrección)
M[0, 2] += (punto_objetivo_izquierdo[0] - centro_ojo_izquierdo[0] * M[0, 0] - centro_ojo_izquierdo[1] * M[0, 1])
M[1, 2] += (punto_objetivo_izquierdo[1] - centro_ojo_izquierdo[0] * M[1, 0] - centro_ojo_izquierdo[1] * M[1, 1])


# 5. Aplicar la Transformación Afín con OpenCV
rostro_alineado = cv2.warpAffine(imagen, M, (ANCHO_ROSTRO_DESEADO, ALTO_ROSTRO_DESEADO), 
                                 flags=cv2.INTER_CUBIC)

# 6. Guardar el resultado (Solución definitiva al problema de imagen negra)
# La conversión a np.uint8 es esencial para que los visores puedan interpretar el archivo JPG.
rostro_alineado_final = rostro_alineado.astype(np.uint8)

# *** SOLUCIÓN ADICIONAL PARA IMAGEN NEGRA: Conversión a BGR de 3 canales ***
# Si por alguna razón la imagen alineada solo tiene 1 canal, la convertimos a 3.
if len(rostro_alineado_final.shape) == 2 or rostro_alineado_final.shape[2] == 1:
    rostro_alineado_final = cv2.cvtColor(rostro_alineado_final, cv2.COLOR_GRAY2BGR)


output_filename = "2_rostro_alineado.jpg"
cv2.imwrite(output_filename, rostro_alineado_final)
cv2.imwrite("temp_rostro_alineado.jpg", rostro_alineado_final)
print(f"✔ Transformación Afín aplicada con éxito. Rostro normalizado y guardado en: {output_filename}")