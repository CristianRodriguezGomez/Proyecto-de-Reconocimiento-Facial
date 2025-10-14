import cv2
import numpy as np
import os

# --- Configuraciones ---
NOMBRE_IMAGEN_ALINEADA = "temp_rostro_alineado.jpg"

print("--- 3. Aplicando Filtros con OpenCV: Normalización Fotométrica ---")

# 1. Cargar la imagen alineada
rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)
if rostro_alineado is None:
    print("ERROR: No se pudo cargar el rostro alineado. Ejecute el Script 2 primero.")
    exit()

# Convertir a escala de grises, requerido para CLAHE y Mediana
gris_rostro = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)


# --- Filtro 1: Realzante / Normalización Fotométrica (CLAHE) ---
# Usa la Ecualización Adaptativa del Histograma con OpenCV
clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imagen_clahe = clahe_obj.apply(gris_rostro)
print("✔ Filtro 1 (CLAHE) aplicado: Corrección de iluminación (Normalización Fotométrica).")


# --- Filtro 2: Suavizante (Mediana) ---
# Usa el Filtro Mediana de OpenCV para reducir ruido.
imagen_suavizada = cv2.medianBlur(imagen_clahe, 5)
print("✔ Filtro 2 (Mediana) aplicado: Reducción de ruido.")


# Opcional: Normalización de Escala de Píxeles (Algorítmica)
# Esto es esencial antes de la fase de Embeddings (ArcFace).
pixeles_normalizados = imagen_suavizada.astype("float32") / 255.0

# 2. Guardar el resultado final
# Para guardar como JPG, lo convertimos de nuevo a 3 canales de color
rostro_procesado_final = cv2.cvtColor(imagen_suavizada, cv2.COLOR_GRAY2BGR)
output_filename = "3_rostro_final_procesado.jpg"
cv2.imwrite(output_filename, rostro_procesado_final)
print(f"Resultado final guardado en: {output_filename}")