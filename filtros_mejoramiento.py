import cv2  # Importación de OpenCV para procesamiento de imágenes y filtros.
import numpy as np  # Importación de NumPy (aunque no se usa directamente en operaciones aquí, es estándar).
import os  # Importación de 'os' (aunque no se usa directamente en operaciones aquí, es estándar).

# PEP 8: Se dejan dos líneas en blanco después de las importaciones.
# --- Configuraciones ---
# Nombre de la imagen de entrada que ya fue alineada geométricamente.
NOMBRE_IMAGEN_ALINEADA = "temp_rostro_alineado.jpg"

# Mensaje informativo.
print("--- 3. Aplicando Filtros con OpenCV: Normalización Fotométrica ---")

# 1. Cargar la imagen alineada
# Lee la imagen que se generó en el Script 2.
rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)
# Verificación si la imagen se cargó correctamente.
if rostro_alineado is None:
    print("ERROR: No se pudo cargar el rostro alineado. Ejecute el Script 2 primero.")
    exit()

# Convertir a escala de grises, requerido para CLAHE y Mediana
# Convierte la imagen BGR (color) a escala de grises, optimizando el procesamiento.
gris_rostro = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)


# PEP 8: Separador de sección lógica con comentarios.
# --- Filtro 1: Realzante / Normalización Fotométrica (CLAHE) ---
# Usa la Ecualización Adaptativa del Histograma con OpenCV
# Crea el objeto CLAHE (Contrast Limited Adaptive Histogram Equalization).
clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# Aplica el CLAHE a la imagen en escala de grises para mejorar el contraste local.
imagen_clahe = clahe_obj.apply(gris_rostro)
print("Filtro 1 (CLAHE) aplicado: Corrección de iluminación (Normalización Fotométrica).")


# PEP 8: Separador de sección lógica con comentarios.
# --- Filtro 2: Suavizante (Mediana) ---
# Usa el Filtro Mediana de OpenCV para reducir ruido.
# Aplica un filtro de mediana con un kernel de 5x5 para reducir ruido tipo "sal y pimienta".
imagen_suavizada = cv2.medianBlur(imagen_clahe, 5)
print("Filtro 2 (Mediana) aplicado: Reducción de ruido.")


# Opcional: Normalización de Escala de Píxeles (Algorítmica)
# Esto es esencial antes de la fase de Embeddings (ArcFace).
# Convierte los valores de píxeles a flotante y los normaliza al rango [0, 1].
pixeles_normalizados = imagen_suavizada.astype("float32") / 255.0

# 2. Guardar el resultado final
# Para guardar como JPG, lo convertimos de nuevo a 3 canales de color
# El filtro Mediana arrojó una imagen en gris; se convierte de nuevo a BGR (3 canales) para el guardado en JPG.
rostro_procesado_final = cv2.cvtColor(imagen_suavizada, cv2.COLOR_GRAY2BGR)
output_filename = "3_rostro_final_procesado.jpg"
# Guarda la imagen final después de la normalización fotométrica.
cv2.imwrite(output_filename, rostro_procesado_final)
print(f"Resultado final guardado en: {output_filename}")