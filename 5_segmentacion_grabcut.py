import cv2
import numpy as np
import os

# --- CONFIGURACIONES ---
NOMBRE_IMAGEN_ENTRADA = "temp_imagen_original.jpg"
ARCHIVO_PUNTOS = "temp_puntos_clave.npy"
CARPETA_SALIDA = "output_fotos_segmentacion_grabcut"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

print("=" * 70)
print("   SCRIPT 5: SEGMENTACIÓN DE ROSTRO COMPLETO (GRABCUT)")
print("   (Usando Landmarks para definir el área de interés)")
print("=" * 70)

# 1. Cargar la imagen original y los puntos clave
# Es CRÍTICO usar la imagen original porque los puntos .npy coinciden con ella.
img = cv2.imread(NOMBRE_IMAGEN_ENTRADA)
if img is None:
    print(f"ERROR: No se encuentra {NOMBRE_IMAGEN_ENTRADA}")
    exit()

try:
    puntos = np.load(ARCHIVO_PUNTOS)
except FileNotFoundError:
    print(f"ERROR: No se encuentra {ARCHIVO_PUNTOS}. Ejecuta el script 1 primero.")
    exit()

print(f"✓ Imagen cargada: {img.shape}")
print(f"✓ Puntos clave cargados: {len(puntos)} puntos")

# 2. Calcular el Rectángulo (Bounding Box) basado en los puntos
# GrabCut necesita un rectángulo inicial que cubra el objeto (la cara).
# Buscamos las coordenadas mínimas y máximas de los puntos.

x_min = np.min(puntos[:, 0])
x_max = np.max(puntos[:, 0])
y_min = np.min(puntos[:, 1])
y_max = np.max(puntos[:, 1])

# --- AÑADIR MARGEN (PADDING) ---
# Los puntos faciales suelen estar en el borde de la cara, pero GrabCut
# necesita un poco de "aire" alrededor (frente, barbilla, orejas) para trabajar bien.
margen_x = 30  # Píxeles extra a los lados
margen_y = 50  # Píxeles extra arriba (frente) y abajo (barbilla)

# Ajustamos el rectángulo asegurándonos de no salirnos de la imagen
h_img, w_img = img.shape[:2]

rect_x = max(0, x_min - margen_x)
rect_y = max(0, y_min - margen_y)
rect_w = min(w_img, x_max + margen_x) - rect_x
rect_h = min(h_img, y_max + margen_y) - rect_y

rectangulo_grabcut = (rect_x, rect_y, rect_w, rect_h)
print(f"✓ Rectángulo calculado (x,y,w,h): {rectangulo_grabcut}")

# 3. Preparar GrabCut
mask = np.zeros(img.shape[:2], np.uint8)
# Modelos internos que usa el algoritmo (arrays de ceros requeridos por OpenCV)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 4. Ejecutar GrabCut
# iterCount=5 es un buen balance entre velocidad y calidad.
# cv2.GC_INIT_WITH_RECT indica que iniciamos con el cuadro que calculamos arriba.
print("⏳ Ejecutando GrabCut... esto puede tardar un momento.")
cv2.grabCut(img, mask, rectangulo_grabcut, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 5. Procesar la máscara resultante
# GrabCut modifica la variable 'mask' con 4 valores:
# 0: Fondo seguro, 1: Primer plano seguro, 2: Probable fondo, 3: Probable primer plano.
# Convertimos todo lo que sea 0 o 2 a 0 (fondo), y 1 o 3 a 1 (rostro).
mascara_binaria = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Multiplicamos la imagen original por la máscara para dejar el fondo negro
img_segmentada = img * mascara_binaria[:, :, np.newaxis]

# 6. Guardar Resultados
nombre_salida = "5_rostro_segmentado_grabcut.jpg"
ruta_salida = os.path.join(CARPETA_SALIDA, nombre_salida)
cv2.imwrite(ruta_salida, img_segmentada)

# Opcional: Guardar imagen con el rectángulo dibujado para verificar (Debug)
img_rect = img.copy()
cv2.rectangle(img_rect, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)
cv2.imwrite(os.path.join(CARPETA_SALIDA, "debug_rectangulo_grabcut.jpg"), img_rect)

print("\n" + "=" * 70)
print(f"✓ GRABCUT COMPLETADO. Resultado guardado en:")
print(f"  -> {ruta_salida}")
print("=" * 70)
