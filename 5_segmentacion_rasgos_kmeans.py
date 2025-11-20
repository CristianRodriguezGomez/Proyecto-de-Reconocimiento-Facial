import cv2
import numpy as np
import os

# --- Configuraciones ---
NOMBRE_IMAGEN_ALINEADA = "temp_rostro_alineado.jpg"
MATRIZ_TRANSFORMACION_TEMP = "temp_matriz_transformacion.npy"
PUNTOS_CLAVE_TEMP = "temp_puntos_clave.npy"
RUTA_SALIDA = "output_fotos_segmentacion_rasgos"

# Crear el directorio de salida si no existe
os.makedirs(RUTA_SALIDA, exist_ok=True)

# --- Índices de los puntos clave (landmarks) para cada rasgo facial según el modelo de 68 puntos de dlib ---
INDICES_RASGOS = {
    "ojo_derecho": list(range(36, 42)),
    "ojo_izquierdo": list(range(42, 48)),
}

print("=" * 70)
print("   SCRIPT 5.2: SEGMENTACIÓN DE OJOS (K-Means Clustering)")
print("=" * 70)

# 1. Cargar la imagen alineada y los puntos clave
try:
    rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)
    puntos_clave_originales = np.load(PUNTOS_CLAVE_TEMP)
    matriz_transformacion = np.load(MATRIZ_TRANSFORMACION_TEMP)
except FileNotFoundError:
    print(f"ERROR: No se encontraron los archivos necesarios. Ejecuta los scripts 1 y 2 primero.")
    exit()

if rostro_alineado is None:
    print(f"ERROR: No se pudo cargar la imagen '{NOMBRE_IMAGEN_ALINEADA}'.")
    exit()

# La umbralización se realiza sobre imágenes en escala de grises
rostro_gris = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
print(f"Imagen '{NOMBRE_IMAGEN_ALINEADA}' cargada y convertida a escala de grises.")

# Transformar los puntos clave para que coincidan con la imagen alineada
puntos_homogeneos = np.hstack([puntos_clave_originales, np.ones((puntos_clave_originales.shape[0], 1))])
puntos_clave_transformados = np.dot(matriz_transformacion, puntos_homogeneos.T).T.astype(int)
print("Puntos clave transformados para coincidir con la imagen alineada.")

# 2. Bucle para procesar cada ojo
for nombre_rasgo, indices in INDICES_RASGOS.items():
    print(f"\n[PROCESANDO] Segmentando '{nombre_rasgo}' con K-Means...")

    # Extraer los puntos clave YA TRANSFORMADOS para el rasgo actual
    puntos_rasgo = puntos_clave_transformados[indices]

    # --- Creación de la Máscara Poligonal ---
    mascara_poligonal = np.zeros_like(rostro_gris)
    puntos_casco = cv2.convexHull(puntos_rasgo)
    cv2.fillConvexPoly(mascara_poligonal, puntos_casco, 255)
    
    # --- Aislamiento del Rasgo ---
    rasgo_aislado = cv2.bitwise_and(rostro_gris, rostro_gris, mask=mascara_poligonal)

    # --- Segmentación con K-Means ---
    # Preparamos los datos: K-Means necesita una lista de píxeles.
    # Tomamos solo los píxeles dentro de la máscara poligonal (los que no son negros).
    pixeles_rasgo = rasgo_aislado[mascara_poligonal == 255]
    Z = np.float32(pixeles_rasgo)

    # Definimos el criterio de parada y el número de clústeres (K)
    # K=3 puede ser bueno para separar pupila, iris y esclerótica.
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruimos la imagen segmentada
    center = np.uint8(center)
    res = center[label.flatten()]
    
    # Creamos una imagen en negro y colocamos los píxeles segmentados en su lugar
    rasgo_segmentado_kmeans = np.zeros_like(rasgo_aislado)
    rasgo_segmentado_kmeans[mascara_poligonal == 255] = res

    # Guardar el resultado
    ruta_segmentado = os.path.join(RUTA_SALIDA, f"5.5_kmeans_K{K}_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_segmentado, rasgo_segmentado_kmeans)
    print(f"  - Rasgo segmentado (K-Means con K={K}) guardado en: {ruta_segmentado}")

print("\n" + "=" * 70)
print("PROCESO DE SEGMENTACIÓN K-MEANS COMPLETADO")
print(f"Los resultados se han guardado en la carpeta: '{RUTA_SALIDA}'")
print("=" * 70)