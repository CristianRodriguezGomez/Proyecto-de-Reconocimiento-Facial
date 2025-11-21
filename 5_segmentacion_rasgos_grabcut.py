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
print("   SCRIPT 5.3: SEGMENTACIÓN DE OJOS (GrabCut)")
print("=" * 70)

# 1. Cargar la imagen alineada y los puntos clave
try:
    # GrabCut funciona mejor con imágenes a color para diferenciar regiones.
    rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)
    puntos_clave_originales = np.load(PUNTOS_CLAVE_TEMP)
    matriz_transformacion = np.load(MATRIZ_TRANSFORMACION_TEMP)
except FileNotFoundError:
    print(f"ERROR: No se encontraron los archivos necesarios. Ejecuta los scripts 1 y 2 primero.")
    exit()

if rostro_alineado is None:
    print(f"ERROR: No se pudo cargar la imagen '{NOMBRE_IMAGEN_ALINEADA}'.")
    exit()

print(f"Imagen '{NOMBRE_IMAGEN_ALINEADA}' cargada.")

# Transformar los puntos clave para que coincidan con la imagen alineada
puntos_homogeneos = np.hstack([puntos_clave_originales, np.ones((puntos_clave_originales.shape[0], 1))])
puntos_clave_transformados = np.dot(matriz_transformacion, puntos_homogeneos.T).T.astype(int)
print("Puntos clave transformados para coincidir con la imagen alineada.")

# 2. Bucle para procesar cada ojo
for nombre_rasgo, indices in INDICES_RASGOS.items():
    print(f"\n[PROCESANDO] Segmentando '{nombre_rasgo}' con GrabCut...")

    # Extraer los puntos clave YA TRANSFORMADOS para el rasgo actual
    puntos_rasgo = puntos_clave_transformados[indices]

    # --- Inicialización de GrabCut ---
    # GrabCut necesita un rectángulo (ROI) que contenga el objeto.
    # Usamos el bounding box de los puntos clave del rasgo.
    (x, y, w, h) = cv2.boundingRect(puntos_rasgo)
    
    # Añadimos un pequeño margen para asegurar que todo el rasgo está dentro
    margen = 5
    rect = (x - margen, y - margen, w + 2*margen, h + 2*margen)

    # GrabCut modifica una máscara. Creamos una máscara inicial y dos arrays temporales.
    mascara_grabcut = np.zeros(rostro_alineado.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Ejecutamos GrabCut. El modo GC_INIT_WITH_RECT le dice que use nuestro rectángulo.
    # El algoritmo itera para refinar la máscara, separando primer plano de fondo.
    cv2.grabCut(rostro_alineado, mascara_grabcut, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # La máscara resultante tiene valores (0, 1, 2, 3).
    # Creamos una máscara binaria donde el primer plano (1 y 3) es blanco (255).
    mascara_binaria = np.where((mascara_grabcut == 1) | (mascara_grabcut == 3), 255, 0).astype('uint8')
    
    # Aplicamos la máscara a la imagen original para visualizar el resultado
    rasgo_segmentado_grabcut = rostro_alineado * mascara_binaria[:, :, np.newaxis]

    # Guardar el resultado
    ruta_segmentado = os.path.join(RUTA_SALIDA, f"5.6_grabcut_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_segmentado, rasgo_segmentado_grabcut)
    print(f"  - Rasgo segmentado (GrabCut) guardado en: {ruta_segmentado}")

print("\n" + "=" * 70)
print("PROCESO DE SEGMENTACIÓN GRABCUT COMPLETADO")
print(f"Los resultados se han guardado en la carpeta: '{RUTA_SALIDA}'")
print("=" * 70)