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
print("   SCRIPT 5.1: SEGMENTACIÓN DE OJOS (Umbralización Adaptativa)")
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
    print(f"\n[PROCESANDO] Segmentando '{nombre_rasgo}' con Umbralización Adaptativa...")

    # Extraer los puntos clave YA TRANSFORMADOS para el rasgo actual
    puntos_rasgo = puntos_clave_transformados[indices]

    # --- Creación de la Máscara Poligonal ---
    mascara_poligonal = np.zeros_like(rostro_gris)
    puntos_casco = cv2.convexHull(puntos_rasgo)
    cv2.fillConvexPoly(mascara_poligonal, puntos_casco, 255)
    
    # --- Aislamiento del Rasgo ---
    rasgo_aislado = cv2.bitwise_and(rostro_gris, rostro_gris, mask=mascara_poligonal)

    # --- Segmentación con Umbralización Adaptativa ---
    # Este método es excelente para condiciones de iluminación variables dentro del rasgo.
    # blockSize: Tamaño del vecindario para calcular el umbral (debe ser impar).
    # C: Constante que se resta de la media calculada.
    rasgo_segmentado_adaptativo = cv2.adaptiveThreshold(
        rasgo_aislado, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Método Gaussiano, suele dar mejores resultados.
        cv2.THRESH_BINARY, 
        blockSize=11, 
        C=2
    )

    # Limpiamos el exterior del polígono por si el método adaptativo introdujo ruido.
    rasgo_segmentado_final = cv2.bitwise_and(rasgo_segmentado_adaptativo, rasgo_segmentado_adaptativo, mask=mascara_poligonal)

    # Guardar el resultado
    ruta_segmentado = os.path.join(RUTA_SALIDA, f"5.4_adaptativo_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_segmentado, rasgo_segmentado_final)
    print(f"  - Rasgo segmentado (Adaptativo) guardado en: {ruta_segmentado}")

print("\n" + "=" * 70)
print("PROCESO DE SEGMENTACIÓN ADAPTATIVA COMPLETADO")
print(f"Los resultados se han guardado en la carpeta: '{RUTA_SALIDA}'")
print("=" * 70)