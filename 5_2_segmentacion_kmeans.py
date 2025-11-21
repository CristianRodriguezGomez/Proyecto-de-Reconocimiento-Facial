import cv2
import numpy as np
import os

# --- Configuraciones ---
NOMBRE_IMAGEN_ALINEADA = "output_fotos_alineacion/2_rostro_alineado.jpg"
MATRIZ_TRANSFORMACION_TEMP = "temp_matriz_transformacion.npy"
PUNTOS_CLAVE_TEMP = "temp_puntos_clave.npy"

# Carpeta de salida diferente para guardar los resultados de K-Means.
RUTA_SALIDA = "output_fotos_segmentacion_kmeans"

# --- Parámetros de K-Means Clustering ---
K_CLUSTERS = 3  # Número de grupos (pupila, iris, esclerótica).
# Criterios de parada para el algoritmo:
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Crear el directorio de salida si no existe.
os.makedirs(RUTA_SALIDA, exist_ok=True)

# --- Índices de los puntos clave para rasgos faciales ---
INDICES_RASGOS = {
    "ojo_derecho": list(range(36, 42)),
    "ojo_izquierdo": list(range(42, 48)),
}

print("=" * 70)
print(f" SCRIPT 5C: SEGMENTACIÓN DE OJOS CON K-MEANS (K={K_CLUSTERS})")
print("=" * 70)

try:
    rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)
    puntos_clave_originales = np.load(PUNTOS_CLAVE_TEMP)
    matriz_transformacion = np.load(MATRIZ_TRANSFORMACION_TEMP)

except FileNotFoundError:
    print(
        f"ERROR: No se encontraron los archivos necesarios. "
        f"Asegúrate de haber ejecutado los scripts 1 y 2 primero."
    )
    exit()

if rostro_alineado is None:
    print(f"ERROR: No se pudo cargar la imagen '{NOMBRE_IMAGEN_ALINEADA}'.")
    exit()

# K-Means funciona mejor con color, pero para intensidad (gris) también es común
rostro_gris = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
print(f"Imagen '{NOMBRE_IMAGEN_ALINEADA}' cargada y convertida a escala de grises.")

# --- Transformación de puntos clave ---
puntos_homogeneos = np.hstack(
    [puntos_clave_originales, np.ones((puntos_clave_originales.shape[0], 1))]
)
puntos_clave_transformados = np.dot(
    matriz_transformacion, puntos_homogeneos.T
).T.astype(int)

print("Puntos clave transformados para coincidir con la imagen alineada.")
print("-" * 70)
print(f"Parámetros K-Means: K={K_CLUSTERS} clústeres.")
print("-" * 70)

# 2. Bucle para procesar cada rasgo facial
for nombre_rasgo, indices in INDICES_RASGOS.items():
    print(f"\n[PROCESANDO] Segmentando '{nombre_rasgo}'...")

    # Extraer puntos clave transformados.
    puntos_rasgo = puntos_clave_transformados[indices]

    # --- Creación de la máscara poligonal ---
    mascara = np.zeros_like(rostro_gris)
    puntos_casco = cv2.convexHull(puntos_rasgo)
    cv2.fillConvexPoly(mascara, puntos_casco, 255)

    # Aislamiento del rasgo facial
    # Primero se aplica la máscara para obtener solo la región del ojo
    rasgo_aislado = cv2.bitwise_and(rostro_gris, rostro_gris, mask=mascara)
    datos_ojo = rasgo_aislado[mascara > 0].reshape(-1, 1).astype(np.float32)
    ret, etiquetas, centros = cv2.kmeans(
        data=datos_ojo,
        K=K_CLUSTERS,
        bestLabels=None,
        criteria=CRITERIA,
        attempts=10, # Número de veces que el algoritmo se ejecutará con diferentes inicializaciones.
        flags=cv2.KMEANS_RANDOM_CENTERS # Inicialización aleatoria.
    )

    # 5. Reconstrucción de la Imagen Segmentada
    centros = np.uint8(centros)
    res = centros[etiquetas.flatten()].flatten()
    rasgo_segmentado_final = np.zeros_like(rostro_gris)
    rasgo_segmentado_final[mascara > 0] = res
    centros_ordenados = np.sort(centros.flatten())
    valor_pupila = centros_ordenados[0] # El centro de intensidad más bajo.
    mascara_pupila = np.zeros_like(rostro_gris)

    #Binarizar la imagen K-Means para aislar el clúster más oscuro (pupila)
    # Se binariza la imagen segmentada con un umbral muy cercano al valor_pupila.
    _, pupila_aislada = cv2.threshold(
        rasgo_segmentado_final, valor_pupila + 1, 255, cv2.THRESH_BINARY_INV
    )
    # Se asegura que solo sea visible la región del ojo (por si el umbral filtró ruido)
    pupila_aislada = cv2.bitwise_and(pupila_aislada, pupila_aislada, mask=mascara)

    ruta_segmentado = os.path.join(
        RUTA_SALIDA, f"5C.1_kmeans_{nombre_rasgo}_K{K_CLUSTERS}.jpg"
    )
    cv2.imwrite(ruta_segmentado, rasgo_segmentado_final)
    print(
        f" Segmentación K-Means (K={K_CLUSTERS}, Centros={centros.flatten()}) guardada en: "
        f"{ruta_segmentado}"
    )

    # Guardar el clúster más oscuro (Pupila)
    ruta_pupila = os.path.join(
        RUTA_SALIDA, f"5C.2_pupila_{nombre_rasgo}.jpg"
    )
    cv2.imwrite(ruta_pupila, pupila_aislada)
    print(
        f" Máscara de pupila (Clúster más oscuro) guardada en: "
        f"{ruta_pupila}"
    )

print("\n" + "=" * 70)
print("PROCESO DE SEGMENTACIÓN K-MEANS COMPLETADO")
print(f"Los resultados se han guardado en la carpeta: '{RUTA_SALIDA}'")
print("=" * 70)