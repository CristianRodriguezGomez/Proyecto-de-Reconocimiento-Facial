import cv2
import numpy as np
import os

# --- Configuraciones ---
# Nombre de la imagen alineada que se usará para la segmentación.
NOMBRE_IMAGEN_ALINEADA = "output_fotos_alineacion/2_rostro_alineado.jpg"

# Archivos temporales donde se guardan matriz de transformación y puntos clave.
MATRIZ_TRANSFORMACION_TEMP = "temp_matriz_transformacion.npy"
PUNTOS_CLAVE_TEMP = "temp_puntos_clave.npy"

# Carpeta de salida donde se guardarán las máscaras y segmentaciones.
RUTA_SALIDA = "output_fotos_segmentacion_ojos"

# Crear el directorio de salida si no existe (PEP 8: os.makedirs con exist_ok=True).
os.makedirs(RUTA_SALIDA, exist_ok=True)

# --- Índices de los puntos clave para rasgos faciales ---
# Diccionario que agrupa los índices de dlib (modelo de 68 puntos) para cada ojo.
# PEP 8: nombres descriptivos y estructura clara.
INDICES_RASGOS = {
    "ojo_derecho": list(range(36, 42)),
    "ojo_izquierdo": list(range(42, 48)),
}

print("=" * 70)
print("   SCRIPT 5: SEGMENTACIÓN DE OJOS")
print("=" * 70)

# 1. Cargar la imagen alineada y puntos clave
try:
    # Cargar imagen alineada del rostro.
    rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)

    # Cargar puntos clave originales (en coordenadas de la imagen previa a alineación).
    puntos_clave_originales = np.load(PUNTOS_CLAVE_TEMP)

    # Cargar la matriz de transformación usada para alinear la imagen.
    matriz_transformacion = np.load(MATRIZ_TRANSFORMACION_TEMP)

except FileNotFoundError:
    # PEP 8: Mensaje claro sobre archivos faltantes.
    print(
        f"ERROR: No se encontraron los archivos necesarios "
        f"('{NOMBRE_IMAGEN_ALINEADA}', '{PUNTOS_CLAVE_TEMP}', "
        f"'{MATRIZ_TRANSFORMACION_TEMP}')."
    )
    print("Asegúrate de haber ejecutado los scripts 1 y 2 primero.")
    exit()

# Verificación adicional por si la imagen no cargó correctamente.
if rostro_alineado is None:
    print(f"ERROR: No se pudo cargar la imagen '{NOMBRE_IMAGEN_ALINEADA}'.")
    exit()

# Convertir la imagen a escala de grises para umbralización posterior.
rostro_gris = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
print(f"Imagen '{NOMBRE_IMAGEN_ALINEADA}' cargada y convertida a escala de grises.")

# --- Transformación de puntos clave (paso crucial) ---
# Convertir puntos a coordenadas homogéneas para aplicar matriz (PEP 8: claridad matemática).
puntos_homogeneos = np.hstack(
    [puntos_clave_originales, np.ones((puntos_clave_originales.shape[0], 1))]
)

# Aplicar la transformación afín (2x3) → resultado en 2D.
puntos_clave_transformados = np.dot(
    matriz_transformacion, puntos_homogeneos.T
).T.astype(int)

print("Puntos clave transformados para coincidir con la imagen alineada.")

# 2. Bucle para procesar cada rasgo facial
for nombre_rasgo, indices in INDICES_RASGOS.items():
    print(f"\n[PROCESANDO] Segmentando '{nombre_rasgo}'...")

    # Extraer puntos clave transformados correspondientes al rasgo actual.
    puntos_rasgo = puntos_clave_transformados[indices]

    # --- Creación de la máscara poligonal ---
    # Crear imagen negra para la máscara.
    mascara = np.zeros_like(rostro_gris)

    # cv2.convexHull obtiene polígono convexo del rasgo (estable y simple).
    puntos_casco = cv2.convexHull(puntos_rasgo)

    # Rellenar el polígono con valor blanco (255).
    cv2.fillConvexPoly(mascara, puntos_casco, 255)

    # --- Aislamiento del rasgo facial ---
    # Con la máscara se extrae únicamente la región del rasgo.
    rasgo_aislado = cv2.bitwise_and(rostro_gris, rostro_gris, mask=mascara)

    # --- Segmentación usando Umbralización Otsu ---
    # Otsu calcula automáticamente el umbral óptimo para separar el rasgo del fondo.
    umbral_otsu, rasgo_segmentado = cv2.threshold(
        rasgo_aislado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Guardar la máscara generada.
    ruta_mascara = os.path.join(RUTA_SALIDA, f"5.1_mascara_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_mascara, mascara)
    print(f"  - Máscara guardada en: {ruta_mascara}")

    # Guardar la imagen final segmentada.
    ruta_segmentado = os.path.join(
        RUTA_SALIDA, f"5.2_segmentado_{nombre_rasgo}.jpg"
    )
    cv2.imwrite(ruta_segmentado, rasgo_segmentado)
    print(
        f"  - Rasgo segmentado (Umbral Otsu={umbral_otsu:.2f}) guardado en: "
        f"{ruta_segmentado}"
    )

print("\n" + "=" * 70)
print("PROCESO DE SEGMENTACIÓN DE RASGOS COMPLETADO")
print(f"Los resultados se han guardado en la carpeta: '{RUTA_SALIDA}'")
print("=" * 70)
