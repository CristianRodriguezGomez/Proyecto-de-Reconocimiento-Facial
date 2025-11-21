import cv2
import numpy as np
import os

# --- Configuraciones ---
NOMBRE_IMAGEN_ALINEADA = "output_fotos_alineacion/2_rostro_alineado.jpg"
MATRIZ_TRANSFORMACION_TEMP = "temp_matriz_transformacion.npy"
PUNTOS_CLAVE_TEMP = "temp_puntos_clave.npy"

# Carpeta de salida diferente para comparar resultados.
RUTA_SALIDA = "output_fotos_segmentacion_adaptativa"

# --- Parámetros de Umbralización Adaptativa (ajustables) 
# Debe ser un número impar mayor que 1 ( 9, 11, 15).
BLOCK_SIZE = 15

# C: Una constante que se resta de la media o media ponderada.
# Se usa para afinar el umbral.
C_CONSTANT = 2

# Crear el directorio de salida si no existe.
os.makedirs(RUTA_SALIDA, exist_ok=True)

# --- Índices de los puntos clave para rasgos faciales ---
INDICES_RASGOS = {
    "ojo_derecho": list(range(36, 42)),
    "ojo_izquierdo": list(range(42, 48)),
}

print("=" * 70)
print(" SCRIPT 5.1: SEGMENTACIÓN DE OJOS CON UMBRALIZACIÓN ADAPTATIVA")
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

# Convertir la imagen a escala de grises.
rostro_gris = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
print(f"Imagen '{NOMBRE_IMAGEN_ALINEADA}' cargada y convertida a escala de grises.")

# --- Transformación de puntos clave
puntos_homogeneos = np.hstack(
    [puntos_clave_originales, np.ones((puntos_clave_originales.shape[0], 1))]
)
puntos_clave_transformados = np.dot(
    matriz_transformacion, puntos_homogeneos.T
).T.astype(int)

print("Puntos clave transformados para coincidir con la imagen alineada.")
print("-" * 70)
print(f"Parámetros Adaptativos: blockSize={BLOCK_SIZE}, C={C_CONSTANT}")
print("-" * 70)

# 2. Bucle para procesar cada rasgo facial
for nombre_rasgo, indices in INDICES_RASGOS.items():
    print(f"\n[PROCESANDO] Segmentando '{nombre_rasgo}'...")

    # Extraer puntos clave transformados.
    puntos_rasgo = puntos_clave_transformados[indices]

    # --- Creación de la máscara poligonal 
    mascara = np.zeros_like(rostro_gris)
    puntos_casco = cv2.convexHull(puntos_rasgo)
    cv2.fillConvexPoly(mascara, puntos_casco, 255)

    # Aislamiento del rasgo facial.
    # Nota: Aquí no usamos bitwise_and, simplemente establecemos a 0 los píxeles
    # fuera de la máscara para garantizar que el cálculo adaptativo se centre solo
    # en la región del ojo. La umbralización adaptativa requiere un paso de imagen.
    rasgo_aislado = rostro_gris.copy()
    rasgo_aislado[mascara == 0] = 255 # Establecer el fondo a blanco (255) para que no afecte.
    # Una vez que tenemos la máscara, se puede aplicar la umbralización adaptativa
    # sobre el área rectangular que contiene el ojo.
    # Sin embargo, para simplicidad y enfocarnos en la umbralización:
    
    # 3. Aplicación de la Umbralización Adaptativa (CAMBIO CLAVE)
    # Se aplica la Umbralización Adaptativa sobre la imagen en escala de grises.
    rasgo_segmentado = cv2.adaptiveThreshold(
        src=rostro_gris,
        maxValue=255, # Valor que se asigna a los píxeles que superan el umbral.
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Usa una media ponderada gaussiana.
        thresholdType=cv2.THRESH_BINARY, # El píxel es 0 si es menor que el umbral, 255 si es mayor.
        blockSize=BLOCK_SIZE, # Tamaño de la vecindad.
        C=C_CONSTANT # Constante de ajuste.
    )

    # 4. Reaplicación de la Máscara
    # Es crucial volver a aplicar la máscara después de la umbralización adaptativa
    # para asegurar que SÓLO el ojo sea visible, ya que la umbralización se aplicó a todo el rostro.
    rasgo_segmentado_final = cv2.bitwise_and(rasgo_segmentado, rasgo_segmentado, mask=mascara)


    # Guardar la máscara generada 
    ruta_mascara = os.path.join(RUTA_SALIDA, f"5B.1_mascara_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_mascara, mascara)
    print(f" Máscara guardada en: {ruta_mascara}")

    # Guardar la imagen final segmentada
    ruta_segmentado = os.path.join(
        RUTA_SALIDA, f"5B.2_segmentado_adaptativo_{nombre_rasgo}.jpg"
    )
    cv2.imwrite(ruta_segmentado, rasgo_segmentado_final)
    print(
        f" Rasgo segmentado (Adaptativo) guardado en: "
        f"{ruta_segmentado}"
    )

print("\n" + "=" * 70)
print("PROCESO DE SEGMENTACIÓN ADAPTATIVA COMPLETADO")
print(f"Los resultados se han guardado en la carpeta: '{RUTA_SALIDA}'")
print("=" * 70)