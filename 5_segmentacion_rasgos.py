import cv2
import numpy as np
import os

# --- Configuraciones ---
NOMBRE_IMAGEN_ALINEADA = "2_rostro_alineado.jpg"
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
print("   SCRIPT 5: SEGMENTACIÓN DE RASGOS FACIALES (Ojos, Nariz, Boca)")
print("=" * 70)

# 1. Cargar la imagen alineada y los puntos clave
try:
    # Cargamos la imagen alineada, que es la base para la segmentación
    rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)
    # Cargamos los puntos clave originales para obtener las coordenadas
    puntos_clave_originales = np.load(PUNTOS_CLAVE_TEMP) # Coordenadas en la imagen original
    # Cargamos la matriz que se usó para alinear el rostro
    matriz_transformacion = np.load(MATRIZ_TRANSFORMACION_TEMP)
except FileNotFoundError:
    print(f"ERROR: No se encontraron los archivos necesarios ('{NOMBRE_IMAGEN_ALINEADA}', '{PUNTOS_CLAVE_TEMP}', '{MATRIZ_TRANSFORMACION_TEMP}').")
    print("Asegúrate de haber ejecutado los scripts 1 y 2 primero.")
    exit()

if rostro_alineado is None:
    print(f"ERROR: No se pudo cargar la imagen '{NOMBRE_IMAGEN_ALINEADA}'.")
    exit()

# La umbralización se realiza sobre imágenes en escala de grises
rostro_gris = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
print(f"Imagen '{NOMBRE_IMAGEN_ALINEADA}' cargada y convertida a escala de grises.")


# --- CORRECCIÓN CLAVE: Transformar los puntos clave ---
# Aplicamos la misma transformación afín a los 68 puntos clave para que coincidan con la imagen alineada.
# 1. Añadimos una columna de '1' para hacerlos homogéneos (necesario para la multiplicación matricial).
puntos_homogeneos = np.hstack([puntos_clave_originales, np.ones((puntos_clave_originales.shape[0], 1))])
# 2. Aplicamos la transformación M (2x3) a los puntos (Nx3). El resultado es (Nx2).
puntos_clave_transformados = np.dot(matriz_transformacion, puntos_homogeneos.T).T.astype(int)
print("Puntos clave transformados para coincidir con la imagen alineada.")


# 2. Bucle para procesar cada rasgo facial
for nombre_rasgo, indices in INDICES_RASGOS.items():
    print(f"\n[PROCESANDO] Segmentando '{nombre_rasgo}'...")

    # Extraer los puntos clave YA TRANSFORMADOS para el rasgo actual
    puntos_rasgo = puntos_clave_transformados[indices]

    # --- Creación de la Máscara Poligonal ---
    # Se crea una imagen negra del mismo tamaño que el rostro alineado
    mascara = np.zeros_like(rostro_gris)
    
    # cv2.convexHull encuentra la envoltura convexa de los puntos del rasgo.
    # Esto es importante para crear un polígono cerrado y simple.
    puntos_casco = cv2.convexHull(puntos_rasgo)
    
    # Rellenamos el polígono en la máscara con color blanco (255)
    cv2.fillConvexPoly(mascara, puntos_casco, 255)
    
    # --- Aislamiento del Rasgo ---
    # Usamos la máscara para quedarnos solo con la región del rasgo en la imagen.
    # El resto de la imagen se vuelve negro (0).
    rasgo_aislado = cv2.bitwise_and(rostro_gris, rostro_gris, mask=mascara)

    # --- Segmentación con Umbralización de Otsu ---
    # Otsu es ideal aquí porque la imagen tiene dos clases claras: el rasgo y el fondo negro.
    # El umbral se calcula automáticamente para separar óptimamente estas dos clases.
    # El primer argumento del umbral (0) es ignorado cuando se usa THRESH_OTSU.
    umbral_otsu, rasgo_segmentado = cv2.threshold(rasgo_aislado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Guardar los resultados
    # 1. La máscara creada
    ruta_mascara = os.path.join(RUTA_SALIDA, f"5.1_mascara_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_mascara, mascara)
    print(f"  - Máscara guardada en: {ruta_mascara}")

    # 2. El rasgo segmentado final
    ruta_segmentado = os.path.join(RUTA_SALIDA, f"5.2_segmentado_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_segmentado, rasgo_segmentado)
    print(f"  - Rasgo segmentado (Umbral Otsu={umbral_otsu:.2f}) guardado en: {ruta_segmentado}")

    # --- NUEVO: Mejora con Operadores Morfológicos (Cierre) ---
    # El Cierre (Closing) es una dilatación seguida de una erosión.
    # Es útil para rellenar pequeños agujeros dentro de los objetos segmentados.
    # 1. Definimos un 'kernel' o elemento estructurante. Es una pequeña matriz que define el vecindario de la operación.
    kernel = np.ones((5,5), np.uint8) # Un cuadrado de 5x5 es un buen punto de partida.
    
    # 2. Aplicamos la operación de Cierre Morfológico.
    rasgo_mejorado = cv2.morphologyEx(rasgo_segmentado, cv2.MORPH_CLOSE, kernel)

    # 3. Guardamos el resultado mejorado.
    ruta_mejorado = os.path.join(RUTA_SALIDA, f"5.3_morfologia_cierre_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_mejorado, rasgo_mejorado)
    print(f"  - Mejora morfológica (Cierre) guardada en: {ruta_mejorado}")

print("\n" + "=" * 70)
print("PROCESO DE SEGMENTACIÓN DE RASGOS COMPLETADO")
print(f"Los resultados se han guardado en la carpeta: '{RUTA_SALIDA}'")
print("=" * 70)