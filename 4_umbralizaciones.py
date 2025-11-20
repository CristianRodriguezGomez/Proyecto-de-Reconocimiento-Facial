import cv2
import numpy as np
import os

# --- Configuraciones de Entrada ---
ARCHIVOS_ENTRADA = [
    "3.1_filtro_mediana.jpg",  # Esquema 1: Mediana
    "3.2_filtro_gaussiano.jpg", # Esquema 2: Mediana + Gaussiano
    "3.3_filtro_highboost.jpg", # Esquema 3: Mediana + Gaussiano + CLAHE
    "3.4_filtro_combinado.jpg", # Esquema 3: Mediana + Gaussiano + CLAHE
]

RUTA_SALIDA_UMBRALIZACION = "output_fotos_umbralizacion"
os.makedirs(RUTA_SALIDA_UMBRALIZACION, exist_ok=True)
UMBRAL_GLOBAL = 127 # Umbral fijo para la técnica global simple (un valor común)

print("=" * 70)
print("Script 4: SEGMENTACIÓN (Umbralización Global y Otsu)")
print("=" * 70)

for nombre_archivo in ARCHIVOS_ENTRADA:
    print(f"\n[PROCESANDO] Esquema: {nombre_archivo}")
    
    # 1. Cargar la imagen (ya en BGR, pero la convertimos a GRIS para umbralización)
    img_bgr = cv2.imread(nombre_archivo)
    
    if img_bgr is None:
        print(f"ERROR: No se pudo cargar el archivo: {nombre_archivo}. Omtiendo.")
        continue
    
    # La umbralización se aplica a imágenes en escala de grises
    img_gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- Umbralización Global Simple ---
    # La segmentación se hace en dos regiones: R1 (fondo=0) y R2 (rostro=255)
    
    # cv2.THRESH_BINARY: Si el pixel > UMBRAL, es 255 (Rostro), sino 0 (Fondo).
    # La función retorna el umbral usado y la imagen umbralizada
    ret, img_global = cv2.threshold(img_gris, UMBRAL_GLOBAL, 255, cv2.THRESH_BINARY)
    
    # Guardar el resultado
    nombre_salida_global = os.path.join(RUTA_SALIDA_UMBRALIZACION, "4.1_segmentado_GLOBAL_" + os.path.basename(nombre_archivo))
    cv2.imwrite(nombre_salida_global, img_global)
    print(f" Umbral Global ({UMBRAL_GLOBAL}): Guardado en {nombre_salida_global}")
    
    
    # --- Umbralización de Otsu (Umbralización Automática) ---
    # cv2.THRESH_OTSU: El umbral (ret_otsu) se calcula automáticamente para
    # maximizar la varianza inter-clase, idealmente separando fondo y objeto.
    
    # Combina THRESH_BINARY y THRESH_OTSU. El valor de UMBRAL_GLOBAL es ignorado
    # ya que Otsu calcula el suyo propio.
    ret_otsu, img_otsu = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Guardar el resultado
    nombre_salida_otsu = os.path.join(RUTA_SALIDA_UMBRALIZACION, "4.2_segmentado_OTSU_" + os.path.basename(nombre_archivo))
    cv2.imwrite(nombre_salida_otsu, img_otsu)
    print(f"Umbral de Otsu ({ret_otsu:.2f}): Guardado en {nombre_salida_otsu}")


print("\n" + "=" * 70)
print("SEGMENTACIÓN COMPLETADA")
print("=" * 70)
print("Se han generado 6 imágenes de salida (3 por técnica de umbralización).")
print("Se recomienda **inspeccionar visualmente** las imágenes guardadas para evaluar el desempeño.")
print("=" * 70)