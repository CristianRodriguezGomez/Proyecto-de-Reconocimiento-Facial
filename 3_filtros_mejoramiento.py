import cv2  # OpenCV para procesamiento de imágenes y filtros.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

NOMBRE_IMAGEN_ALINEADA = "output_fotos_alineacion/2_rostro_alineado.jpg"  # Asegúrate que este archivo existe.
RUTA_SALIDA_FILTROS = "output_fotos_filtrado"

# Crear el directorio de salida si no existe
os.makedirs(RUTA_SALIDA_FILTROS, exist_ok=True)

print("=" * 70)
print("   SCRIPT 3: APLICACIÓN DE FILTROS DE MEJORAMIENTO")
print("=" * 70)
print("\nGenerando 4 esquemas de filtrado:")
print("  1. Filtro Estadístico: Mediana")
print("  2. Filtro Suavizante: Gaussiano")
print("  3. Filtro Realzante: High-Boost")
print("  4. Combinación Secuencial de los 3 filtros\n")

# 1. Cargar la imagen alineada
rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)
if rostro_alineado is None:
    print("ERROR: No se pudo cargar el rostro alineado.")
    print(f"Asegúrate de que existe el archivo: {NOMBRE_IMAGEN_ALINEADA}")
    exit()

# Convertir a escala de grises (requerido para muchos filtros)
gris_rostro = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
print(f"Imagen cargada: {gris_rostro.shape}")


# FILTROS INDEPENDIENTES Y SECUENCIAL

# 1. Mediana
print("[1/4] Aplicando Filtro Mediana (Estadístico)...")
imagen_mediana = cv2.medianBlur(gris_rostro, 5)
output_1 = os.path.join(RUTA_SALIDA_FILTROS, "3.1_filtro_mediana.jpg")
rostro_mediana_bgr = cv2.cvtColor(imagen_mediana, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_1, rostro_mediana_bgr)
print(f"Guardado: {output_1}")

# 2. Gaussiano
print("[2/4] Aplicando Filtro Gaussiano (Suavizante)...")
imagen_gaussiana_ind = cv2.GaussianBlur(gris_rostro, (5, 5), 0)
output_2 = os.path.join(RUTA_SALIDA_FILTROS, "3.2_filtro_gaussiano.jpg")
rostro_gaussiano_bgr_ind = cv2.cvtColor(imagen_gaussiana_ind, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_2, rostro_gaussiano_bgr_ind)
print(f"Guardado: {output_2}")

# 3. High-Boost
print("[3/4] Aplicando Filtro High-Boost (Realzante)...")
A = 1.5
blur_low_ind = cv2.GaussianBlur(gris_rostro, (5, 5), 0)
mask_ind = cv2.subtract(gris_rostro, blur_low_ind)
imagen_highboost_ind = cv2.addWeighted(gris_rostro, 1.0, mask_ind, A, 0)
imagen_highboost_ind = np.clip(imagen_highboost_ind, 0, 255).astype(np.uint8)
output_3 = os.path.join(RUTA_SALIDA_FILTROS, "3.3_filtro_highboost.jpg")
rostro_highboost_bgr_ind = cv2.cvtColor(imagen_highboost_ind, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_3, rostro_highboost_bgr_ind)
print(f"Guardado: {output_3}")

# COMBINACIÓN SECUENCIAL DE LOS 3 FILTROS
print("[4/4] Aplicando Combinación Secuencial (Mediana -> Gaussiano -> High-Boost)...")
imagen_mediana_seq = cv2.medianBlur(gris_rostro, 5)
imagen_gaussiana_seq = cv2.GaussianBlur(imagen_mediana_seq, (5, 5), 0)
blur_low_seq = cv2.GaussianBlur(imagen_gaussiana_seq, (5, 5), 0)
mask_seq = cv2.subtract(imagen_gaussiana_seq, blur_low_seq)
imagen_highboost_seq = cv2.addWeighted(imagen_gaussiana_seq, 1.0, mask_seq, A, 0)
imagen_highboost_seq = np.clip(imagen_highboost_seq, 0, 255).astype(np.uint8)
output_4 = os.path.join(RUTA_SALIDA_FILTROS, "3.4_filtro_combinado.jpg")
rostro_combinado_bgr = cv2.cvtColor(imagen_highboost_seq, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_4, rostro_combinado_bgr)
print(f"Guardado: {output_4}")

# CREAR PDF CON LAS 4 IMÁGENES Y SUS TÍTULOS
print("\nCreando PDF con los resultados...")
pdf_path = 'filtros_resultados.pdf'
with PdfPages(pdf_path) as pdf:
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Resultados de Filtros de Mejoramiento', fontsize=16)
    imgs = [rostro_mediana_bgr, rostro_gaussiano_bgr_ind, rostro_highboost_bgr_ind, rostro_combinado_bgr]
    titles = [
        'Filtro Mediana (Estadístico)',
        'Filtro Gaussiano (Suavizante)',
        'Filtro High-Boost (Realzante)',
        'Combinación Secuencial de los 3 Filtros'
    ]
    for ax, img, title in zip(axs.flat, imgs, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)
print(f"PDF guardado en: {pdf_path}")

# IMAGEN FINAL PROCESADA
print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)
print(f"\nLas imágenes de los filtros se han guardado en la carpeta: '{RUTA_SALIDA_FILTROS}'")
print(f"También se ha generado un PDF resumen: '{pdf_path}'")
print("=" * 70)