import cv2  # OpenCV para procesamiento de imágenes y filtros.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Configuraciones ---
NOMBRE_IMAGEN_ALINEADA = "2_rostro_alineado.jpg"  # Asegúrate que este archivo existe.

print("=" * 70)
print("   SCRIPT 3: APLICACIÓN DE FILTROS DE MEJORAMIENTO")
print("=" * 70)
print("\nAplicando filtros secuencialmente:")
print("  1. Filtro Estadístico: Mediana (Reduce ruido sal y pimienta)")
print("  2. Filtro Suavizante: Gaussiano (Suaviza la imagen)")
print("  3. Filtro Realzante: High-Boost (Realce mediante enmascaramiento de alta frecuencia)\n")

# 1. Cargar la imagen alineada
rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)
if rostro_alineado is None:
    print("ERROR: No se pudo cargar el rostro alineado.")
    print(f"Asegúrate de que existe el archivo: {NOMBRE_IMAGEN_ALINEADA}")
    exit()

# Convertir a escala de grises (requerido para muchos filtros)
gris_rostro = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
print(f"✓ Imagen cargada: {gris_rostro.shape}")


# ============================================================================
# FILTRO 1: ESTADÍSTICO - Filtro de Mediana
# ============================================================================
print("\n[1/3] Aplicando Filtro ESTADÍSTICO (Mediana - ksize=5)...")
print("      → Elimina ruido tipo 'sal y pimienta' sin desenfocar bordes")

imagen_mediana = cv2.medianBlur(gris_rostro, 5)

# Guardar salida del primer filtro
output_1 = "3.1_filtro_mediana.jpg"
rostro_mediana_bgr = cv2.cvtColor(imagen_mediana, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_1, rostro_mediana_bgr)
print(f"      ✓ Guardado: {output_1}")


# ============================================================================
# FILTRO 2: SUAVIZANTE - Filtro Gaussiano
# ============================================================================
print("\n[2/3] Aplicando Filtro SUAVIZANTE (Gaussiano - ksize=5)...")
print("      → Suaviza la imagen reduciendo detalles y ruido de alta frecuencia")

imagen_gaussiana = cv2.GaussianBlur(imagen_mediana, (5, 5), 0)

# Guardar salida del segundo filtro
output_2 = "3.2_filtro_gaussiano.jpg"
rostro_gaussiano_bgr = cv2.cvtColor(imagen_gaussiana, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_2, rostro_gaussiano_bgr)
print(f"      ✓ Guardado: {output_2}")


# ============================================================================
# FILTRO 3: REALZANTE - High-Boost
# ============================================================================
print("\n[3/3] Aplicando Filtro REALZANTE (High-Boost)...")
print("      → Realce por enmascaramiento de alta frecuencia (A >= 1.0)")

# Parámetro de realce (A > 1.0: más realce)
A = 1.5  # puedes ajustar entre 1.0 y 3.0 según necesidad

# Usamos un desenfoque para obtener la versión de baja frecuencia
blur_low = cv2.GaussianBlur(imagen_gaussiana, (5, 5), 0)

# Máscara de alta frecuencia
mask = cv2.subtract(imagen_gaussiana, blur_low)

# Imagen high-boost: original + A * mask
imagen_highboost = cv2.addWeighted(imagen_gaussiana, 1.0, mask, A, 0)

# Asegurar rango válido y tipo uint8
imagen_highboost = np.clip(imagen_highboost, 0, 255).astype(np.uint8)

# Guardar salida del tercer filtro
output_3 = "3.3_filtro_highboost.jpg"
rostro_highboost_bgr = cv2.cvtColor(imagen_highboost, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_3, rostro_highboost_bgr)
print(f"      ✓ Guardado: {output_3}")


# ============================================================================
# FILTROS INDEPENDIENTES
# ============================================================================
# 1. Mediana
imagen_mediana = cv2.medianBlur(gris_rostro, 5)
output_1 = "3.1_filtro_mediana.jpg"
rostro_mediana_bgr = cv2.cvtColor(imagen_mediana, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_1, rostro_mediana_bgr)

# 2. Gaussiano
imagen_gaussiana_ind = cv2.GaussianBlur(gris_rostro, (5, 5), 0)
output_2 = "3.2_filtro_gaussiano.jpg"
rostro_gaussiano_bgr_ind = cv2.cvtColor(imagen_gaussiana_ind, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_2, rostro_gaussiano_bgr_ind)

# 3. High-Boost
A = 1.5
blur_low_ind = cv2.GaussianBlur(gris_rostro, (5, 5), 0)
mask_ind = cv2.subtract(gris_rostro, blur_low_ind)
imagen_highboost_ind = cv2.addWeighted(gris_rostro, 1.0, mask_ind, A, 0)
imagen_highboost_ind = np.clip(imagen_highboost_ind, 0, 255).astype(np.uint8)
output_3 = "3.3_filtro_highboost.jpg"
rostro_highboost_bgr_ind = cv2.cvtColor(imagen_highboost_ind, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_3, rostro_highboost_bgr_ind)

# ============================================================================
# COMBINACIÓN SECUENCIAL DE LOS 3 FILTROS
# ============================================================================
imagen_mediana_seq = cv2.medianBlur(gris_rostro, 5)
imagen_gaussiana_seq = cv2.GaussianBlur(imagen_mediana_seq, (5, 5), 0)
blur_low_seq = cv2.GaussianBlur(imagen_gaussiana_seq, (5, 5), 0)
mask_seq = cv2.subtract(imagen_gaussiana_seq, blur_low_seq)
imagen_highboost_seq = cv2.addWeighted(imagen_gaussiana_seq, 1.0, mask_seq, A, 0)
imagen_highboost_seq = np.clip(imagen_highboost_seq, 0, 255).astype(np.uint8)
output_4 = "3.4_filtro_combinado.jpg"
rostro_combinado_bgr = cv2.cvtColor(imagen_highboost_seq, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_4, rostro_combinado_bgr)

# ============================================================================
# CREAR PDF CON LAS 4 IMÁGENES Y SUS TÍTULOS
# ============================================================================
with PdfPages('filtros_resultados.pdf') as pdf:
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

# ============================================================================
# IMAGEN FINAL PROCESADA
# ============================================================================
print("\n" + "=" * 70)
print("✓ PROCESO COMPLETADO")
print("=" * 70)
print("\nArchivos generados:")
print(f"  1. {output_1}  ← Filtro Mediana (Estadístico)")
print(f"  2. {output_2}  ← Filtro Gaussiano (Suavizante)")
print(f"  3. {output_3}  ← Filtro High-Boost (Realzante)")
print(f"  4. {output_4}  ← Combinación Secuencial de los 3 Filtros")
print(f"  PDF: filtros_resultados.pdf ← Las 4 imágenes con sus títulos")
print("\nLa imagen final ha pasado por los 3 tipos de filtros secuencialmente y también se guardan los resultados individuales.")
print("=" * 70)
