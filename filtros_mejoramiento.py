import cv2  # OpenCV para procesamiento de imágenes y filtros.
import numpy as np

# --- Configuraciones ---
NOMBRE_IMAGEN_ALINEADA = "temp_rostro_alineado.jpg"

print("=" * 70)
print("   SCRIPT 3: APLICACIÓN DE FILTROS DE MEJORAMIENTO")
print("=" * 70)
print("\nAplicando filtros secuencialmente:")
print("  1. Filtro Estadístico: Mediana (Reduce ruido sal y pimienta)")
print("  2. Filtro Suavizante: Gaussiano (Suaviza la imagen)")
print("  3. Filtro Realzante: CLAHE (Mejora contraste local)\n")

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
# FILTRO 3: REALZANTE - CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ============================================================================
print("\n[3/3] Aplicando Filtro REALZANTE (CLAHE)...")
print("      → Mejora el contraste local de forma adaptativa")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imagen_clahe = clahe.apply(imagen_gaussiana)

# Guardar salida del tercer filtro
output_3 = "3.3_filtro_clahe.jpg"
rostro_clahe_bgr = cv2.cvtColor(imagen_clahe, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_3, rostro_clahe_bgr)
print(f"      ✓ Guardado: {output_3}")


# ============================================================================
# IMAGEN FINAL PROCESADA
# ============================================================================
print("\n" + "=" * 70)
print("✓ PROCESO COMPLETADO")
print("=" * 70)
print("\nArchivos generados:")
print(f"  1. {output_1}  ← Después del filtro Mediana (Estadístico)")
print(f"  2. {output_2}  ← Después del filtro Gaussiano (Suavizante)")
print(f"  3. {output_3}  ← Después del filtro CLAHE (Realzante)")
print("\nLa imagen final ha pasado por los 3 tipos de filtros secuencialmente.")
print("=" * 70)