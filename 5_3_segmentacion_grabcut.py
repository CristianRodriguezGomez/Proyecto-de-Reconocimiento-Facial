import cv2
import numpy as np
import os

# --- Configuraciones ---
NOMBRE_IMAGEN_ALINEADA = "output_fotos_alineacion/2_rostro_alineado.jpg"
MATRIZ_TRANSFORMACION_TEMP = "temp_matriz_transformacion.npy"
PUNTOS_CLAVE_TEMP = "temp_puntos_clave.npy"
RUTA_SALIDA = "output_fotos_segmentacion_rasgos"

# Crear el directorio de salida si no existe
os.makedirs(RUTA_SALIDA, exist_ok=True)

# --- Índices de los puntos clave (landmarks) para dlib 68 puntos ---
INDICES_RASGOS = {
    "ojo_derecho": list(range(36, 42)),
    "ojo_izquierdo": list(range(42, 48)),
}

print("=" * 70)
print("   SCRIPT 5.3: SEGMENTACIÓN DE OJOS (GrabCut) - V2 CORREGIDO")
print("=" * 70)

# 1. Cargar la imagen alineada y los puntos clave
try:
    if not os.path.exists(NOMBRE_IMAGEN_ALINEADA):
        raise FileNotFoundError(f"No existe la imagen: {NOMBRE_IMAGEN_ALINEADA}")
        
    rostro_alineado = cv2.imread(NOMBRE_IMAGEN_ALINEADA)
    puntos_clave_originales = np.load(PUNTOS_CLAVE_TEMP)
    matriz_transformacion = np.load(MATRIZ_TRANSFORMACION_TEMP)

    if rostro_alineado is None:
        raise ValueError("La imagen se cargó como None. Revisa el formato.")

except Exception as e:
    print(f"ERROR CRÍTICO: {e}")
    print("Asegúrate de haber ejecutado los scripts de alineación primero.")
    exit()

print(f"Imagen '{NOMBRE_IMAGEN_ALINEADA}' cargada correctamente.")
im_h, im_w = rostro_alineado.shape[:2]

# Transformar los puntos clave
puntos_homogeneos = np.hstack([puntos_clave_originales, np.ones((puntos_clave_originales.shape[0], 1))])
puntos_clave_transformados = np.dot(matriz_transformacion, puntos_homogeneos.T).T.astype(int)
print("Puntos clave transformados.")

# 2. Bucle para procesar cada ojo
for nombre_rasgo, indices in INDICES_RASGOS.items():
    print(f"\n[PROCESANDO] Segmentando '{nombre_rasgo}'...")

    # Obtener los puntos de este rasgo
    puntos_rasgo = puntos_clave_transformados[indices]
    
    # --- 1. CALCULAR ROI (CON MARGEN AJUSTADO) ---
    (x, y, w, h) = cv2.boundingRect(puntos_rasgo)
    
    # CORRECCIÓN 1: Margen más ajustado para evitar orejas/pelo
    # 25% de la altura del ojo en lugar de 60%, mínimo 5px
    margen = max(5, int(h * 0.25)) 

    x_inicio = max(0, x - margen)
    y_inicio = max(0, y - margen)
    x_fin = min(im_w, x + w + margen)
    y_fin = min(im_h, y + h + margen)
    
    rect_w = x_fin - x_inicio
    rect_h = y_fin - y_inicio
    rect_seguro = (x_inicio, y_inicio, rect_w, rect_h)

    # --- PASO DE DEPURACIÓN VISUAL ---
    debug_img = rostro_alineado.copy()
    for (px, py) in puntos_rasgo:
        cv2.circle(debug_img, (px, py), 1, (0, 255, 0), -1)
    cv2.rectangle(debug_img, (x_inicio, y_inicio), (x_fin, y_fin), (0, 0, 255), 1)
    ruta_debug = os.path.join(RUTA_SALIDA, f"debug_rect_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_debug, debug_img)
    print(f"  > [DEBUG] Rectángulo ajustado guardado en: {ruta_debug}")


    # --- 2. PROCESO GRABCUT ---
    mascara_grabcut = np.zeros(rostro_alineado.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    iteraciones = 5
    cv2.grabCut(rostro_alineado, mascara_grabcut, rect_seguro, bgdModel, fgdModel, iteraciones, cv2.GC_INIT_WITH_RECT)

    # Generar máscara binaria (1 y 3 son primer plano)
    mascara_binaria = np.where((mascara_grabcut == 1) | (mascara_grabcut == 3), 255, 0).astype('uint8')
    
    # --- Verificación y Plan de Respaldo (Otsu) ---
    conteo_blancos = cv2.countNonZero(mascara_binaria)
    
    if conteo_blancos < 10: 
        print(f"  ADVERTENCIA: Fallo GrabCut. Usando Otsu.")
        roi_ojo = rostro_alineado[y_inicio:y_fin, x_inicio:x_fin]
        roi_gray = cv2.cvtColor(roi_ojo, cv2.COLOR_BGR2GRAY)
        _, mascara_otsu = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mascara_binaria = np.zeros(rostro_alineado.shape[:2], np.uint8)
        mascara_binaria[y_inicio:y_fin, x_inicio:x_fin] = mascara_otsu

    # --- 3. LIMPIEZA FINAL (CORRECCIÓN DE RUIDO) ---
    # Paso A: Limpieza morfológica suave para quitar ruido pequeño
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_binaria = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel)

    # Paso B: Quedarse solo con el contorno más grande (El Ojo)
    # Esto elimina la oreja o el pelo si quedaron separados del ojo
    contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contornos:
        # Encontrar el contorno con mayor área
        cnt_mayor = max(contornos, key=cv2.contourArea)
        
        # Crear una nueva máscara vacía
        mascara_limpia = np.zeros_like(mascara_binaria)
        
        # Dibujar solo el contorno mayor (relleno)
        cv2.drawContours(mascara_limpia, [cnt_mayor], -1, 255, thickness=cv2.FILLED)
        
        # Actualizar la máscara final
        mascara_binaria = mascara_limpia
    else:
        print("  ADVERTENCIA: No se encontraron contornos finales.")

    # Guardar el resultado final
    ruta_segmentado = os.path.join(RUTA_SALIDA, f"5.6_grabcut_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_segmentado, mascara_binaria)
    print(f"  - Máscara limpia guardada en: {ruta_segmentado}")

print("\n" + "=" * 70)
print("PROCESO COMPLETADO - Revisa output_fotos_segmentacion_rasgos")
print("=" * 70)