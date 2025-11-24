import cv2
import numpy as np
import os
import pandas as pd

# --- Configuraciones ---
RUTA_ENTRADA = "output_fotos_segmentacion_ojos" 
RUTA_SALIDA = "output_fotos_smorfologia_descriptores"
os.makedirs(RUTA_SALIDA, exist_ok=True)

RASGOS_A_PROCESAR = ["ojo_derecho", "ojo_izquierdo"]
PREFIJO_ARCHIVO = "5.2_segmentado_" 

print("=" * 70)
print("   SCRIPT 6: SUAVIZADO AGRESIVO (SOLUCIÓN ESCALERA)")
print("=" * 70)

resultados = []

for nombre_rasgo in RASGOS_A_PROCESAR:
    print(f"\n[PROCESANDO] Rasgo: '{nombre_rasgo}'")
    
    nombre_archivo = f"{PREFIJO_ARCHIVO}{nombre_rasgo}.jpg"
    ruta_completa = os.path.join(RUTA_ENTRADA, nombre_archivo)

    if not os.path.exists(ruta_completa):
        continue

    mascara = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
    
    # 1. Morfología: Kernel 5x5 (Un poco más pequeño que 7 para no perder forma, pero efectivo)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_mejorada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
    mascara_mejorada = cv2.morphologyEx(mascara_mejorada, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2. Encontrar contornos
    contornos, _ = cv2.findContours(mascara_mejorada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        continue

    cnt_crudo = max(contornos, key=cv2.contourArea)

    # --- EL CAMBIO CLAVE: SUAVIZADO AGRESIVO ---
    perimetro_crudo = cv2.arcLength(cnt_crudo, True)
    
    # Aumentamos epsilon al 4% o 5% del perímetro. 
    # Esto elimina cualquier detalle fino (como los escalones de píxeles)
    epsilon = 0.045 * perimetro_crudo 
    cnt_suave = cv2.approxPolyDP(cnt_crudo, epsilon, True)
    
    print(f"  - Puntos reducidos de {len(cnt_crudo)} a {len(cnt_suave)} (Forma simplificada).")

    # 3. Dibujado de Alta Calidad (Anti-Aliasing)
    img_debug = cv2.cvtColor(mascara_mejorada, cv2.COLOR_GRAY2BGR)
    
    # Dibujamos el contorno crudo en ROJO (finito) para comparar
    cv2.drawContours(img_debug, [cnt_crudo], -1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    
    # Dibujamos el contorno suave en VERDE BRILLANTE
    # cv2.LINE_AA es vital para que la línea no se vea pixelada
    cv2.drawContours(img_debug, [cnt_suave], -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    ruta_debug = os.path.join(RUTA_SALIDA, f"6.1_contorno_suave_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_debug, img_debug)
    
    # --- CÁLCULOS (Usando cnt_suave) ---
    cnt = cnt_suave
    
    # Compacidad
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    if area == 0: continue
    compacidad = (perimetro ** 2) / area

    # Distancia Radial
    M = cv2.moments(cnt)
    if M["m00"] == 0: continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    distancias_radiales = []
    for punto in cnt:
        dist = np.sqrt((punto[0][0] - cx)**2 + (punto[0][1] - cy)**2)
        distancias_radiales.append(dist)
    
    distancias_radiales = np.array(distancias_radiales)
    media_dist_radial = np.mean(distancias_radiales)
    
    if media_dist_radial > 0:
        dist_radial_norm = distancias_radiales / media_dist_radial
        drn_desv_est = np.std(dist_radial_norm)
        
        # Cruces por cero
        signos = np.sign(dist_radial_norm - 1)
        cruces_por_cero = np.sum(np.abs(np.diff(signos))) / 2
        
        # Rugosidad
        indice_rugosidad = np.sum(np.abs(np.diff(dist_radial_norm)))

        resultados.append({
            "Rasgo": nombre_rasgo,
            "Compacidad": round(compacidad, 4),
            "DRN_Desv_Est": round(drn_desv_est, 4),
            "DRN_Cruces_Cero": int(cruces_por_cero),
            "DRN_Rugosidad": round(indice_rugosidad, 4)
        })

print("\n" + "=" * 70)
print(" RESULTADOS (Versión Suavizada)")
print("=" * 70)
if resultados:
    df = pd.DataFrame(resultados)
    print(df.to_string(index=False))