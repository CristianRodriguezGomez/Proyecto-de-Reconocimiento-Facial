import cv2
import numpy as np
import os
import pandas as pd

# --- Configuraciones ---
RUTA_ENTRADA = "output_fotos_segmentacion_rasgos"
RUTA_SALIDA= "output_fotos_smorfologia_descriptores"
os.makedirs(RUTA_SALIDA, exist_ok=True)

RASGOS_A_PROCESAR = ["ojo_derecho", "ojo_izquierdo"]

# El script anterior (5.6) guarda los archivos con este prefijo
PREFIJO_ARCHIVO = "5.6_grabcut_"

print("=" * 70)
print("   SCRIPT 6: OPERADORES MORFOLÓGICOS Y DESCRIPTORES DE FORMA (FINAL)")
print("=" * 70)

# Lista para almacenar los resultados
resultados = []

for nombre_rasgo in RASGOS_A_PROCESAR:
    print(f"\n[PROCESANDO] Rasgo: '{nombre_rasgo}'")
    
    # 1. Cargar la máscara segmentada
    nombre_archivo = f"{PREFIJO_ARCHIVO}{nombre_rasgo}.jpg"
    ruta_completa = os.path.join(RUTA_ENTRADA, nombre_archivo)

    if not os.path.exists(ruta_completa):
        print(f"  ADVERTENCIA: No se encontró el archivo {ruta_completa}. Omitiendo.")
        continue

    mascara = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
    if mascara is None:
        print(f"  ERROR: No se pudo cargar la imagen {ruta_completa}. Omitiendo.")
        continue
    
    print(f"  - Archivo '{nombre_archivo}' cargado.")

    # 2. Aplicar operadores morfológicos (AJUSTADO PARA CONSERVAR FORMA)
    #    Usamos un kernel pequeño (3x3) y solo 1 iteración para no deformar el ojo
    #    que ya viene limpio del GrabCut.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Cierre (Dilatación -> Erosión): Rellena pequeños huecos negros DENTRO del ojo
    mascara_mejorada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Apertura (Erosión -> Dilatación): Elimina ruido blanco FUERA del ojo (suave)
    mascara_mejorada = cv2.morphologyEx(mascara_mejorada, cv2.MORPH_OPEN, kernel, iterations=1)
    
    print("  - Operadores morfológicos suaves aplicados.")

    # 3. Encontrar el contorno principal
    contornos, _ = cv2.findContours(mascara_mejorada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        print("  ERROR: No se encontraron contornos en la máscara mejorada. Omitiendo.")
        continue

    # Suponemos que el contorno más grande es el del rasgo de interés
    cnt = max(contornos, key=cv2.contourArea)
    print(f"  - Contorno principal encontrado con {len(cnt)} puntos.")

    # --- Guardar imagen de depuración con el contorno ---
    img_debug = cv2.cvtColor(mascara_mejorada, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_debug, [cnt], -1, (0, 255, 0), 1)
    ruta_debug = os.path.join(RUTA_SALIDA, f"6.1_contorno_{nombre_rasgo}.jpg")
    cv2.imwrite(ruta_debug, img_debug)
    print(f"  - Imagen de depuración guardada en: {ruta_debug}")

    # 4. Extracción de descriptores
    
    # --- 4.1 Compacidad ---
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    
    if area == 0 or perimetro == 0:
        print("  ADVERTENCIA: Área o perímetro del contorno es cero. Omitiendo.")
        continue
        
    compacidad = (perimetro ** 2) / area
    print(f"  - Compacidad: {compacidad:.4f}")

    # --- 4.2 Descriptores de Distancia Radial Normalizada ---
    # a) Calcular centroide
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        print("  ADVERTENCIA: Momento m00 es cero. No se puede calcular centroide.")
        continue
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # b) Calcular distancias radiales
    distancias_radiales = []
    for punto in cnt:
        # punto[0] contiene las coordenadas (x, y)
        dist = np.sqrt((punto[0][0] - cx)**2 + (punto[0][1] - cy)**2)
        distancias_radiales.append(dist)
    
    distancias_radiales = np.array(distancias_radiales)

    # c) Normalizar las distancias (dividiendo por la media)
    media_dist_radial = np.mean(distancias_radiales)
    if media_dist_radial == 0:
        print("  ADVERTENCIA: La distancia radial media es cero.")
        continue
        
    dist_radial_norm = distancias_radiales / media_dist_radial
    print("  - Distancia radial normalizada calculada.")

    # d) Calcular descriptores estadísticos
    drn_media = np.mean(dist_radial_norm) # Debería ser ~1.0
    drn_desv_est = np.std(dist_radial_norm)
    
    # Cruces por cero (cuántas veces cruza la media "1")
    signos = np.sign(dist_radial_norm - 1)
    # Convertimos a int para evitar problemas de tipo, luego comparamos vecinos
    cruces_por_cero = np.sum(np.abs(np.diff(signos))) / 2
    
    # Índice de área (Fórmula simplificada basada en radiolandia)
    # Nota: Esto es una aproximación del área bajo la curva de la firma
    indice_area = np.sum(dist_radial_norm) 

    # Índice de rugosidad (Suma de diferencias absolutas entre vecinos)
    indice_rugosidad = np.sum(np.abs(np.diff(dist_radial_norm)))

    print(f"  --- Descriptores Radiales ---")
    print(f"    - Media: {drn_media:.4f}")
    print(f"    - Desviación Estándar: {drn_desv_est:.4f}")
    print(f"    - Cruces por Cero: {int(cruces_por_cero)}")
    print(f"    - Índice de Rugosidad: {indice_rugosidad:.4f}")

    # Almacenar en la lista de resultados
    resultados.append({
        "Rasgo": nombre_rasgo,
        "Compacidad": round(compacidad, 4),
        "DRN_Desv_Est": round(drn_desv_est, 4),
        "DRN_Cruces_Cero": int(cruces_por_cero),
        "DRN_Rugosidad": round(indice_rugosidad, 4)
    })

# 5. Mostrar resultados finales en una tabla
print("\n" + "=" * 70)
print("                 RESULTADOS FINALES DE DESCRIPTORES")
print("=" * 70)
if resultados:
    df = pd.DataFrame(resultados)
    print(df.to_string(index=False))
    
    # Opcional: Guardar CSV
    ruta_csv = os.path.join(RUTA_SALIDA, "resultados_descriptores.csv")
    df.to_csv(ruta_csv, index=False)
    print(f"\nResultados guardados en: {ruta_csv}")
else:
    print("No se pudieron generar resultados.")
print("=" * 70)