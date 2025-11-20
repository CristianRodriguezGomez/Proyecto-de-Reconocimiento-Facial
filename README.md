👁️ Detección y Normalización Facial
Proyecto de Detección de Puntos Clave Faciales (Landmarks)
Este proyecto implementa un script en Python para la detección precisa de rostros y la localización de 68 puntos clave faciales (landmarks) utilizando las librerías dlib y OpenCV.

El proceso es un paso fundamental en sistemas de visión por computadora, como el Reconocimiento Facial, ya que proporciona las coordenadas necesarias para la posterior alineación geométrica y normalización fotométrica del rostro.

🖼️ Ejemplo de Salida
Una vez ejecutado, el script genera una imagen con el bounding box del rostro y los 68 puntos clave superpuestos, como se muestra a continuación:

!

📜 Descripción Detallada
El objetivo principal del script es identificar y marcar puntos anatómicos específicos en un rostro (esquinas de los ojos, punta de la nariz, contorno de la boca y mandíbula). Esta información es la base para aplicaciones avanzadas como:

Alineación Facial Biometríca: Corrección de inclinación y escala de rostros.

Reconocimiento de expresiones faciales.

Seguimiento de rostros en video.

Realidad Aumentada (ej. filtros faciales).

📂 Estructura del Proyecto
El proyecto mantiene una estructura clara para la gestión de modelos y datos de entrada/salida:

Proyecto_Reconocimiento_Facial/
│
├── input_fotos/
│   └── rostro_prueba.jpg         # 📥 Directorio para las imágenes de entrada
├── modelos/
│   └── shape_predictor_68_face_landmarks.dat # 🧠 Modelo pre-entrenado de Dlib
├── env_facial/                   # 🐍 Entorno virtual de Python (Recomendado)
│
├── deteccion_puntos_clave.py     # 🚀 Script principal de detección
│
├── 1_deteccion_puntos_clave.jpg  # 🖼️ Imagen de salida con los puntos clave
├── temp_puntos_clave.npy         # 💾 Array de NumPy con las coordenadas (68x2)
└── README.md                     # 📖 Este archivo
⚙️ Requisitos y Configuración
Para ejecutar este proyecto, necesitas tener Python 3 y las siguientes dependencias instaladas.

1. Dependencias
Se recomienda encarecidamente trabajar dentro de un entorno virtual para aislar las dependencias:

Bash

# 1. Crear y activar un entorno virtual (env_facial)
python -m venv env_facial
# En Windows
.\env_facial\Scripts\activate
# En macOS/Linux
source env_facial/bin/activate

# 2. Instalar las dependencias requeridas
pip install opencv-python dlib numpy
2. Modelo Pre-entrenado
El script requiere el archivo binario del modelo de puntos clave de dlib: shape_predictor_68_face_landmarks.dat.

Descarga: Obtén el archivo shape_predictor_68_face_landmarks.dat del repositorio oficial de dlib (o la versión comprimida .bz2).

Ubicación: Coloca el archivo descomprimido directamente dentro de la carpeta modelos/.

🚀 Cómo Usar el Script
Sigue estos sencillos pasos para procesar una nueva imagen:

Coloca tu Imagen: Añade la imagen que deseas procesar dentro de la carpeta input_fotos/.

Actualiza la Configuración: Abre el script deteccion_puntos_clave.py y modifica la variable NOMBRE_IMAGEN_ENTRADA con el nombre de tu archivo:

Python

# --- Configuraciones ---
# ... otras variables
NOMBRE_IMAGEN_ENTRADA = os.path.join("input_fotos", "tu_imagen.jpg") # <-- Cambia esto
Ejecuta el script: Desde tu terminal (con el entorno virtual activado), ejecuta el comando:

Bash

python deteccion_puntos_clave.py
Archivos de Salida
El script generará los siguientes archivos en la carpeta raíz del proyecto:

Archivo	Descripción
1_deteccion_puntos_clave.jpg	Imagen de salida con el rostro enmarcado y los 68 puntos clave en color azul.
temp_puntos_clave.npy	Array binario de NumPy (68x2) con las coordenadas x,y de los landmarks. (Usado por scripts posteriores)
temp_imagen_original.jpg	Copia sin modificar de la imagen de entrada. (Usado por scripts posteriores)

## 🎨 Filtros de Mejoramiento de Imagen

El script `filtros_mejoramiento.py` aplica **3 tipos de filtros secuencialmente**:

### Filtros Aplicados (en orden)

1. **Filtro ESTADÍSTICO - Mediana (ksize=5)**
   - Elimina ruido tipo "sal y pimienta"
   - Preserva bordes sin desenfocar
   - Salida: `3.1_filtro_mediana.jpg`

2. **Filtro SUAVIZANTE - Gaussiano (ksize=5)**
   - Suaviza la imagen reduciendo ruido de alta frecuencia
   - Reduce detalles finos manteniendo estructura general
   - Salida: `3.2_filtro_gaussiano.jpg`

3. **Filtro REALZANTE - CLAHE**
   - Mejora el contraste local de forma adaptativa
   - Normalización fotométrica para reconocimiento facial
   - Salida: `3.3_filtro_clahe.jpg`

### Uso

```bash
# Ejecutar el script (procesa temp_rostro_alineado.jpg por defecto)
python filtros_mejoramiento.py
```

### Archivos Generados

El script genera **3 imágenes de salida**, una después de cada filtro:
- `3.1_filtro_mediana.jpg` - Imagen tras filtro estadístico
- `3.2_filtro_gaussiano.jpg` - Imagen tras filtro suavizante  
- `3.3_filtro_clahe.jpg` - Imagen final con los 3 filtros aplicados

---

## 🎭 Segmentación de Rasgos Faciales

El script `5_segmentacion_rasgos.py` se encarga de aislar y segmentar características faciales clave (ojos, nariz y boca) utilizando los puntos clave detectados previamente.

### Método Utilizado

1.  **Creación de Máscaras Poligonales**: Para cada rasgo, se genera una máscara poligonal conectando los puntos clave correspondientes (landmarks) para definir el área de interés.
2.  **Aislamiento del Rasgo**: Se utiliza la máscara para recortar el rasgo de la imagen del rostro alineado, dejando el resto de la imagen en negro.
3.  **Umbralización de Otsu**: Sobre el rasgo aislado, se aplica la umbralización de Otsu. Este método calcula automáticamente el umbral óptimo para binarizar la imagen, separando eficazmente el rasgo del fondo.

### Uso

```bash
# Ejecutar el script (procesa 2_rostro_alineado.jpg y temp_puntos_clave.npy)
python 5_segmentacion_rasgos.py
```

### Archivos Generados

El script genera dos imágenes por cada rasgo en la carpeta `output_fotos_segmentacion_rasgos/`:
- `5.1_mascara_[rasgo].jpg`: La máscara poligonal blanca sobre fondo negro.
- `5.2_segmentado_[rasgo].jpg`: El rasgo final segmentado (binarizado).

Exportar a Hojas de cálculo
