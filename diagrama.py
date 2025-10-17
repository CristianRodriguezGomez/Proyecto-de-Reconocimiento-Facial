from diagrams import Diagram, Cluster, Edge
from diagrams.generic.blank import Blank
from diagrams.generic.blank import Blank as File  # reemplazo
from diagrams.generic.compute import Rack
from diagrams.generic.device import Tablet
from diagrams.generic.network import Switch
from diagrams.generic.os import Ubuntu
from diagrams.generic.place import Datacenter

with Diagram("Pipeline de Preprocesamiento Facial", show=False, direction="LR", graph_attr={"splines": "spline"}):
    start = Datacenter("Inicio")
    end = Datacenter("Fin")

    with Cluster("Fase 1: Adquisición y Detección"):
        acquisition = File("Obtención de Imágenes\n(5 fotos por integrante)")
        detection = Rack("1. Detección de 68 \n Puntos Clave\n(Landmarks Faciales)")
        acquisition >> detection

    with Cluster("Bucle para cada imagen del dataset"):
        process_loop = Switch("...")

        detection_check = Ubuntu("¿Se detectaron los \n puntos clave?")
        detection >> detection_check

        no_detection = Blank("Descartar o marcar para revisión")
        detection_check - Edge(label="No") >> no_detection
        no_detection >> process_loop

        with Cluster("Fase 2: Alineación Geométrica"):
            geometric_calc = Rack("2. Cálculo Geométrico")
            detection_check - Edge(label="Sí") >> geometric_calc
            
            center_eyes = Blank("Calcular centro\n de ojos")
            rotation_angle = Blank("Calcular ángulo\n de rotación (θ)")
            scale_factor = Blank("Calcular factor \n de escala")
            translation_vector = Blank("Calcular vector \n de traslación")
            geometric_calc >> center_eyes >> rotation_angle >> scale_factor >> translation_vector

        affine_transform = Rack("3. Aplicar Transformación Afín\n(cv2.warpAffine)")
        aligned_face = Tablet("Rostro Alineado \n y Normalizado")
        translation_vector >> affine_transform >> aligned_face

        with Cluster("Fase 3: Mejora de Calidad"):
            filters_application = Rack("4. Aplicar Filtros de Mejora")
            median_filter = Blank("Filtro Estadístico\n(Ej. Mediana)")
            gaussian_filter = Blank("Filtro Suavizante\n(Ej. Gaussiano)")
            high_boost = Blank("Filtro Realzante\n(Ej. High-Boost)")
            aligned_face >> filters_application >> median_filter >> gaussian_filter >> high_boost

        enhanced_face = Tablet("Rostro Final Mejorado")
        high_boost >> enhanced_face
        
        save_image = File("Guardar imagen procesada")
        enhanced_face >> save_image
        save_image >> process_loop

    final_dataset = File("Dataset Final de Rostros Preprocesados")

    start >> acquisition
    process_loop >> Edge(label="Fin del bucle") >> final_dataset
    final_dataset >> end

print("¡Diagrama generado exitosamente!")
