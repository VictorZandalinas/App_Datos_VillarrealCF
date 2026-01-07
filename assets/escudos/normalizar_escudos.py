import os
from PIL import Image

# --- CONFIGURACIÓN ---
# Define la altura que quieres que tengan todas las imágenes
NUEVA_ALTURA = 300
# Nombre de la carpeta donde se guardarán los escudos redimensionados
CARPETA_SALIDA = "escudos_normalizados"
# ---------------------

# Obtiene la ruta de la carpeta donde se está ejecutando el script
ruta_actual = os.getcwd()
# Crea la ruta completa para la nueva carpeta de salida
ruta_salida = os.path.join(ruta_actual, CARPETA_SALIDA)

# Crea la carpeta de salida si no existe
if not os.path.exists(ruta_salida):
    os.makedirs(ruta_salida)
    print(f"Carpeta '{CARPETA_SALIDA}' creada.")

# Recorre todos los archivos en la carpeta actual
for nombre_archivo in os.listdir(ruta_actual):
    # Comprueba si el archivo es un .png
    if nombre_archivo.lower().endswith(".png"):
        print(f"Procesando: {nombre_archivo}...")
        
        # Construye la ruta completa del archivo de imagen
        ruta_imagen = os.path.join(ruta_actual, nombre_archivo)
        
        try:
            # Abre la imagen
            img = Image.open(ruta_imagen)
            
            # Calcula la nueva anchura manteniendo la proporción
            anchura, altura = img.size
            ratio = anchura / altura
            nueva_anchura = int(NUEVA_ALTURA * ratio)
            
            # Redimensiona la imagen con alta calidad
            img_redimensionada = img.resize((nueva_anchura, NUEVA_ALTURA), Image.Resampling.LANCZOS)
            
            # Construye la ruta de salida para la nueva imagen
            ruta_archivo_salida = os.path.join(ruta_salida, nombre_archivo)
            
            # Guarda la imagen redimensionada en la nueva carpeta
            img_redimensionada.save(ruta_archivo_salida)
            
            print(f" -> Guardado como '{nombre_archivo}' en '{CARPETA_SALIDA}' con tamaño {nueva_anchura}x{NUEVA_ALTURA}.")

        except Exception as e:
            print(f"No se pudo procesar el archivo {nombre_archivo}. Error: {e}")

print("\n¡Proceso completado!")