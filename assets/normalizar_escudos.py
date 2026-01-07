from PIL import Image
import os

def normalizar_escudos(carpeta_origen="escudos", carpeta_destino="escudos_normalizados", altura_objetivo=100):
    """
    Normaliza el tamaño de todos los escudos PNG para que tengan la misma altura,
    manteniendo la proporción original de cada escudo.
    
    Args:
        carpeta_origen: Carpeta donde están los escudos originales
        carpeta_destino: Carpeta donde se guardarán los escudos normalizados
        altura_objetivo: Altura en píxeles que tendrán todos los escudos
    """
    
    # Crear carpeta de destino si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    
    # Obtener todos los archivos PNG de la carpeta
    archivos_png = [f for f in os.listdir(carpeta_origen) if f.lower().endswith('.png')]
    
    if not archivos_png:
        print("No se encontraron archivos PNG en la carpeta 'escudos'")
        return
    
    print(f"Procesando {len(archivos_png)} escudos...")
    print(f"Altura objetivo: {altura_objetivo} píxeles")
    
    for archivo in archivos_png:
        try:
            # Abrir imagen original
            ruta_original = os.path.join(carpeta_origen, archivo)
            imagen = Image.open(ruta_original)
            
            # Obtener dimensiones originales
            ancho_original, alto_original = imagen.size
            
            # Calcular nuevo ancho manteniendo la proporción
            factor_escala = altura_objetivo / alto_original
            nuevo_ancho = int(ancho_original * factor_escala)
            
            # Redimensionar imagen
            imagen_redimensionada = imagen.resize((nuevo_ancho, altura_objetivo), Image.Resampling.LANCZOS)
            
            # Guardar imagen procesada
            ruta_destino = os.path.join(carpeta_destino, archivo)
            imagen_redimensionada.save(ruta_destino, 'PNG')
            
            print(f"✓ {archivo}: {ancho_original}x{alto_original} → {nuevo_ancho}x{altura_objetivo}")
            
        except Exception as e:
            print(f"✗ Error procesando {archivo}: {str(e)}")
    
    print(f"\n¡Proceso completado! Los escudos normalizados están en '{carpeta_destino}'")

def encontrar_altura_minima(carpeta_origen="escudos"):
    """
    Encuentra la altura mínima entre todos los escudos para usarla como referencia.
    """
    archivos_png = [f for f in os.listdir(carpeta_origen) if f.lower().endswith('.png')]
    
    if not archivos_png:
        return None
    
    altura_minima = float('inf')
    archivo_menor = ""
    
    for archivo in archivos_png:
        try:
            ruta = os.path.join(carpeta_origen, archivo)
            imagen = Image.open(ruta)
            _, altura = imagen.size
            
            if altura < altura_minima:
                altura_minima = altura
                archivo_menor = archivo
                
        except Exception as e:
            print(f"Error leyendo {archivo}: {str(e)}")
    
    print(f"Altura mínima encontrada: {altura_minima} píxeles (archivo: {archivo_menor})")
    return altura_minima

if __name__ == "__main__":
    # Verificar que existe la carpeta
    if not os.path.exists("escudos"):
        print("La carpeta 'escudos' no existe. Créala y coloca los archivos PNG ahí.")
    else:
        # Opción 1: Usar altura fija (elige la que prefieras)
        normalizar_escudos(altura_objetivo=200)  # Cambia 100 por la altura que prefieras
        
        # Opción 2: Usar la altura del escudo más pequeño (descomenta la siguiente línea)
        # altura_min = encontrar_altura_minima()
        # if altura_min:
        #     normalizar_escudos(altura_objetivo=int(altura_min))