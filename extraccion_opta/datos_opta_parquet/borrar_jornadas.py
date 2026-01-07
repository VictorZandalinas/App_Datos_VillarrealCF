import pandas as pd
import os

def filtrar_archivos_parquet_por_semana_robusto():
    """
    Busca todos los archivos Parquet en el directorio actual,
    convierte la columna 'Week' a tipo numérico para asegurar la compatibilidad,
    elimina las filas donde la semana es 35 o 36, y sobrescribe los archivos.
    """
    directorio_actual = os.getcwd()
    print(f"Buscando archivos Parquet en: {directorio_actual}")

    for nombre_archivo in os.listdir(directorio_actual):
        if nombre_archivo.endswith(".parquet"):
            ruta_completa = os.path.join(directorio_actual, nombre_archivo)
            print(f"\nProcesando archivo: {nombre_archivo}")

            try:
                df = pd.read_parquet(ruta_completa)

                if 'Week' in df.columns:
                    filas_originales = len(df)
                    
                    # --- AQUÍ ESTÁ LA MAGIA ---
                    # Convierte la columna 'Week' a tipo numérico.
                    # Si algún valor no es un número, se convertirá en 'NaN' (Not a Number).
                    df['Week_temp'] = pd.to_numeric(df['Week'], errors='coerce')
                    
                    # Filtra usando la nueva columna numérica temporal
                    df_filtrado = df[~df['Week_temp'].isin([35, 36])]
                    
                    # Elimina la columna temporal antes de guardar
                    df_filtrado = df_filtrado.drop(columns=['Week_temp'])

                    filas_nuevas = len(df_filtrado)

                    if filas_nuevas < filas_originales:
                        df_filtrado.to_parquet(ruta_completa, index=False)
                        print(f"  - ¡Éxito! Se eliminaron {filas_originales - filas_nuevas} filas. Archivo actualizado.")
                    else:
                        print("  - No se encontraron filas con Semana 35 o 36 (después de la conversión a número). El archivo no ha cambiado.")
                else:
                    print(f"  - La columna 'Week' no se encontró en {nombre_archivo}. Se omite el archivo.")
                    print(f"  - Columnas disponibles: {list(df.columns)}")

            except Exception as e:
                print(f"  - Ocurrió un error al procesar el archivo {nombre_archivo}: {e}")

if __name__ == "__main__":
    filtrar_archivos_parquet_por_semana_robusto()
    print("\nProceso completado.")