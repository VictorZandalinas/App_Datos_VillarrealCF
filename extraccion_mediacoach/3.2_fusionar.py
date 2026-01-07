import pandas as pd
import os

# --- Configuración de Archivos ---
OUTPUT_BASE_PATH = 'data'
# Lista de los nombres base de los archivos que queremos procesar
ARCHIVOS_A_FUSIONAR = [
    'rendimiento_fisico',
    'rendimiento_5',
    'rendimiento_10',
    'rendimiento_15'
]

# --- Funciones de Ayuda ---

def leer_parquet(ruta_archivo):
    """
    Lee un archivo Parquet de forma segura.
    Devuelve un DataFrame vacío si no existe o está vacío.
    Devuelve None si está realmente corrupto.
    """
    if not os.path.exists(ruta_archivo) or os.path.getsize(ruta_archivo) == 0:
        # Si no existe o está vacío, no es un error, solo no hay datos.
        return pd.DataFrame()
    try:
        # Usamos el motor pyarrow, que es el estándar de la industria.
        return pd.read_parquet(ruta_archivo, engine='pyarrow')
    except Exception as e:
        print(f"❌ Error Crítico al leer el archivo '{os.path.basename(ruta_archivo)}': {e}")
        return None

def generar_clave_unica(df):
    """
    Genera una columna de clave única para un DataFrame de Pandas.
    """
    columnas_clave = [
        'Id Jugador', 'Partido', 'Equipo', 
        'archivo_origen', 'tipo_reporte', 'hoja'
    ]
    # Aseguramos que todas las columnas clave existan
    for col in columnas_clave:
        if col not in df.columns:
            df[col] = '' # Añade la columna vacía si no existe
            
    # Combina las columnas en una sola clave, manejando valores nulos (NaN)
    return df[columnas_clave].fillna('').astype(str).agg('|'.join, axis=1)


# --- Lógica Principal ---

def fusionar():
    print('🚀 Iniciando fusión con Python (mucho más robusto)...')

    # Bucle `for` para procesar cada nombre base de tu lista
    for base_name in ARCHIVOS_A_FUSIONAR:
        print(f"\n--- Procesando: {base_name} ---")

        # Definimos las rutas dinámicamente dentro del bucle
        ARCHIVO_PRINCIPAL = os.path.join(OUTPUT_BASE_PATH, f'{base_name}.parquet')
        ARCHIVO_PROVISIONAL = os.path.join(OUTPUT_BASE_PATH, f'{base_name}_provisional.parquet')

        # 1. Verificar si hay algo que fusionar
        if not os.path.exists(ARCHIVO_PROVISIONAL):
            print(f'✅ No se encontró "{os.path.basename(ARCHIVO_PROVISIONAL)}". No hay nada que hacer.')
            continue  # Pasa al siguiente elemento de la lista

        # 2. Leer ambos archivos de forma segura
        df_principal = leer_parquet(ARCHIVO_PRINCIPAL)
        df_provisional = leer_parquet(ARCHIVO_PROVISIONAL)

        # 3. Validar que los archivos se pudieron leer
        if df_principal is None or df_provisional is None:
            print(f"❌ Fusión cancelada para {base_name} debido a un error de lectura.")
            continue

        print(f"📖 Filas en principal: {len(df_principal)}. Filas en provisional: {len(df_provisional)}.")

        # 4. Si no hay datos nuevos en el provisional, limpiar y salir.
        if df_provisional.empty:
            print("✅ El archivo provisional está vacío. No hay filas nuevas que añadir.")
            os.remove(ARCHIVO_PROVISIONAL)
            print("🧹 Archivo provisional eliminado.")
            continue

        # 5. Filtrar duplicados de forma eficiente
        claves_existentes = set(generar_clave_unica(df_principal))
        df_provisional['clave_unica'] = generar_clave_unica(df_provisional)
        datos_realmente_nuevos = df_provisional[~df_provisional['clave_unica'].isin(claves_existentes)]
        datos_realmente_nuevos = datos_realmente_nuevos.drop(columns=['clave_unica'])

        if len(datos_realmente_nuevos) == 0:
            print("✅ Las filas del archivo provisional ya existían en el principal.")
            os.remove(ARCHIVO_PROVISIONAL)
            print("🧹 Archivo provisional eliminado.")
            continue

        # 6. Combinar los DataFrames
        print(f"✨ Se añadirán {len(datos_realmente_nuevos)} filas nuevas.")
        df_combinado = pd.concat([df_principal, datos_realmente_nuevos], ignore_index=True)
        print(f"📈 Total de filas después de la fusión: {len(df_combinado)}")

        # 7. Guardar el resultado final
        try:
            df_combinado.to_parquet(ARCHIVO_PRINCIPAL, index=False, engine='pyarrow')
            print(f"✅ Fusión completada. '{os.path.basename(ARCHIVO_PRINCIPAL)}' ha sido actualizado.")
            
            # 8. Limpiar el archivo provisional
            os.remove(ARCHIVO_PROVISIONAL)
            print("🧹 Archivo provisional eliminado.")
        except Exception as e:
            print(f"❌ Error Crítico al guardar el archivo fusionado para {base_name}: {e}")


# Esto hace que el script se ejecute cuando lo llamas desde la terminal
if __name__ == "__main__":
    fusionar()