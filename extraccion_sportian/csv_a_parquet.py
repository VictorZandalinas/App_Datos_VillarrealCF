import pandas as pd
import os
import numpy as np
import glob

# --- FUNCIÓN REUTILIZABLE PARA PROCESAR ---
def procesar_dataset(csv_source, parquet_dest, id_col_name):
    """
    Lee un CSV, lo combina con el Parquet destino, normaliza y guarda.
    """
    print(f"\n🔵 PROCESANDO: {csv_source}  -->  DESTINO: {parquet_dest}")

    df_new = None
    df_old = None
    rows_iniciales = 0
    filas_nuevas_reales = 0
    
    # Variables de control
    archivo_procesado = False
    guardado_ok = False
    cantidad_invertida = 0 
    duplicados_eliminados = 0 

    # 1. CARGAR CSV NUEVO
    if os.path.exists(csv_source):
        try:
            df_new = pd.read_csv(csv_source)
            
            # Unificar nombre de columna de tiempo (aplica igual para corners y faltas)
            if 'Segundos_Relativos_Al_Saque' in df_new.columns:
                df_new.rename(columns={'Segundos_Relativos_Al_Saque': 'Segundos_Desde_Saque'}, inplace=True)
            
            # Validar columna clave dinámica
            if id_col_name not in df_new.columns:
                # Intenta buscar si existe la columna genérica si falla la específica
                print(f"⚠️  Advertencia: No se encontró '{id_col_name}'. Buscando alternativas...")
                possible_ids = [c for c in df_new.columns if 'ID_Evento' in c]
                if possible_ids:
                    id_col_name = possible_ids[0]
                    print(f"👌 Se usará la columna '{id_col_name}' como ID.")
                else:
                    print(f"❌ ERROR: El CSV no tiene una columna de ID válida.")
                    return # Salimos de la función si no hay ID

            print(f"📄 CSV cargado: {len(df_new)} filas.")
            archivo_procesado = True 

        except Exception as e:
            print(f"❌ Error al leer CSV: {e}")
            return
    else:
        print(f"⚠️ No se encontró '{csv_source}'.")
        return

    # 2. CARGAR PARQUET EXISTENTE
    if os.path.exists(parquet_dest):
        try:
            df_old = pd.read_parquet(parquet_dest)
            rows_iniciales = len(df_old)
            print(f"📚 Parquet existente cargado: {rows_iniciales} filas.")
        except Exception as e:
            print(f"❌ Error leyendo Parquet: {e}")

    # 3. COMBINAR DATOS
    df_combined = None
    
    if df_old is None and df_new is None:
        print("❌ No hay datos.")
        return

    if df_new is not None and df_old is not None:
        # Usamos id_col_name dinámico
        if id_col_name in df_old.columns:
            ids_old = set(df_old[id_col_name].unique())
            df_new_filtered = df_new[~df_new[id_col_name].isin(ids_old)]
            filas_nuevas_reales = len(df_new_filtered)
            
            if filas_nuevas_reales > 0:
                print(f"   ✨ Añadiendo {filas_nuevas_reales} filas nuevas.")
                df_combined = pd.concat([df_old, df_new_filtered], ignore_index=True)
            else:
                print("   zzz No hay datos nuevos (IDs ya existentes).")
                df_combined = df_old
        else:
            # Si el parquet viejo no tiene la columna ID (raro), concatenamos todo
            print("⚠️ El parquet antiguo no tiene la misma columna ID. Concatenando forzosamente.")
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            filas_nuevas_reales = len(df_new)

    elif df_new is not None and df_old is None:
        print("✨ Creando archivo nuevo.")
        df_combined = df_new
        filas_nuevas_reales = len(df_new)
    
    elif df_new is None and df_old is not None:
        df_combined = df_old

    # 4. NORMALIZACIÓN DE COORDENADAS
    cols_coords = ['X_Jugador', 'Y_Jugador', 'X_Balon', 'Y_Balon']
    col_tiempo = 'Segundos_Desde_Saque'
    
    # Verificamos que existan las columnas necesarias
    if all(col in df_combined.columns for col in cols_coords + [col_tiempo, id_col_name]):
        # A. Detectar saques desde la izquierda (X < 50) en el segundo 0
        saques = df_combined[df_combined[col_tiempo] == 0.0]
        ids_a_invertir = saques.loc[saques['X_Balon'] < 50, id_col_name].unique()
        
        if len(ids_a_invertir) > 0:
            print(f"   🔄 Normalizando {len(ids_a_invertir)} eventos sacados desde la izquierda...")
            
            mask = df_combined[id_col_name].isin(ids_a_invertir)
            
            # Aplicar 100 - X a las columnas
            for col in cols_coords:
                df_combined.loc[mask, col] = 100 - df_combined.loc[mask, col]
            
            cantidad_invertida = mask.sum()
            print(f"   ✅ Se han invertido coordenadas de {cantidad_invertida} filas.")
        else:
            print("   ✅ Todos los eventos ya están orientados correctamente.")
    else:
        print("⚠️ Faltan columnas de coordenadas o tiempo. No se pudo normalizar.")

    # 5. GUARDAR PARQUET
    hay_cambios = filas_nuevas_reales > 0 or cantidad_invertida > 0 or (df_new is not None and df_old is None)

    if hay_cambios:
        try:
            filas_antes = len(df_combined)
            df_combined = df_combined.drop_duplicates()
            filas_despues = len(df_combined)
            duplicados_eliminados = filas_antes - filas_despues
            
            if duplicados_eliminados > 0:
                print(f"   🧹 Eliminados {duplicados_eliminados} duplicados.")
            
            df_combined.to_parquet(parquet_dest, index=False)
            print(f"   💾 GUARDADO EXITOSO: {parquet_dest}")
            guardado_ok = True
        except Exception as e:
            print(f"❌ Error guardando: {e}")
            guardado_ok = False
    else:
        print("   ✅ Archivo al día. No se requiere guardar.")
        guardado_ok = True

    # 6. BORRAR CSV
    if archivo_procesado and guardado_ok:
        try:
            os.remove(csv_source)
            print(f"   🗑️  CSV eliminado: '{csv_source}'")
        except Exception as e:
            print(f"⚠️ No se pudo eliminar el CSV: {e}")


# --- MAIN ---
def main():
    print("🚀 INICIANDO PROCESO DE MANTENIMIENTO (CORNERS Y FALTAS)...")
    
    # Buscar todos los CSVs en la carpeta
    csvs = glob.glob("*.csv")

    if not csvs:
        print("⚠️ No se encontraron archivos CSV.")
        return

    for csv_file in csvs:
        nombre_lower = csv_file.lower()
        
        # LOGICA DE CLASIFICACIÓN
        if "corners" in nombre_lower:
            # Asumimos que el ID es ID_Evento_Corner, si cambia el script lo detectará auto
            procesar_dataset(csv_file, "corners_tracking.parquet", "ID_Evento_Corner")
            
        elif "faltas" in nombre_lower:
            # Asumimos que el ID podría ser ID_Evento_Falta
            procesar_dataset(csv_file, "faltas_tracking.parquet", "ID_Evento_Falta")
            
        else:
            print(f"\n⚠️ El archivo '{csv_file}' no contiene 'corners' ni 'faltas' en el nombre. Se omite.")

    print("\n" + "="*60)
    print("🏁 PROCESO COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()