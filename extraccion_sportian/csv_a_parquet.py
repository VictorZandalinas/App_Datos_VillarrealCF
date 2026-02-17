import pandas as pd
import os
import numpy as np
import glob

# ==========================================
# 1. FUNCIONES DE LÓGICA DE NEGOCIO (ETL)
# ==========================================

def clasificar_zona_simple(x, y, y_balon_start):
    """Calcula la zona táctica basándose en coordenadas normalizadas."""
    es_izq = y_balon_start > 50
    
    if x < 70: return "VIGILANCIA"
    if 75 <= x <= 100:
        if (es_izq and y >= 81) or (not es_izq and y <= 25): return "CORTO"
    
    if x > 94.2: # Area Pequeña
        if 45 <= y <= 55: return "AP CENTRO"
        return "AP 1ER PALO" if ((es_izq and y > 55) or (not es_izq and y < 45)) else "AP 2DO PALO"
    
    if 83 <= x <= 94.2: # Area Grande / Penalti
        if 45 <= y <= 55: return "PENALTI" if x < 88 else "CENTRO AREA"
        return "1ER PALO" if ((es_izq and y > 55) or (not es_izq and y < 45)) else "2DO PALO"
        
    if 70 <= x < 83: return "RECHACE"
    return "OTRO"

# ==========================================
# 2. FUNCIÓN DE ACTUALIZACIÓN INCREMENTAL
# ==========================================

def actualizar_parquets_especializados(df_maestro, id_col):
    """
    Compara el DF Maestro con los Parquets especializados existentes.
    Procesa SOLO los IDs nuevos y los añade (Append).
    """
    print("\n🏭 VERIFICANDO ACTUALIZACIONES EN PARQUETS ESPECIALIZADOS...")
    
    # Nombres de archivo
    file_off = "corners_ofensivo_enrich.parquet"
    file_def = "corners_defensivo_snapshot.parquet"
    
    # --- 1. CARGAR EXISTENTES Y DETECTAR FALTANTES ---
    ids_procesados = set()
    df_old_off = pd.DataFrame()
    df_old_def = pd.DataFrame()

    # Intentamos leer el Ofensivo para ver qué IDs ya tenemos
    if os.path.exists(file_off):
        try:
            df_old_off = pd.read_parquet(file_off)
            if id_col in df_old_off.columns:
                ids_procesados = set(df_old_off[id_col].unique())
            print(f"   📂 Archivo ofensivo existente: {len(df_old_off)} filas ({len(ids_procesados)} eventos).")
        except: pass
    
    # Identificar IDs nuevos en el maestro
    ids_maestro = set(df_maestro[id_col].unique())
    ids_nuevos = ids_maestro - ids_procesados
    
    if not ids_nuevos:
        print("   ✅ Todos los datos están al día. No se requiere procesamiento ETL.")
        return

    print(f"   ⚡ Detectados {len(ids_nuevos)} eventos nuevos. Procesando...")

    # --- 2. PROCESAR SOLO LO NUEVO ---
    # Filtramos el maestro para trabajar solo con lo nuevo (ahorra RAM y CPU)
    df_delta = df_maestro[df_maestro[id_col].isin(ids_nuevos)].copy()
    
    new_data_off = []
    new_data_def = []

    # Agrupar por evento
    for evento_id, df_ev in df_delta.groupby(id_col):
        # Datos comunes
        try:
            row_0 = df_ev.iloc[0]
            # Detectar lado (según Y balón al inicio)
            y_saque = df_ev[df_ev['Segundos_Desde_Saque'].between(-0.1, 0.5)]['Y_Balon'].mean()
            if pd.isna(y_saque): y_saque = df_ev.iloc[0]['Y_Balon']
            
            # Detectar tipo lanzamiento
            tipo_raw = str(row_0.get('Tipo_Lanzamiento', ''))
            if 'Out-swinger' in tipo_raw:
                tipo_lanz = 'Abierto'
            elif 'In-swinger' in tipo_raw:
                tipo_lanz = 'Cerrado'
            else:
                tipo_lanz = 'Neutro'
            
            # --- PROCESO OFENSIVO (Secuencia) ---
            # Recortamos tiempos inútiles (-0.5 a 6.0s)
            mask_off = df_ev['Segundos_Desde_Saque'].between(-0.5, 6.0)
            df_off_temp = df_ev[mask_off].copy()
            
            # Enriquecemos
            df_off_temp['Tipo_Lanzamiento_Calc'] = tipo_lanz
            df_off_temp['Zona_Precalc'] = df_off_temp.apply(
                lambda r: clasificar_zona_simple(r['X_Jugador'], r['Y_Jugador'], y_saque), axis=1
            )
            new_data_off.append(df_off_temp)
            
            # --- PROCESO DEFENSIVO (Snapshot) ---
            # Buscamos frame exacto en -3.0
            idx_def = (df_ev['Segundos_Desde_Saque'] - (-3.0)).abs().idxmin()
            t_def = df_ev.loc[idx_def, 'Segundos_Desde_Saque']
            
            if abs(t_def - (-3.0)) < 0.5: # Si existe el frame
                df_snap = df_ev[df_ev['Segundos_Desde_Saque'] == t_def].copy()
                
                # Filtrar: Quitamos Balón y Equipo Lanzador (quedan defensores)
                equipo_lanz = str(row_0.get('Equipo_Lanzador', '')).strip()
                # NOTA: Guardamos todo el snapshot por si acaso queremos ver atacantes también en el defensivo
                # Pero marcamos quién es quién
                
                df_snap['Zona_Precalc'] = df_snap.apply(
                    lambda r: clasificar_zona_simple(r['X_Jugador'], r['Y_Jugador'], y_saque), axis=1
                )
                new_data_def.append(df_snap)
        except Exception as e:
            continue

    # --- 3. CONCATENAR Y GUARDAR ---
    
    # Actualizar Ofensivo
    if new_data_off:
        df_new_off = pd.concat(new_data_off)
        if not df_old_off.empty:
            df_final_off = pd.concat([df_old_off, df_new_off], ignore_index=True)
        else:
            df_final_off = df_new_off
            
        df_final_off.to_parquet(file_off, index=False)
        print(f"   💾 Actualizado {file_off}: +{len(df_new_off)} filas (Total: {len(df_final_off)})")

    # Actualizar Defensivo (cargar existente si no se cargó antes)
    if new_data_def:
        if os.path.exists(file_def) and df_old_def.empty:
             try: df_old_def = pd.read_parquet(file_def)
             except: pass

        df_new_def = pd.concat(new_data_def)
        if not df_old_def.empty:
            df_final_def = pd.concat([df_old_def, df_new_def], ignore_index=True)
        else:
            df_final_def = df_new_def
            
        df_final_def.to_parquet(file_def, index=False)
        print(f"   💾 Actualizado {file_def}: +{len(df_new_def)} filas (Total: {len(df_final_def)})")


# ==========================================
# 3. PROCESO PRINCIPAL (CSV -> MAESTRO)
# ==========================================

def procesar_dataset(csv_source, parquet_dest, id_col_name):
    print(f"\n🔵 PROCESANDO: {csv_source}  -->  DESTINO: {parquet_dest}")

    df_new = None
    df_old = None
    
    # 1. CARGAR CSV NUEVO
    if os.path.exists(csv_source):
        try:
            df_new = pd.read_csv(csv_source)
            if 'Segundos_Relativos_Al_Saque' in df_new.columns:
                df_new.rename(columns={'Segundos_Relativos_Al_Saque': 'Segundos_Desde_Saque'}, inplace=True)
            
            if id_col_name not in df_new.columns:
                # Busqueda alternativa de ID
                possible_ids = [c for c in df_new.columns if 'ID_Evento' in c]
                if possible_ids: id_col_name = possible_ids[0]
                else: return # Sin ID no hacemos nada

            print(f"📄 CSV cargado: {len(df_new)} filas.")
        except Exception as e:
            print(f"❌ Error al leer CSV: {e}")
            return
    else: return

    # 2. CARGAR PARQUET MAESTRO EXISTENTE
    if os.path.exists(parquet_dest):
        try:
            df_old = pd.read_parquet(parquet_dest)
            print(f"📚 Maestro existente: {len(df_old)} filas.")
        except: pass

    # 3. COMBINAR DATOS (Incremental)
    df_combined = None
    hay_cambios = False
    
    if df_new is not None and df_old is not None:
        if id_col_name in df_old.columns:
            ids_old = set(df_old[id_col_name].unique())
            df_new_filtered = df_new[~df_new[id_col_name].isin(ids_old)]
            
            if not df_new_filtered.empty:
                print(f"   ✨ Añadiendo {len(df_new_filtered)} filas nuevas al maestro.")
                df_combined = pd.concat([df_old, df_new_filtered], ignore_index=True)
                hay_cambios = True
            else:
                print("   zzz Sin datos nuevos para el maestro.")
                df_combined = df_old
        else:
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            hay_cambios = True

    elif df_new is not None and df_old is None:
        print("✨ Creando archivo maestro nuevo.")
        df_combined = df_new
        hay_cambios = True
    
    elif df_new is None and df_old is not None:
        df_combined = df_old

    # 4. NORMALIZACIÓN (Solo si hubo cambios y tenemos las columnas)
    cols_coords = ['X_Jugador', 'Y_Jugador', 'X_Balon', 'Y_Balon']
    if hay_cambios and all(col in df_combined.columns for col in cols_coords + ['Segundos_Desde_Saque', id_col_name]):
        # Identificar eventos a invertir (Saque en X<50 en T=0)
        # Optimizacion: Solo chequear los nuevos si es posible, pero aqui chequeamos todo por seguridad rápida
        saques = df_combined[df_combined['Segundos_Desde_Saque'] == 0.0]
        ids_invertir = saques.loc[saques['X_Balon'] < 50, id_col_name].unique()
        
        if len(ids_invertir) > 0:
            mask = df_combined[id_col_name].isin(ids_invertir)
            # Solo invertimos si no fueron invertidos antes (asumimos que lo nuevo viene raw)
            # Para simplificar: aplicamos normalización a todo el bloque nuevo
            for col in cols_coords:
                df_combined.loc[mask, col] = 100 - df_combined.loc[mask, col]
            print(f"   🔄 Normalizados {len(ids_invertir)} eventos.")

    # 5. GUARDAR MAESTRO
    if hay_cambios:
        try:
            df_combined.to_parquet(parquet_dest, index=False)
            print(f"   💾 MAESTRO GUARDADO: {parquet_dest}")
        except Exception as e:
            print(f"❌ Error guardando maestro: {e}")
    
    # 6. BORRAR CSV
    if hay_cambios or (df_new is not None):
        try: os.remove(csv_source)
        except: pass

    # ============================================================
    # 7. LLAMADA CRÍTICA: ACTUALIZAR LOS PARQUETS ESPECIALIZADOS
    # ============================================================
    # Solo si estamos procesando Córners (no faltas)
    if "corners" in parquet_dest:
        actualizar_parquets_especializados(df_combined, id_col_name)


def main():
    print("🚀 INICIANDO SISTEMA ETL DE MANTENIMIENTO...")
    csvs = glob.glob("*.csv")

    # --- CASO 1: HAY ARCHIVOS CSV NUEVOS ---
    if csvs:
        for csv_file in csvs:
            nombre_lower = csv_file.lower()
            
            if "corners" in nombre_lower:
                # Esto procesa el CSV, actualiza el maestro y AL FINAL llama a actualizar_parquets_especializados
                procesar_dataset(csv_file, "corners_tracking.parquet", "ID_Evento_Corner")
            elif "faltas" in nombre_lower:
                procesar_dataset(csv_file, "faltas_tracking.parquet", "ID_Evento_Falta")
            else:
                print(f"⚠️ Omitido: {csv_file}")

    # --- CASO 2: NO HAY CSVs (REVISIÓN DE MANTENIMIENTO) ---
    else:
        print("⚠️ No hay archivos CSV nuevos.")
        
        # Comprobamos si existe el maestro de Corners para ver si le falta sincronizar algo
        maestro_corners = "corners_tracking.parquet"
        
        if os.path.exists(maestro_corners):
            print(f"🔎 Revisando integridad entre '{maestro_corners}' y los archivos especializados...")
            try:
                # 1. Cargamos el maestro actual
                df_maestro = pd.read_parquet(maestro_corners)
                
                # 2. Forzamos la actualización (la función detectará sola si faltan datos)
                actualizar_parquets_especializados(df_maestro, "ID_Evento_Corner")
                
            except Exception as e:
                print(f"❌ Error leyendo el maestro para revisión: {e}")
        else:
            print("❌ No existe el archivo maestro de corners. No hay nada que procesar.")

    print("\n🏁 PROCESO COMPLETADO")

if __name__ == "__main__":
    main()