import pandas as pd
import os

def modify_player_data(input_file, output_file=None):
    """
    Modifica los datos de jugadores específicos del Real Oviedo:
    - Cambia 'Corner taken' a 'No'
    - Cambia 'outcome' a 0 si estaba en 1
    """
    
    # Lista de jugadores a modificar
    players_to_modify = [
        "Alemão",
        "Jaime Seoane", 
        "Portillo",
        "Lucas Ahijado",
        "D. Paraschiv"
    ]
    
    team_name = "Real Oviedo"
    
    print(f"Leyendo archivo: {input_file}")
    
    try:
        # Leer el archivo parquet
        df = pd.read_parquet(input_file)
        print(f"Datos cargados exitosamente. Filas: {len(df)}")
        
        # Mostrar las columnas disponibles para verificar
        print(f"Columnas disponibles: {list(df.columns)}")
        
        # Verificar que las columnas necesarias existen
        required_columns = ['playerName', 'Team Name', 'Corner taken', 'outcome']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"ADVERTENCIA: Las siguientes columnas no se encontraron: {missing_columns}")
            return
        
        # Crear una copia para trabajar
        df_modified = df.copy()
        
        # Contar filas modificadas
        total_modified = 0
        
        print(f"\nBuscando jugadores del {team_name}...")
        
        for player in players_to_modify:
            # Crear máscara para el jugador específico del equipo específico
            mask = (df_modified['playerName'] == player) & (df_modified['Team Name'] == team_name)
            
            # Contar filas que coinciden
            rows_found = mask.sum()
            
            if rows_found > 0:
                print(f"  - {player}: {rows_found} registros encontrados")
                
                # Modificar 'Corner taken' a 'No'
                df_modified.loc[mask, 'Corner taken'] = 'No'
                
                # Modificar 'outcome' a 0 si estaba en 1
                outcome_mask = mask & (df_modified['outcome'] == 1)
                outcome_changed = outcome_mask.sum()
                df_modified.loc[outcome_mask, 'outcome'] = 0
                
                print(f"    * Corner taken cambiado a 'No' en {rows_found} registros")
                print(f"    * Outcome cambiado de 1 a 0 en {outcome_changed} registros")
                
                total_modified += rows_found
            else:
                print(f"  - {player}: No se encontraron registros")
        
        print(f"\nTotal de registros modificados: {total_modified}")
        
        # Definir nombre del archivo de salida
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_modified.parquet"
        
        # Guardar el archivo modificado
        df_modified.to_parquet(output_file, index=False)
        print(f"\nArchivo guardado como: {output_file}")
        
        # Mostrar resumen de cambios
        print("\n=== RESUMEN DE CAMBIOS ===")
        print(f"Archivo original: {input_file}")
        print(f"Archivo modificado: {output_file}")
        print(f"Jugadores procesados: {', '.join(players_to_modify)}")
        print(f"Equipo: {team_name}")
        print(f"Total de registros modificados: {total_modified}")
        
        return df_modified
        
    except FileNotFoundError:
        print(f"ERROR: No se pudo encontrar el archivo {input_file}")
        print("Asegúrate de que el archivo existe y la ruta es correcta.")
    except Exception as e:
        print(f"ERROR: {str(e)}")

def main():
    """
    Función principal - modifica aquí el nombre de tu archivo
    """
    
    # CAMBIA ESTA RUTA POR LA DE TU ARCHIVO
    input_file = "abp_events.parquet"  # Pon aquí el nombre de tu archivo
    
    # Opcional: especifica un nombre para el archivo de salida
    # Si no lo especificas, se creará automáticamente agregando "_modified"
    output_file = None  # o especifica como "archivo_modificado.parquet"
    
    # Ejecutar la modificación
    modified_df = modify_player_data(input_file, output_file)
    
    if modified_df is not None:
        print("\n¡Proceso completado exitosamente!")

if __name__ == "__main__":
    main()