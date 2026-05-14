#!/usr/bin/env python3
"""
Script de prueba para depurar el procesamiento de CSV de Sportian
"""
import os
import sys
import gc
import logging
import pandas as pd
import pyarrow.parquet as pq

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger('test_sportian')

# Añadir extraccion_sportian al path
sportian_dir = os.path.join(os.path.dirname(__file__), 'extraccion_sportian')
sys.path.insert(0, sportian_dir)

def test_csv_processing():
    """Prueba el procesamiento de un CSV de Sportian"""
    logger.info("=" * 60)
    logger.info("🧪 INICIANDO TEST DE PROCESAMIENTO CSV SPORTIAN")
    logger.info("=" * 60)

    # Buscar CSVs existentes
    csv_files = [f for f in os.listdir(sportian_dir) if f.endswith('.csv')]

    if not csv_files:
        logger.error("❌ No hay archivos CSV en extraccion_sportian/")
        logger.info("💡 Copia un archivo de test con 'corners' o 'faltas' en el nombre")
        return

    logger.info(f"📂 CSVs encontrados: {csv_files}")

    for csv_file in csv_files:
        csv_path = os.path.join(sportian_dir, csv_file)
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 PROBANDO: {csv_file}")
        logger.info(f"{'='*60}")

        try:
            # 1. Leer CSV
            logger.info("1️⃣ Leyendo CSV...")
            df = pd.read_csv(csv_path)
            logger.info(f"   → Filas: {len(df)}, Columnas: {len(df.columns)}")
            logger.info(f"   → Columnas: {list(df.columns)[:10]}...")

            # 2. Verificar columna ID
            id_col = 'ID_Evento_Corner' if 'corner' in csv_file.lower() else 'ID_Evento_Falta'
            if id_col not in df.columns:
                possible_ids = [c for c in df.columns if 'ID_Evento' in c]
                if possible_ids:
                    id_col = possible_ids[0]
                    logger.info(f"   → ID alternativo detectado: {id_col}")
                else:
                    logger.error(f"   ❌ No se encontró columna ID")
                    continue

            logger.info(f"2️⃣ Columna ID: {id_col}")
            logger.info(f"   → IDs únicos: {df[id_col].nunique()}")

            # 3. Determinar destino parquet
            parquet_dest = os.path.join(
                sportian_dir,
                'corners_tracking.parquet' if 'corner' in csv_file.lower() else 'faltas_tracking.parquet'
            )
            logger.info(f"3️⃃ Destino parquet: {parquet_dest}")

            # 4. Verificar parquet existente
            if os.path.exists(parquet_dest):
                logger.info("4️⃣ Leyendo parquet existente...")
                table = pq.read_table(parquet_dest, columns=[id_col])
                ids_old = set(table.column(id_col).to_pylist())
                logger.info(f"   → {len(ids_old)} IDs existentes")

                # Filtrar nuevos
                new_ids = set(df[id_col].unique()) - ids_old
                logger.info(f"   → {len(new_ids)} IDs nuevos")
            else:
                logger.info("4️⃣ No existe parquet existente (primera vez)")
                new_ids = set(df[id_col].unique())

            # 5. Probar import de csv_a_parquet
            logger.info("5️⃣ Probando import de csv_a_parquet.procesar_dataset...")
            from csv_a_parquet import procesar_dataset

            # 6. Simular procesamiento (sin guardar)
            logger.info("6️⃣ Simulando procesamiento...")
            gc.collect()

            # Probar la función de normalización
            cols_coords = ['X_Jugador', 'Y_Jugador', 'X_Balon', 'Y_Balon']
            if all(col in df.columns for col in cols_coords):
                logger.info("   → Columnas de coordenadas OK")

                # Verificar segundos desde saque
                if 'Segundos_Relativos_Al_Saque' in df.columns:
                    logger.info("   → Detectada columna Segundos_Relativos_Al_Saque")
                elif 'Segundos_Desde_Saque' in df.columns:
                    logger.info("   → Detectada columna Segundos_Desde_Saque")
                else:
                    logger.warning("   ⚠️ No se encontró columna de tiempo")
            else:
                missing = [c for c in cols_coords if c not in df.columns]
                logger.warning(f"   ⚠️ Faltan columnas: {missing}")

            logger.info(f"\n✅ TEST COMPLETADO PARA: {csv_file}")

        except Exception as e:
            logger.error(f"❌ ERROR en {csv_file}: {e}", exc_info=True)
            import traceback
            traceback.print_exc()

        gc.collect()

if __name__ == "__main__":
    test_csv_processing()
