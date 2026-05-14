#!/usr/bin/env python3
"""
Script de prueba para depurar el procesamiento de CSV de Sportian
Simula exactamente lo que pasa en process_sportian_csv_upload()
"""
import os
import sys
import gc
import logging
import base64
import shutil

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger('test_sportian')

# Añadir directorios al path
main_dir = os.path.dirname(__file__)
sportian_dir = os.path.join(main_dir, 'extraccion_sportian')
sys.path.insert(0, main_dir)
sys.path.insert(0, sportian_dir)

def test_full_upload_flow():
    """Prueba el flujo completo de subida de CSV"""
    logger.info("=" * 60)
    logger.info("🧪 INICIANDO TEST DE FLUJO COMPLETO SPORTIAN")
    logger.info("=" * 60)

    # Buscar el CSV en Downloads
    csv_source = os.path.expanduser("~/Downloads/Tracking_corners_la_liga (16).csv")

    if not os.path.exists(csv_source):
        logger.error(f"❌ CSV no encontrado: {csv_source}")
        return

    csv_size_mb = os.path.getsize(csv_source) / (1024 * 1024)
    logger.info(f"📄 CSV encontrado: {csv_source}")
    logger.info(f"   → Tamaño: {csv_size_mb:.2f} MB")

    # Copiar CSV a sportian (simulando subida web)
    temp_csv_path = os.path.join(sportian_dir, "Tracking_corners_la_liga (16).csv")
    logger.info(f"\n1️⃣ Copiando CSV a carpeta sportian...")
    shutil.copy2(csv_source, temp_csv_path)
    logger.info(f"   → Copiado a: {temp_csv_path}")

    try:
        # Simular el flujo de process_sportian_csv_upload
        logger.info(f"\n2️⃣ Determinando tipo y destino...")
        nombre_lower = "Tracking_corners_la_liga (16).csv".lower()

        if 'corner' in nombre_lower:
            parquet_dest = os.path.join(sportian_dir, 'corners_tracking.parquet')
            id_col = 'ID_Evento_Corner'
            tipo = 'corners'
        else:
            logger.error("❌ El archivo debe contener 'corners' o 'faltas'")
            return

        logger.info(f"   → Tipo: {tipo}")
        logger.info(f"   → ID: {id_col}")
        logger.info(f"   → Destino: {parquet_dest}")

        # Importar csv_a_parquet
        logger.info(f"\n3️⃣ Importando procesar_dataset...")
        from csv_a_parquet import procesar_dataset

        # Ejecutar procesamiento
        logger.info(f"\n4️⃣ EJECUTANDO procesar_dataset()...")
        logger.info(f"   → CSV: {temp_csv_path}")
        logger.info(f"   → Parquet: {parquet_dest}")

        gc.collect()
        procesar_dataset(temp_csv_path, parquet_dest, id_col)

        logger.info(f"\n✅ TEST COMPLETADO EXITOSAMENTE")

    except Exception as e:
        logger.error(f"❌ ERROR DURANTE PROCESAMIENTO: {e}", exc_info=True)
        import traceback
        traceback.print_exc()

    finally:
        # Limpieza
        if os.path.exists(temp_csv_path):
            logger.info(f"\n🗑️ Eliminando CSV temporal...")
            os.remove(temp_csv_path)

if __name__ == "__main__":
    test_full_upload_flow()
