#!/usr/bin/env python3
"""
Script de prueba que simula EXACTAMENTE lo que pasa cuando Dash envía el contenido
"""
import os
import sys
import gc
import logging
import base64

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger('test_base64')

# Añadir directorios al path
main_dir = os.path.dirname(__file__)
sportian_dir = os.path.join(main_dir, 'extraccion_sportian')
sys.path.insert(0, main_dir)
sys.path.insert(0, sportian_dir)

def test_base64_decoding():
    """Prueba la decodificación base64 como lo hace Dash"""
    logger.info("=" * 60)
    logger.info("🧪 INICIANDO TEST DE DECODIFICACIÓN BASE64")
    logger.info("=" * 60)

    # Leer el archivo CSV original
    csv_source = os.path.expanduser("~/Downloads/Tracking_corners_la_liga (16).csv")

    if not os.path.exists(csv_source):
        logger.error(f"❌ CSV no encontrado: {csv_source}")
        return

    csv_size_mb = os.path.getsize(csv_source) / (1024 * 1024)
    logger.info(f"📄 CSV encontrado: {csv_source}")
    logger.info(f"   → Tamaño: {csv_size_mb:.2f} MB")

    try:
        # 1. Leer archivo como bytes (simulando lo que el navegador hace)
        logger.info(f"\n1️⃣ Leyendo archivo como bytes...")
        with open(csv_source, 'rb') as f:
            file_bytes = f.read()
        logger.info(f"   → Bytes leídos: {len(file_bytes)}")

        # 2. Codificar a base64 (simulando lo que Dash.Upload hace)
        logger.info(f"\n2️⃣ Codificando a base64...")
        encoded = base64.b64encode(file_bytes).decode('utf-8')
        logger.info(f"   → String base64: {len(encoded)} caracteres")

        # Liberar file_bytes
        del file_bytes
        gc.collect()

        # 3. Simular el formato que envía Dash: "data:text/csv;base64,{encoded}"
        content_type = "data:text/csv;base64"
        contents = f"{content_type},{encoded}"
        logger.info(f"   → Contents completo: {len(contents)} caracteres")

        # Liberar encoded
        del encoded
        gc.collect()

        # 4. Ahora simular process_sportian_csv_upload()
        logger.info(f"\n3️⃣ Simulando process_sportian_csv_upload()...")

        # Esto es LO MISMO que hace actualizar_datos.py línea 3543
        try:
            content_type, content_string = contents.split(',')
            logger.info(f"   → Split OK: content_type={content_type[:20]}...")
        except ValueError:
            content_string = contents
            logger.info(f"   → Split fallido, usando contents completo")

        # Liberar contents
        del contents
        gc.collect()

        # 5. Decodificar base64 (línea 3552)
        logger.info(f"\n4️⃣ Decodificando base64...")
        decoded = base64.b64decode(content_string)
        logger.info(f"   → Bytes decodificados: {len(decoded)}")

        # Liberar content_string
        del content_string
        gc.collect()

        # 6. Verificar que es un CSV válido
        logger.info(f"\n5️⃣ Verificando CSV decodificado...")
        decoded_size_mb = len(decoded) / (1024 * 1024)
        logger.info(f"   → Tamaño: {decoded_size_mb:.2f} MB")

        # 7. Guardar para verificar
        temp_path = os.path.join(sportian_dir, "test_decoded.csv")
        with open(temp_path, 'wb') as f:
            f.write(decoded)
        logger.info(f"   → Guardado en: {temp_path}")

        del decoded
        gc.collect()

        # 8. Ahora probar leerlo con pandas
        logger.info(f"\n6️⃣ Leyendo CSV decodificado con pandas...")
        import pandas as pd
        df = pd.read_csv(temp_path)
        logger.info(f"   → Filas: {len(df)}, Columnas: {len(df.columns)}")
        logger.info(f"   → Columnas: {list(df.columns)[:10]}...")

        # 9. Limpieza
        os.remove(temp_path)
        logger.info(f"\n✅ TEST BASE64 COMPLETADO EXITOSAMENTE")

    except Exception as e:
        logger.error(f"❌ ERROR: {e}", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_base64_decoding()
