#!/usr/bin/env python3
"""
Test diagnóstico del sistema de informes tácticos
"""
import sys
import os

print("="*70)
print("DIAGNÓSTICO DEL SISTEMA DE INFORMES TÁCTICOS")
print("="*70)

# 1. Verificar imports críticos
print("\n1. Verificando imports críticos...")
try:
    import pandas as pd
    print("   ✅ pandas")
except ImportError as e:
    print(f"   ❌ pandas: {e}")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("   ✅ matplotlib")
except ImportError as e:
    print(f"   ❌ matplotlib: {e}")
    sys.exit(1)

try:
    from PyPDF2 import PdfReader, PdfWriter
    print("   ✅ PyPDF2")
except ImportError as e:
    print(f"   ❌ PyPDF2: {e}")
    sys.exit(1)

try:
    import psutil
    print("   ✅ psutil")
except ImportError:
    print("   ⚠️ psutil (opcional - monitoreo limitado)")

# 2. Verificar módulos auxiliares
print("\n2. Verificando módulos auxiliares...")
try:
    from memory_utils import clean_memory_agresivo, get_memory_info, check_memory_threshold
    print("   ✅ memory_utils")
except ImportError as e:
    print(f"   ❌ memory_utils: {e}")
    sys.exit(1)

try:
    from pdf_fusion import fusionar_pdfs_incremental
    print("   ✅ pdf_fusion")
except ImportError as e:
    print(f"   ❌ pdf_fusion: {e}")
    sys.exit(1)

try:
    from memory_monitor import MemoryMonitor
    print("   ✅ memory_monitor")
except ImportError as e:
    print(f"   ❌ memory_monitor: {e}")
    sys.exit(1)

try:
    from informe_wrapper_chunked import InformeGeneratorChunked
    print("   ✅ informe_wrapper_chunked")
except ImportError as e:
    print(f"   ❌ informe_wrapper_chunked: {e}")
    sys.exit(1)

# 3. Verificar archivos de datos críticos
print("\n3. Verificando archivos de datos críticos...")
data_files = [
    'extraccion_opta/datos_opta_parquet/passes.parquet',
    'extraccion_opta/datos_opta_parquet/standings.parquet',
    'extraccion_mediacoach/data/estadisticas_equipo.parquet',
]

for data_file in data_files:
    if os.path.exists(data_file):
        size_mb = os.path.getsize(data_file) / 1024 / 1024
        print(f"   ✅ {data_file} ({size_mb:.1f} MB)")
    else:
        print(f"   ⚠️ {data_file} (no encontrado)")

# 4. Verificar assets
print("\n4. Verificando assets...")
assets = [
    'assets/portada_tactico_informe.png',
    'assets/escudos/',
]

for asset in assets:
    if os.path.exists(asset):
        if os.path.isdir(asset):
            count = len([f for f in os.listdir(asset) if f.endswith('.png')])
            print(f"   ✅ {asset} ({count} archivos)")
        else:
            size_kb = os.path.getsize(asset) / 1024
            print(f"   ✅ {asset} ({size_kb:.1f} KB)")
    else:
        print(f"   ❌ {asset} (no encontrado)")

# 5. Test del wrapper chunked con TACTICO
print("\n5. Testeando InformeGeneratorChunked con tipo TACTICO...")
try:
    wrapper = InformeGeneratorChunked(tipo_informe='TACTICO', chunk_size=4)
    print(f"   ✅ Wrapper creado")
    print(f"   📊 Scripts encontrados: {len(wrapper.scripts)}")
    print(f"   📊 Chunks: {len(wrapper.chunks)}")

    if len(wrapper.scripts) == 0:
        print("   ❌ ERROR: No se encontraron scripts tactic*.py")
    else:
        print(f"   📋 Primeros scripts:")
        for i, script in enumerate(wrapper.scripts[:5], 1):
            print(f"      {i}. {script}")
        if len(wrapper.scripts) > 5:
            print(f"      ... y {len(wrapper.scripts) - 5} más")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# 6. Verificar variables de entorno
print("\n6. Variables de entorno relevantes...")
env_vars = ['INFORME_USE_CHUNKS', 'TACTIC_J_INI', 'TACTIC_J_FIN']
for var in env_vars:
    val = os.environ.get(var, '(no definida)')
    print(f"   {var}: {val}")

# 7. Información de memoria
print("\n7. Información de memoria...")
mem_info = get_memory_info()
if mem_info['available']:
    print(f"   RSS: {mem_info['rss_mb']:.0f} MB")
    print(f"   Porcentaje: {mem_info['percent']:.1f}%")
else:
    print("   ⚠️ Información no disponible (psutil no instalado)")

print("\n" + "="*70)
print("DIAGNÓSTICO COMPLETADO")
print("="*70)
