#!/usr/bin/env python3
import os
import sys
import re
import shutil
import gc
import textwrap
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg') # CRÍTICO: Backend sin interfaz gráfica
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader, PdfWriter

# --- CONFIGURACIÓN DE MEMORIA ---
def clean_memory():
    """Fuerza el cierre de figuras y limpieza de RAM."""
    plt.close('all')
    gc.collect()

# --- PARCHE DE PANDAS (FILTRADO EN LECTURA) ---
# Esto evita cargar gigas de datos innecesarios en RAM
_original_read_parquet = pd.read_parquet

def patched_read_parquet(*args, **kwargs):
    df = _original_read_parquet(*args, **kwargs)
    
    # Obtener límites de argumentos globales (definidos en main)
    try:
        j_ini = int(os.environ.get('ABP_J_INI', 0))
        j_fin = int(os.environ.get('ABP_J_FIN', 99))
    except:
        return df

    # Filtrado automático si existen columnas de jornada
    cols_lower = [c.lower() for c in df.columns]
    
    # Lógica de detección de columnas
    col_jornada = None
    for c in df.columns:
        if any(x in c.lower() for x in ['jornada', 'week', 'match', 'round']):
            col_jornada = c
            break
            
    if col_jornada:
        try:
            # Normalizar a numérico
            s = df[col_jornada].astype(str).str.lower().str.replace('j', '').str.strip()
            df['__temp_j'] = pd.to_numeric(s, errors='coerce')
            
            # Filtrar
            if j_ini > 0:
                df = df[(df['__temp_j'] >= j_ini) & (df['__temp_j'] <= j_fin)]
            
            # Limpiar columna temporal
            df.drop(columns=['__temp_j'], inplace=True)
        except Exception:
            pass
            
    return df

# Aplicar el parche
pd.read_parquet = patched_read_parquet

# --- FUNCIONES AUXILIARES ---
def natural_sort_key(s):
    filename = os.path.basename(s)
    parts = re.split(r'(\d+\.?\d*)', filename)
    return [float(p) if re.match(r'^\d+\.?\d*$', p) else p.lower() for p in parts if p]

def ejecutar_script_aislado(script_path, inputs_simulados):
    """
    Ejecuta el script dentro de una función para que al terminar,
    todas sus variables locales (DataFrames gigantes) se eliminen.
    """
    print(f"   ▶️ Ejecutando {script_path} en entorno aislado...")
    
    # Simulamos inputs para input()
    original_stdin = sys.stdin
    sys.stdin = io.StringIO(inputs_simulados)
    
    try:
        # Leemos el código
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Diccionario local para este script (Scope aislado)
        local_scope = {}
        
        # Ejecutamos
        exec(code, globals(), local_scope)
        
    except Exception as e:
        print(f"   ❌ Error en {script_path}: {e}")
    finally:
        sys.stdin = original_stdin
        # Forzar limpieza de variables grandes creadas en el script
        del local_scope
        clean_memory()

def main():
    import io
    
    # 1. Argumentos (Recibidos directamente desde app.py o consola)
    if len(sys.argv) > 3:
        equipo = sys.argv[1]
        j_ini = sys.argv[2]
        j_fin = sys.argv[3]
        
        # Configurar variables de entorno para el parche de pandas
        os.environ['ABP_J_INI'] = str(j_ini)
        os.environ['ABP_J_FIN'] = str(j_fin)
        
        # Mapeo simple para inputs de los scripts hijos
        # (Ajusta esto según tus índices de Opta si es necesario)
        inputs = f"1\n{j_fin}\n" 
        if 'mediacoach' in sys.argv[0]:
             inputs = f"1\n{j_fin}\n{j_fin}\n"
    else:
        print("Uso: python abp_informe_todo.py 'Equipo' J_INI J_FIN")
        sys.exit(1)

    print(f"🚀 Iniciando Generador Optimizado para {equipo} (J{j_ini}-J{j_fin})")

    # Limpieza inicial
    temp_dir = "reportes_temporales"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Escanear scripts
    scripts = sorted([f for f in os.listdir('.') if re.match(r'^abp[0-9].*\.py$', f)], key=natural_sort_key)
    
    pdfs_generados = []

    # Bucle principal
    for i, script in enumerate(scripts, 1):
        clean_memory() # Limpiar antes de empezar
        
        pdfs_antes = set(f for f in os.listdir('.') if f.endswith(".pdf"))
        
        # EJECUCIÓN AISLADA (CLAVE PARA AHORRAR RAM)
        ejecutar_script_aislado(script, inputs)
        
        pdfs_despues = set(f for f in os.listdir('.') if f.endswith(".pdf"))
        nuevos = pdfs_despues - pdfs_antes
        
        for pdf in nuevos:
            destino = os.path.join(temp_dir, f"{i:03d}_{pdf}")
            shutil.move(pdf, destino)
            pdfs_generados.append(destino)
            print(f"   ✅ Generado: {pdf}")

    # Unir PDFs (Usando PdfWriter que es más eficiente en memoria que Merger)
    if pdfs_generados:
        print("🔄 Uniendo PDF final...")
        writer = PdfWriter()
        pdfs_generados.sort()
        
        for p_path in pdfs_generados:
            f = open(p_path, 'rb')
            reader = PdfReader(f)
            for page in reader.pages:
                writer.add_page(page)
            # Nota: No cerramos f inmediatamente para mantener el stream, 
            # pero Python lo cerrará al terminar el script.
        
        nombre_final = f"Informe_ABP_{equipo.replace(' ', '_')}_J{j_ini}_J{j_fin}.pdf"
        with open(nombre_final, "wb") as f_out:
            writer.write(f_out)
        
        print(f"🎉 Informe Guardado: {nombre_final}")
        
    shutil.rmtree(temp_dir)
    clean_memory()

if __name__ == "__main__":
    main()