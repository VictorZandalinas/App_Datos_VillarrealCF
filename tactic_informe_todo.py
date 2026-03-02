#!/usr/bin/env python3
"""
GENERADOR MAESTRO DE REPORTES DE BALÓN PARADO (ABP) - OPTIMIZADO SERVER
"""

import os
import sys
import re
import shutil
import gc
import io
import textwrap
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg') # <-- CRÍTICO: Evita ventanas y ahorra RAM gráfica
import matplotlib.pyplot as plt

# Intenta importar PyPDF2
try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError as e:
    print(f"❌ Error: Falta PyPDF2. pip install PyPDF2")
    sys.exit(1)

# --- Scripts que necesitan datos de TODA la liga (carpeta general, no equipo específico) ---
# Se usa matching por prefijo numérico para no depender del nombre completo del archivo.
PREFIJOS_LIGA_TACTIC = (
    'tactic1_', 'tactic1.1_', 'tactic1.2_', 'tactic1.3_', 'tactic1.4_',
    'tactic2.2.1_',
)

def _es_script_liga_tactic(script_path):
    """Devuelve True si el script tactic debe leer de la carpeta general."""
    nombre = os.path.basename(script_path)
    return any(nombre.startswith(p) for p in PREFIJOS_LIGA_TACTIC)

# --- 1. CONFIGURACIÓN DE MEMORIA Y PARCHE PANDAS ---

def clean_memory():
    """Limpia gráficos y fuerza la recolección de basura de RAM."""
    plt.close('all')
    gc.collect()

# Guardamos la función original de pandas
_original_read_parquet = pd.read_parquet

# Variable global para indicar si el script actual es de "liga" (datos generales)
_current_script_is_liga = False

# --- Helpers para resolución de rutas por equipo ---

def _norm_team(s):
    for x in [' cf', ' fc', ' rc', ' rcd', ' ca', ' ud', ' ', '-']:
        s = s.lower().replace(x, '')
    return s

def _find_team_folder(name):
    _b = 'data_por_equipos'
    if not os.path.exists(_b):
        return None
    _t = _norm_team(name)
    for _d in os.listdir(_b):
        if os.path.isdir(os.path.join(_b, _d)):
            _dn = _norm_team(_d)
            if _t in _dn or _dn in _t:
                return _d
    return None

def _to_team_path(orig, folder):
    if folder is None:
        return None
    c = orig[2:] if orig.startswith('./') else orig
    t = os.path.join('data_por_equipos', folder, c)
    if os.path.exists(t):
        return t
    idx = c.find('/')
    if idx > 0:
        t2 = os.path.join('data_por_equipos', folder, c[idx + 1:])
        if os.path.exists(t2):
            return t2
    return None

def _duckdb_filter(path, j_ini, j_fin):
    """Aplica DuckDB pushdown de jornadas. Devuelve DataFrame o None si no aplica."""
    try:
        import duckdb as _ddb
        _con = _ddb.connect()
        _safe = path.replace("'", "\\'")
        _info = _con.execute(
            "DESCRIBE SELECT * FROM read_parquet('" + _safe + "') LIMIT 0"
        ).df()
        _jrow = [(r['column_name'], r['column_type']) for _, r in _info.iterrows()
                 if any(_x in r['column_name'].lower() for _x in ['jornada', 'week', 'semana', 'matchday'])]
        _jcol = _jrow[0][0] if _jrow else None
        _jtype = _jrow[0][1] if _jrow else ''
        if _jcol:
            if any(_t in _jtype.upper() for _t in ['INT', 'BIGINT', 'SMALLINT', 'HUGEINT', 'DOUBLE', 'FLOAT', 'DECIMAL']):
                _sql = "SELECT * FROM read_parquet(?) WHERE " + _jcol + " BETWEEN ? AND ?"
            else:
                _sql = (
                    "SELECT * FROM read_parquet(?) WHERE "
                    "TRY_CAST(TRIM(replace(replace(lower(CAST(" + _jcol + " AS VARCHAR)),'j',''),'w','')) AS INTEGER) BETWEEN ? AND ?"
                )
            _df = _con.execute(_sql, [path, j_ini, j_fin]).df()
            _con.close()
            return _df
        _con.close()
    except Exception:
        pass
    return None

def patched_read_parquet(path, *args, **kwargs):
    """
    Parche con DuckDB pushdown para filtrar por rango de jornadas.
    Lee env vars TACTIC_J_INI / TACTIC_J_FIN para el rango.
    Lee TACTIC_EQUIPO para redirigir a carpetas de equipo y mergear con Villarreal.
    Para scripts de SCRIPTS_LIGA: usa carpeta general (no redirección).
    """
    global _current_script_is_liga

    try:
        j_ini = int(os.environ.get('TACTIC_J_INI', 0))
        j_fin = int(os.environ.get('TACTIC_J_FIN', 99))
    except Exception:
        return _original_read_parquet(path, *args, **kwargs)

    if j_ini == 0:
        return _original_read_parquet(path, *args, **kwargs)

    # Scripts de liga: usar carpeta general (no redirección a carpetas de equipo)
    if _current_script_is_liga:
        if isinstance(path, str) and path.endswith('.parquet'):
            _df = _duckdb_filter(path, j_ini, j_fin)
            if _df is not None:
                if len(_df) == 0:
                    print(f"   ⚠️ ALERTA DE FILTRO: 0 filas. (Rango: {j_ini}-{j_fin})")
                return _df
        return _original_read_parquet(path, *args, **kwargs)

    # Resolver rutas a carpetas de equipo (rival + Villarreal)
    equipo = os.environ.get('TACTIC_EQUIPO', '')
    if equipo and isinstance(path, str) and path.endswith('.parquet'):
        equipo_folder = _find_team_folder(equipo)
        villa_folder = _find_team_folder('Villarreal')
        if villa_folder == equipo_folder:
            villa_folder = None
        p1 = _to_team_path(path, equipo_folder)
        p2 = _to_team_path(path, villa_folder)
        if p1 and p2:
            df1 = _duckdb_filter(p1, j_ini, j_fin)
            if df1 is None:
                df1 = _original_read_parquet(p1, *args, **kwargs)
            df2 = _duckdb_filter(p2, j_ini, j_fin)
            if df2 is None:
                df2 = _original_read_parquet(p2, *args, **kwargs)
            return pd.concat([df1, df2], ignore_index=True).drop_duplicates()
        elif p1:
            path = p1
        elif p2:
            path = p2

    # Intentar DuckDB pushdown (con path posiblemente resuelto)
    if isinstance(path, str) and path.endswith('.parquet'):
        _df = _duckdb_filter(path, j_ini, j_fin)
        if _df is not None:
            if len(_df) == 0:
                print(f"   ⚠️ ALERTA DE FILTRO: 0 filas. (Rango: {j_ini}-{j_fin})")
            return _df

    # Fallback pandas
    df = _original_read_parquet(path, *args, **kwargs)
    candidatos = ['Jornada', 'jornada', 'Jornada_num',
                  'Week', 'week', 'Matchday', 'matchday', 'Round', 'round']
    col_jornada = next((c for c in candidatos if c in df.columns), None)
    if not col_jornada:
        col_jornada = next(
            (c for c in df.columns if any(x in c.lower() for x in ['jornada', 'week', 'match', 'stg'])),
            None
        )
    if col_jornada:
        try:
            s = (df[col_jornada].astype(str).str.lower()
                 .str.replace('j', '', regex=False)
                 .str.replace('week', '', regex=False)
                 .str.replace('matchday', '', regex=False)
                 .str.strip())
            v = pd.to_numeric(s, errors='coerce')
            if v.notna().any():
                filas_antes = len(df)
                df = df[(v >= j_ini) & (v <= j_fin)]
                if filas_antes > 0 and len(df) == 0:
                    print(f"   ⚠️ ALERTA DE FILTRO: {filas_antes} -> 0 filas. Columna: {col_jornada} (Rango: {j_ini}-{j_fin})")
        except Exception as e:
            print(f"   ⚠️ Error filtrando por {col_jornada}: {e}")
    return df

# Aplicar el parche globalmente
pd.read_parquet = patched_read_parquet


# --- 2. CONFIGURACIÓN PRINCIPAL ---

def limpiar_pdfs_antiguos():
    print("🧹 Limpiando PDFs antiguos...")
    for filename in os.listdir('.'):
        if filename.endswith('.pdf'):
            try:
                os.remove(filename)
            except: pass

def natural_sort_key(s):
    filename = os.path.basename(s)
    parts = re.split(r'(\d+\.?\d*)', filename)
    return [float(p) if re.match(r'^\d+\.?\d*$', p) else p.lower() for p in parts if p]

def generar_lista_de_reportes():
    print("🔍 Buscando scripts 'tactic'...")
    patron = re.compile(r'^tactic[0-9].*\.py$')
    scripts = sorted([f for f in os.listdir('.') if os.path.isfile(f) and patron.match(f)], key=natural_sort_key)
    return scripts

REPORTES_A_GENERAR = generar_lista_de_reportes()

TEAM_NAME_MAPPING = {
    'Alavés': {'opta': 'Alavés', 'mediacoach': 'Deportivo Alavés'},
    'Athletic Club': {'opta': 'Athletic Club', 'mediacoach': 'Athletic Club'},
    'Atlético de Madrid': {'opta': 'Atlético de Madrid', 'mediacoach': 'Atlético de Madrid'},
    'Barcelona': {'opta': 'Barcelona', 'mediacoach': 'FC Barcelona'},
    'Celta': {'opta': 'Celta de Vigo', 'mediacoach': 'RC Celta'},
    'Elche': {'opta': 'Elche', 'mediacoach': 'Elche CF'},
    'Espanyol': {'opta': 'Espanyol', 'mediacoach': 'RCD Espanyol'},
    'Getafe': {'opta': 'Getafe', 'mediacoach': 'Getafe CF'},
    'Girona': {'opta': 'Girona', 'mediacoach': 'Girona FC'},
    'Levante': {'opta': 'Levante', 'mediacoach': 'Levante UD'},
    'Mallorca': {'opta': 'Mallorca', 'mediacoach': 'RCD Mallorca'},
    'Osasuna': {'opta': 'Osasuna', 'mediacoach': 'CA Osasuna'},
    'Rayo Vallecano': {'opta': 'Rayo Vallecano', 'mediacoach': 'Rayo Vallecano'},
    'Real Betis': {'opta': 'Real Betis', 'mediacoach': 'Real Betis'},
    'Real Madrid': {'opta': 'Real Madrid', 'mediacoach': 'Real Madrid'},
    'Real Oviedo': {'opta': 'Real Oviedo', 'mediacoach': 'Real Oviedo'},
    'Real Sociedad': {'opta': 'Real Sociedad', 'mediacoach': 'Real Sociedad'},
    'Sevilla': {'opta': 'Sevilla', 'mediacoach': 'Sevilla FC'},
    'Valencia': {'opta': 'Valencia', 'mediacoach': 'Valencia CF'},
    'Villarreal': {'opta': 'Villarreal', 'mediacoach': 'Villarreal CF'},
}

EQUIPOS_OPTA = sorted(list(TEAM_NAME_MAPPING.keys()))
EQUIPOS_MEDIACOACH = [
    'Athletic Club', 'Atlético de Madrid', 'CA Osasuna', 'Deportivo Alavés',
    'Elche CF', 'FC Barcelona', 'Getafe CF', 'Girona FC', 'Levante UD',
    'RC Celta', 'RCD Espanyol', 'RCD Mallorca', 'Rayo Vallecano', 'Real Betis',
    'Real Madrid', 'Real Oviedo', 'Real Sociedad', 'Sevilla FC', 'Valencia CF', 'Villarreal CF'
]

# --- 3. FUNCIONES AUXILIARES DE PORTADA ---

def generar_portada_temporal(equipo, jornada, ruta_salida):
    """Genera la portada (Misma lógica visual que tenías)"""
    print("🎨 Generando portada...")
    try:
        from matplotlib import patheffects
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import numpy as np
        
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        efecto_relieve = [patheffects.withStroke(linewidth=4, foreground='white', alpha=0.9), patheffects.Normal()]
        
        ruta_portada = "assets/portada_tactico_informe.png"
        if os.path.exists(ruta_portada):
            try:
                img_fondo = plt.imread(ruta_portada)
                ax.imshow(img_fondo, aspect='auto', extent=[0, 1, 0, 1])
            except:
                ax.set_facecolor('#1e3d59')
        else:
            ax.set_facecolor('#1e3d59')

        ax.text(0.5, 0.92, 'INFORME SITUACIONES DE JUEGO', ha='center', va='center', fontsize=42, fontweight='bold', color='#1e3d59', family='serif', path_effects=efecto_relieve)
        ax.text(0.20, 0.76, equipo.upper(), ha='center', va='center', fontsize=32, fontweight='bold', color='#e74c3c', family='serif', path_effects=efecto_relieve)
        
        logo_img = buscar_escudo_equipo(equipo)
        if logo_img is not None:
            try:
                imagebox = OffsetImage(logo_img, zoom=0.6)
                ab = AnnotationBbox(imagebox, (0.20, 0.58), frameon=False)
                ax.add_artist(ab)
            except: pass
        
        ax.text(0.5, 0.18, f'Jornada: {jornada}', ha='center', va='center', fontsize=16, color='#34495e', family='serif', path_effects=efecto_relieve)
        
        fig.savefig(ruta_salida, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"❌ Error portada: {e}")
        return False

def buscar_escudo_equipo(equipo):
    from difflib import SequenceMatcher
    if not os.path.exists('assets/escudos'): return None
    equipo_clean = equipo.lower().replace(' ', '').replace('cf', '').replace('fc', '')
    best_match, best_sim = None, 0
    for f in os.listdir('assets/escudos'):
        if not f.endswith('.png'): continue
        name = f.replace('.png', '').lower().replace('_', '').replace('cf', '').replace('fc', '')
        sim = SequenceMatcher(None, equipo_clean, name).ratio()
        if sim > best_sim and sim > 0.4:
            best_sim = sim
            best_match = f
    return plt.imread(f"assets/escudos/{best_match}") if best_match else None

# --- 4. FUNCIÓN DE EJECUCIÓN OPTIMIZADA (SIN SUBPROCESS) ---

def ejecutar_script_en_memoria(script_path, inputs_simulados):
    """
    Ejecuta el script usando exec() en lugar de subprocess.
    Ahorra mucha CPU y RAM al no abrir nuevos interpretes.
    """
    print(f"   ▶️ Procesando: {os.path.basename(script_path)}")

    # Determinar si es script de liga (usa carpeta general)
    is_script_liga = _es_script_liga_tactic(script_path)
    global _current_script_is_liga
    _current_script_is_liga = is_script_liga

    # 1. Redirigir stdin para que el script 'crea' que el usuario escribe
    stdin_original = sys.stdin
    sys.stdin = io.StringIO(inputs_simulados)

    # 2. Crear entorno aislado (Scope)
    # Importante: Pasamos 'pd' ya parcheado para que filtren automáticamente
    script_scope = {
        '__builtins__': __builtins__,
        '__name__': '__main__',
        '__file__': script_path,
        'pd': pd,
        'plt': plt,
        'sys': sys,
        'os': os,
        '_is_script_liga': is_script_liga,  # Flag para el parche
    }
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            codigo = f.read()
        
        # 3. Ejecutar código en el scope (globals=scope, locals=scope)
        # Esto soluciona los errores de clases no definidas
        exec(codigo, script_scope, script_scope)
        
    except Exception as e:
        print(f"   ⚠️ Error ejecutando {script_path}: {e}")
        # import traceback
        # traceback.print_exc()
    finally:
        # 4. Restaurar y Limpiar
        sys.stdin = stdin_original
        script_scope.clear()
        del script_scope
        clean_memory() # <--- Limpieza clave para servidor

# --- 5. SCRIPT PRINCIPAL ---

def main():
    print("=" * 70)
    print("📋 INICIANDO GENERACIÓN DE INFORME (MODO OPTIMIZADO)")
    print("=" * 70)

    limpiar_pdfs_antiguos()

    # Gestión de Argumentos
    if len(sys.argv) > 3:
        nombre_sucio = sys.argv[1]
        jornada_inicio = int(sys.argv[2])
        jornada_fin = int(sys.argv[3])
        jornada = jornada_fin
        
        # Limpieza de nombre
        nombre_recibido = re.sub(r'^\d+\.\s*', '', nombre_sucio).strip()
        equipo_canonico = None

        # Primero buscar en las claves
        for key in TEAM_NAME_MAPPING.keys():
            if nombre_recibido.lower() == key.lower():
                equipo_canonico = key
                break

        # Si no encuentra, buscar en los valores de 'opta'
        if not equipo_canonico:
            for key, valores in TEAM_NAME_MAPPING.items():
                if nombre_recibido.lower() == valores['opta'].lower():
                    equipo_canonico = key
                    break

        if not equipo_canonico:
            print(f"❌ Error: No se encontró '{nombre_recibido}'")
            print(f"   Nombre recibido: '{nombre_recibido}'")
            print(f"   Claves disponibles: {list(TEAM_NAME_MAPPING.keys())}")
            sys.exit(1)
        indice = EQUIPOS_OPTA.index(equipo_canonico)
        print(f"✅ Web: {equipo_canonico} (J{jornada_inicio}-J{jornada_fin})")
    else:
        # Modo manual
        for i, eq in enumerate(EQUIPOS_OPTA, 1): print(f"{i:2d}. {eq}")
        indice = int(input("\n➤ Selecciona equipo: ")) - 1
        equipo_canonico = EQUIPOS_OPTA[indice]
        jornada = input("➤ Jornada: ").strip()
        jornada_inicio, jornada_fin = 1, int(jornada)

    # === SISTEMA DE CHUNKS ===
    # Optimizado para servidores con RAM limitada (4GB)
    # Divide ejecución en grupos de 2 scripts con limpieza agresiva entre chunks
    USE_CHUNKED = os.environ.get('INFORME_USE_CHUNKS', '1') == '1'

    if USE_CHUNKED:
        print("🚀 Modo CHUNKED activado (optimizado para servidor)")
        try:
            from informe_wrapper_chunked import InformeGeneratorChunked
            wrapper = InformeGeneratorChunked(
                tipo_informe='TACTIC',
                chunk_size=2,  # Reducido de 4 a 2 para evitar OOM en scripts de mapa de pases
                team_mappings=TEAM_NAME_MAPPING,
                equipos_opta=EQUIPOS_OPTA,
                equipos_mediacoach=EQUIPOS_MEDIACOACH
            )

            print(f"🎯 Ejecutando wrapper para {equipo_canonico}, J{jornada_inicio}-J{jornada_fin}")
            output_name = wrapper.ejecutar(equipo_canonico, jornada_inicio, jornada_fin)

            if output_name:
                print(f"\n✅ GENERADO: {output_name}")
                print(f"📂 Ubicación: {os.path.abspath(output_name) if os.path.exists(str(output_name)) else 'NO ENCONTRADO'}")

                # Limpiar directorio temporal
                if os.path.exists("reportes_temporales"):
                    print("🧹 Limpiando directorio temporal...")
                    shutil.rmtree("reportes_temporales")

                print("🏁 Proceso completado exitosamente")
                sys.exit(0)
            else:
                print(f"\n❌ Error en generación chunked")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error en modo chunked: {e}")
            print("⚠️ Cayendo a modo legacy...")
            import traceback
            traceback.print_exc()
            # Si falla chunked, continuar con modo original

    # === MODO LEGACY (código original) ===
    print("⚠️ Modo LEGACY activado (alto uso de memoria)")

    # CONFIGURAR VARIABLES DE ENTORNO PARA EL PARCHE PANDAS
    # Así los scripts hijos sabrán qué filtrar al leer parquets
    os.environ['TACTIC_J_INI'] = str(jornada_inicio)
    os.environ['TACTIC_J_FIN'] = str(jornada_fin)
    os.environ['TACTIC_EQUIPO'] = equipo_canonico

    # Preparar directorios
    temp_dir = "reportes_temporales"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Portada
    ruta_portada = os.path.join(temp_dir, "00_portada.pdf")
    generar_portada_temporal(equipo_canonico, int(jornada), ruta_portada)
    pdfs_para_unir = [ruta_portada]
    
    # --- BUCLE DE EJECUCIÓN ---
    for i, script_py in enumerate(REPORTES_A_GENERAR, 1):
        print(f"\n--- [{i}/{len(REPORTES_A_GENERAR)}] Ejecutando: {script_py} ---")
        
        pdfs_antes = set(f for f in os.listdir('.') if f.endswith(".pdf"))
        
        # Preparar respuestas automáticas para los inputs de los scripts
        if 'mediacoach' in script_py.lower():
            idx_mc = EQUIPOS_MEDIACOACH.index(TEAM_NAME_MAPPING[equipo_canonico]['mediacoach'])
            # Mediacoach suele pedir: Indice -> Jornada -> Jornada
            respuestas = f"{idx_mc + 1}\n{jornada}\n{jornada}\n"
        elif 'sportian' in script_py.lower():
            print("   ⚠️ Saltando Sportian (sin mapeo)")
            continue
        else:
            # Opta suele pedir: Indice -> Jornada
            respuestas = f"{indice + 1}\n{jornada}\n"

        # >>> EJECUCIÓN EN MEMORIA (EL CAMBIO IMPORTANTE) <<<
        ejecutar_script_en_memoria(script_py, respuestas)

        # Capturar PDF resultante
        pdfs_despues = set(f for f in os.listdir('.') if f.endswith(".pdf"))
        nuevos = pdfs_despues - pdfs_antes
        
        if nuevos:
            pdf_gen = list(nuevos)[0]
            ruta_dest = os.path.join(temp_dir, f"{i:02d}_{os.path.basename(pdf_gen)}")
            shutil.move(pdf_gen, ruta_dest)
            pdfs_para_unir.append(ruta_dest)
            print(f"✅ PDF generado: {pdf_gen}")
        else:
            print(f"⚠️ {script_py} no generó PDF")

    # --- UNIÓN FINAL ---
    if len(pdfs_para_unir) > 1:
        print("\n🔄 Uniendo reportes...")
        writer = PdfWriter()
        pdfs_para_unir.sort(key=natural_sort_key)
        
        files_abiertos = []
        try:
            for p_path in pdfs_para_unir:
                if os.path.exists(p_path) and os.path.getsize(p_path) > 100:
                    f = open(p_path, 'rb')
                    files_abiertos.append(f)
                    reader = PdfReader(f)
                    for page in reader.pages: writer.add_page(page)
            
            output_name = f"Informe_Situaciones_Juego_{equipo_canonico.replace(' ', '_')}_J{jornada_inicio}_J{jornada_fin}.pdf"
            with open(output_name, "wb") as f_out: writer.write(f_out)
            print(f"\n✅ GENERADO: {output_name}")
            
        finally:
            for f in files_abiertos: f.close()
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    clean_memory()

if __name__ == "__main__":
    main()