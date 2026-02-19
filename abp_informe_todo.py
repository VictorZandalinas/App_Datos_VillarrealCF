#!/usr/bin/env python3
"""
GENERADOR MAESTRO DE REPORTES DE BALÓN PARADO (ABP)

Este script orquesta la ejecución de múltiples análisis individuales de ABP,
genera una portada personalizada y une todos los resultados en un único
informe en formato PDF.

Uso:
  python3 abp_informe_todo.py

"""

import os
import sys
import re
import subprocess
import shutil
from matplotlib import patheffects
from datetime import datetime
from difflib import get_close_matches
from PyPDF2 import PdfReader, PdfWriter
import textwrap

# Intenta importar las librerías necesarias y da un aviso si faltan.
try:
    from PyPDF2 import PdfMerger, PdfReader, PdfWriter 
    import matplotlib
    matplotlib.use('Agg') # <-- CRÍTICO: Evita que matplotlib abra ventanas.
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"❌ Error: Falta una librería necesaria. Por favor, instálala.")
    print(f"-> Detalle: {e}")
    print("-> Posible solución: pip install PyPDF2 matplotlib")
    sys.exit(1)


# --- 1. CONFIGURACIÓN PRINCIPAL ---
def limpiar_pdfs_antiguos():
    """Elimina todos los PDFs existentes antes de empezar el proceso."""
    print("🧹 Limpiando PDFs antiguos...")
    pdfs_encontrados = []
    for filename in os.listdir('.'):
        if filename.endswith('.pdf'):
            try:
                os.remove(filename)
                pdfs_encontrados.append(filename)
            except Exception as e:
                print(f"   ⚠️  No se pudo eliminar {filename}: {e}")
    
    if pdfs_encontrados:
        print(f"✅ Eliminados {len(pdfs_encontrados)} PDFs antiguos:")
        for pdf in pdfs_encontrados:
            print(f"   🗑️  {pdf}")
    else:
        print("✅ No se encontraron PDFs antiguos para limpiar")

def natural_sort_key(s):
    """
    Crea una clave para ordenar cadenas de forma natural, tratando los
    números con y sin decimales como una secuencia numérica continua.
    Ej: 'abp2', 'abp2.1', 'abp3'
    """
    # Extrae el nombre del fichero para asegurar que la ordenación se basa en él
    filename = os.path.basename(s)
    
    # Divide la cadena en partes de texto y partes de números
    parts = re.split(r'(\d+\.?\d*)', filename)
    
    # Procesa cada parte
    key_parts = []
    for part in parts:
        if not part:
            continue
        # Si la parte es un número (con o sin decimal), conviértela a float para una comparación numérica correcta
        if re.match(r'^\d+\.?\d*$', part):
            key_parts.append(float(part))
        # Si no, es texto, conviértelo a minúsculas
        else:
            key_parts.append(part.lower())
            
    return key_parts

def generar_lista_de_reportes():
    """Escanea la carpeta y crea lista ordenada de scripts 'abp'."""
    print("🔍 Escaneando la carpeta en busca de scripts 'abp' para ejecutar...")
    
    patron = re.compile(r'^abp[0-9].*\.py$')
    scripts_encontrados = [
        f for f in os.listdir('.') 
        if os.path.isfile(f) and patron.match(f)
    ]
    
    scripts_encontrados.sort(key=natural_sort_key)
    
    print(f"✅ {len(scripts_encontrados)} scripts encontrados:")
    for i, script in enumerate(scripts_encontrados, 1):
        print(f"   {i:2d}. {script}")
        
    return scripts_encontrados

# Se genera la lista de reportes dinámicamente al iniciar el script.
REPORTES_A_GENERAR = generar_lista_de_reportes()

# Diccionario maestro para la normalización de nombres de equipos.
# Diccionario maestro para la normalización de nombres de equipos.
TEAM_NAME_MAPPING = {
    'Alavés': {'opta': 'Alavés', 'mediacoach': 'Deportivo Alavés', 'sportian': 'Alavés'},
    'Athletic Club': {'opta': 'Athletic Club', 'mediacoach': 'Athletic Club', 'sportian': 'Athletic Club'},
    'Atlético de Madrid': {'opta': 'Atlético de Madrid', 'mediacoach': 'Atlético de Madrid', 'sportian': 'Atlético de Madrid'},
    'Barcelona': {'opta': 'Barcelona', 'mediacoach': 'FC Barcelona', 'sportian': 'Barcelona'},
    'Celta': {'opta': 'Celta de Vigo', 'mediacoach': 'RC Celta', 'sportian': 'Celta de Vigo'},
    'Elche': {'opta': 'Elche', 'mediacoach': 'Elche CF', 'sportian': 'Elche'},
    'Espanyol': {'opta': 'Espanyol', 'mediacoach': 'RCD Espanyol', 'sportian': 'Espanyol'},
    'Getafe': {'opta': 'Getafe', 'mediacoach': 'Getafe CF', 'sportian': 'Getafe'},
    'Girona': {'opta': 'Girona', 'mediacoach': 'Girona FC', 'sportian': 'Girona'},
    'Levante': {'opta': 'Levante', 'mediacoach': 'Levante UD', 'sportian': 'Levante'},
    'Mallorca': {'opta': 'Mallorca', 'mediacoach': 'RCD Mallorca', 'sportian': 'Mallorca'},
    'Osasuna': {'opta': 'Osasuna', 'mediacoach': 'CA Osasuna', 'sportian': 'Osasuna'},
    'Rayo Vallecano': {'opta': 'Rayo Vallecano', 'mediacoach': 'Rayo Vallecano', 'sportian': 'Rayo Vallecano'},
    'Real Betis': {'opta': 'Real Betis', 'mediacoach': 'Real Betis', 'sportian': 'Real Betis'},
    'Real Madrid': {'opta': 'Real Madrid', 'mediacoach': 'Real Madrid', 'sportian': 'Real Madrid'},
    'Real Oviedo': {'opta': 'Real Oviedo', 'mediacoach': 'Real Oviedo', 'sportian': 'Real Oviedo'},
    'Real Sociedad': {'opta': 'Real Sociedad', 'mediacoach': 'Real Sociedad', 'sportian': 'Real Sociedad'},
    'Sevilla': {'opta': 'Sevilla', 'mediacoach': 'Sevilla FC', 'sportian': 'Sevilla'},
    'Valencia': {'opta': 'Valencia', 'mediacoach': 'Valencia CF', 'sportian': 'Valencia'},
    'Villarreal': {'opta': 'Villarreal', 'mediacoach': 'Villarreal CF', 'sportian': 'Villarreal'},
}

# Lista de equipos en el orden que usan los SCRIPTS DE OPTA
EQUIPOS_OPTA = sorted(list(TEAM_NAME_MAPPING.keys()))

# Lista de equipos en el orden que usan los SCRIPTS DE MEDIACOACH
# ¡¡IMPORTANTE!! Este orden debe coincidir EXACTAMENTE con sorted(df['EQUIPO'].unique()) del parquet.
EQUIPOS_MEDIACOACH = [
    'Athletic Club', 'Atlético de Madrid', 'CA Osasuna', 'Deportivo Alavés',
    'Elche CF', 'FC Barcelona', 'Getafe CF', 'Girona FC', 'Levante UD',
    'RC Celta', 'RCD Espanyol', 'RCD Mallorca', 'Rayo Vallecano', 'Real Betis',
    'Real Madrid', 'Real Oviedo', 'Real Sociedad', 'Sevilla FC', 'Valencia CF', 'Villarreal CF'
]

# NUEVO: Lista de equipos en el orden que usa el script de SPORTIAN (Extraído de tu lista)
EQUIPOS_SPORTIAN = [
    'Alavés', 'Athletic Club', 'Atlético de Madrid', 'Barcelona', 'Celta de Vigo',
    'Elche', 'Espanyol', 'Getafe', 'Girona', 'Levante', 'Mallorca', 'Osasuna',
    'Rayo Vallecano', 'Real Betis', 'Real Madrid', 'Real Oviedo', 'Real Sociedad',
    'Sevilla', 'Valencia', 'Villarreal'
]


# --- 2. FUNCIONES AUXILIARES ---

def obtener_clave_equipo(nombre_completo):
    """
    Extrae la parte más significativa del nombre de un equipo para la validación.
    Ignora palabras comunes y devuelve la palabra más larga y representativa.
    """
    palabras_a_ignorar = {'real', 'de', 'club', 'deportivo', 'atlético', 'unión', 'ca', 'rc', 'fc', 'cf'}
    
    nombre_normalizado = nombre_completo.lower()
    for acentuada, sin_acento in [('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u')]:
        nombre_normalizado = nombre_normalizado.replace(acentuada, sin_acento)
    
    palabras = nombre_normalizado.split()
    palabras_significativas = [p for p in palabras if p not in palabras_a_ignorar]
    
    if not palabras_significativas:
        return nombre_normalizado.replace(' ', '')
        
    return max(palabras_significativas, key=len)

def generar_portada_temporal(equipo, jornada, ruta_salida):
    """Genera la portada en un PDF temporal con diseño mejorado"""
    print("🎨 Generando portada...")
    try:
        from matplotlib import patheffects
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import numpy as np
        
        # Crear figura A4 horizontal
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        # Efecto de relieve para todos los textos
        efecto_relieve = [
            patheffects.withStroke(linewidth=4, foreground='white', alpha=0.9),
            patheffects.Normal()
        ]
        
        # 1. Cargar imagen de fondo (si existe)
        ruta_portada = "assets/portada_abp_informe.png"
        if os.path.exists(ruta_portada):
            try:
                img_fondo = plt.imread(ruta_portada)
                ax.imshow(img_fondo, aspect='auto', extent=[0, 1, 0, 1])
            except Exception as e:
                print(f"⚠️ No se pudo cargar imagen de fondo: {e}")
                ax.set_facecolor('#1e3d59')
        else:
            # Fondo degradado si no hay imagen
            gradient = np.linspace(0, 1, 256).reshape(256, 1)
            gradient = np.hstack((gradient, gradient))
            ax.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto', 
                      cmap='Blues', alpha=0.3, zorder=0)
        
        # 2. TÍTULO "INFORME ABP" - MÁS ARRIBA (0.92 en lugar de 0.88)
        ax.text(0.5, 0.92, 'INFORME ABP',
                ha='center', va='center',
                fontsize=56, fontweight='bold',
                color='#1e3d59',
                family='Georgia',  # Fuente artística
                path_effects=efecto_relieve)
        
        # 4. NOMBRE DEL EQUIPO - MÁS PEQUEÑO Y MÁS ARRIBA
        # Cambiar fontsize de 38 a 32, y y de 0.68 a 0.72
        ax.text(0.20, 0.76, equipo.upper(),
                ha='center', va='center',
                fontsize=32, fontweight='bold',  # <-- Reducido de 38 a 32
                color='#e74c3c',
                family='Georgia',
                path_effects=efecto_relieve)
        
        # 5. ESCUDO DEL EQUIPO - MÁS ARRIBA
        # Cambiar coordenadas de (0.28, 0.52) a (0.28, 0.58)
        logo_img = buscar_escudo_equipo(equipo)
        if logo_img is not None:
            try:
                imagebox = OffsetImage(logo_img, zoom=0.8)  # Reducido zoom de 0.7 a 0.6
                ab = AnnotationBbox(imagebox, (0.20, 0.58), frameon=False)  # Subido de 0.52 a 0.58
                ax.add_artist(ab)
            except Exception as e:
                print(f"⚠️ Error al añadir escudo: {e}")
        
        # 6. FECHA Y JORNADA - Con relieve
        fecha_actual = datetime.now().strftime("%d/%m/%Y - %H:%M")
        ax.text(0.5, 0.18,
                f'Jornada: {jornada}',  # JORNADA AÑADIDA
                ha='center', va='center',
                fontsize=16,
                color='#34495e',
                family='Georgia',
                path_effects=efecto_relieve)
        
        # 7. Línea decorativa
        ax.plot([0.1, 0.9], [0.25, 0.25], 'k-', linewidth=2, alpha=0.3)
        
        # Guardar
        fig.savefig(ruta_salida, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print("✅ Portada generada exitosamente.")
        return True
        
    except Exception as e:
        print(f"❌ Error crítico al generar la portada: {e}")
        import traceback
        traceback.print_exc()
        return False


def buscar_escudo_equipo(equipo):
    """Busca el escudo del equipo por similitud en la carpeta assets/escudos"""
    from difflib import SequenceMatcher
    
    if not os.path.exists('assets/escudos'):
        return None
    
    escudos_disponibles = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
    if not escudos_disponibles:
        return None
    
    # Limpiar nombre del equipo
    equipo_clean = equipo.lower().replace(' ', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '').replace('rc', '').replace('ca', '')
    
    best_match = None
    best_similarity = 0
    
    # Buscar el escudo más similar
    for escudo_file in escudos_disponibles:
        escudo_name = escudo_file.replace('.png', '').lower().replace('_', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '').replace('rc', '').replace('ca', '')
        similarity = SequenceMatcher(None, equipo_clean, escudo_name).ratio()
        
        if similarity > best_similarity and similarity > 0.4:
            best_similarity = similarity
            best_match = escudo_file
    
    if best_match:
        try:
            logo_path = f"assets/escudos/{best_match}"
            print(f"🎯 Escudo encontrado para '{equipo}': {best_match} (similitud: {best_similarity:.2f})")
            return plt.imread(logo_path)
        except Exception as e:
            print(f"⚠️ Error al cargar {best_match}: {e}")
    else:
        print(f"⚠️ No se encontró escudo para '{equipo}'")
    
    return None


# --- 3. SCRIPT PRINCIPAL ---

def main():
    print("=" * 70)
    print("📋 INICIANDO GENERACIÓN DE INFORME")
    print("=" * 70)

    limpiar_pdfs_antiguos()

    # --- Lógica de Argumentos o Selección Manual ---
    if len(sys.argv) > 3:
        nombre_sucio = sys.argv[1]
        jornada_inicio = int(sys.argv[2])
        jornada_fin = int(sys.argv[3])
        jornada = jornada_fin
        nombre_recibido = re.sub(r'^\d+\.\s*', '', nombre_sucio).strip()
        equipo_canonico = None

        print(f"🔍 DEBUG: Nombre recibido desde web: '{nombre_recibido}'")

        # Buscar nombre canónico en el Mapping
        # Primero buscar en las claves
        for key in TEAM_NAME_MAPPING.keys():
            if nombre_recibido.lower() == key.lower():
                equipo_canonico = key
                print(f"✅ DEBUG: Encontrado en claves canónicas: '{key}'")
                break

        # Si no encuentra, buscar en los valores de 'opta'
        if not equipo_canonico:
            for key, valores in TEAM_NAME_MAPPING.items():
                if nombre_recibido.lower() == valores['opta'].lower():
                    equipo_canonico = key
                    print(f"✅ DEBUG: Encontrado en valores Opta: '{key}' (valor opta: '{valores['opta']}')")
                    break

        if not equipo_canonico:
            print(f"❌ Error: No se encontró '{nombre_recibido}' en el mapping.")
            print(f"   Nombre recibido: '{nombre_recibido}'")
            print(f"   Claves disponibles: {list(TEAM_NAME_MAPPING.keys())}")
            print(f"   Valores Opta disponibles: {[v['opta'] for v in TEAM_NAME_MAPPING.values()]}")
            sys.exit(1)

        # Para Opta seguimos usando índice si es necesario, o nombre si cambiaste esos scripts
        # Asumimos que Opta sigue funcionando por índice:
        print(f"🔍 DEBUG: Buscando '{equipo_canonico}' en EQUIPOS_OPTA: {EQUIPOS_OPTA}")
        try:
            indice = EQUIPOS_OPTA.index(equipo_canonico)
            print(f"✅ DEBUG: Índice encontrado: {indice} (equipo: {EQUIPOS_OPTA[indice]})")
        except ValueError:
            print(f"❌ DEBUG: '{equipo_canonico}' NO encontrado en EQUIPOS_OPTA, usando índice 0 (Alavés)")
            indice = 0 # Fallback por seguridad

        print(f"✅ Web: {equipo_canonico} (J{jornada_inicio}-J{jornada_fin}), índice Opta: {indice}")
    else:
        # Selección manual
        for i, eq in enumerate(EQUIPOS_OPTA, 1): print(f"{i:2d}. {eq}")
        indice = int(input("\n➤ Selecciona equipo: ")) - 1
        equipo_canonico = EQUIPOS_OPTA[indice]
        jornada = input("➤ Jornada: ").strip()
        jornada_inicio, jornada_fin = 1, int(jornada)

    # === SISTEMA DE CHUNKS ===
    # Optimizado para servidores con RAM limitada (4GB)
    # Divide ejecución en grupos de 3 scripts con limpieza agresiva entre chunks
    USE_CHUNKED = os.environ.get('INFORME_USE_CHUNKS', '1') == '1'

    if USE_CHUNKED:
        print("🚀 Modo CHUNKED activado (optimizado para servidor)")
        try:
            from informe_wrapper_chunked import InformeGeneratorChunked
            wrapper = InformeGeneratorChunked(
                tipo_informe='ABP',
                chunk_size=3,
                team_mappings=TEAM_NAME_MAPPING,
                equipos_opta=EQUIPOS_OPTA,
                equipos_mediacoach=EQUIPOS_MEDIACOACH
            )
            output_name = wrapper.ejecutar(equipo_canonico, jornada_inicio, jornada_fin)
            if output_name:
                print(f"\n✅ GENERADO: {output_name}")
                sys.exit(0)
            else:
                print(f"\n❌ Error en generación chunked")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error en modo chunked: {e}")
            print("⚠️ Cayendo a modo legacy...")
            # Si falla chunked, continuar con modo original

    # === MODO LEGACY (código original) ===
    print("⚠️ Modo LEGACY activado (alto uso de memoria)")

    # --- Preparación de temporales ---
    temp_dir = "reportes_temporales"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    ruta_portada = os.path.join(temp_dir, "00_portada.pdf")
    generar_portada_temporal(equipo_canonico, int(jornada), ruta_portada)
    pdfs_para_unir = [ruta_portada]
    
    # --- Bucle de Ejecución de Scripts ---
    for i, script_py in enumerate(REPORTES_A_GENERAR, 1):
        print(f"\n--- [{i}/{len(REPORTES_A_GENERAR)}] Ejecutando: {script_py} ---")
        if not os.path.exists(script_py): 
            print(f"⚠️ El archivo {script_py} no existe. Saltando...")
            continue
        
        pdfs_antes = set(f for f in os.listdir('.') if f.endswith(".pdf"))
        
        # --- DEFINICIÓN DE RESPUESTAS (INPUTS) ---
        if 'mediacoach' in script_py.lower():
            # Mediacoach sigue usando índices (si no lo has cambiado)
            idx_mc = EQUIPOS_MEDIACOACH.index(TEAM_NAME_MAPPING[equipo_canonico]['mediacoach'])
            respuestas = f"{idx_mc + 1}\n{jornada}\n"
            
        elif 'sportian' in script_py.lower():
            # === CORRECCIÓN AQUÍ ===
            # Ya NO calculamos índice. Enviamos el NOMBRE exacto.
            nombre_sportian = TEAM_NAME_MAPPING[equipo_canonico]['sportian']
            print(f"   🎯 Enviando nombre directo a Sportian: '{nombre_sportian}'")
            respuestas = f"{nombre_sportian}\n"
            # =======================
            
        else:
            # Otros (Opta/Wyscout)
            respuestas = f"{indice + 1}\n{jornada}\n"

        # --- CÓDIGO INYECTADO (FILTRO JORNADAS con DuckDB pushdown) ---
        injected_code = textwrap.dedent(f"""\
            import matplotlib, pandas as pd, sys, numpy as np
            matplotlib.use('Agg')
            _j0, _j1 = {jornada_inicio}, {jornada_fin}
            _orig_rp = pd.read_parquet
            def _r(path, *a, **kw):
                try:
                    import duckdb as _ddb
                    if not (isinstance(path, str) and path.endswith('.parquet')):
                        return _orig_rp(path, *a, **kw)
                    _con = _ddb.connect()
                    _safe = str(path).replace("'", "\\'")
                    _info = _con.execute("DESCRIBE SELECT * FROM read_parquet('" + _safe + "') LIMIT 0").df()
                    _jcol = next((_c for _c in _info['column_name'] if any(_x in _c.lower() for _x in ['jornada','week','semana','matchday'])), None)
                    if _jcol:
                        _sql = ("SELECT * FROM read_parquet(?) WHERE "
                                "TRY_CAST(TRIM(replace(replace(lower(CAST(" + _jcol + " AS VARCHAR)),'j',''),'w','')) AS INTEGER) BETWEEN ? AND ?")
                        _df = _con.execute(_sql, [path, _j0, _j1]).df()
                        _con.close()
                        return _df
                    _con.close()
                except Exception:
                    pass
                df = _orig_rp(path, *a, **kw)
                for c in df.columns:
                    if any(x in c.lower() for x in ['jornada', 'week', 'semana']):
                        try:
                            s = df[c].astype(str).str.lower().str.replace('j', '').str.replace('w', '').str.strip()
                            v = pd.to_numeric(s, errors='coerce')
                            if v.notna().any():
                                df = df[(v >= _j0) & (v <= _j1)]
                                break
                        except: pass
                return df
            pd.read_parquet = _r

            # Ejecutamos el script original
            exec(open('{script_py}', encoding='utf-8').read())
        """)

        # Ejecución del subproceso
        proceso = subprocess.Popen(["python3", "-c", injected_code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        stdout, _ = proceso.communicate(input=respuestas, timeout=300) # Aumentado timeout por seguridad
        
        # Log parcial para depurar si falla
        print(f"SALIDA (últimos bytes): {stdout[-200:]}") 

        # Captura de nuevos PDFs
        pdfs_despues = set(f for f in os.listdir('.') if f.endswith(".pdf"))
        nuevos = pdfs_despues - pdfs_antes
        if nuevos:
            pdf_gen = list(nuevos)[0]
            ruta_dest = os.path.join(temp_dir, f"{i:02d}_{os.path.basename(pdf_gen)}")
            shutil.move(pdf_gen, ruta_dest)
            pdfs_para_unir.append(ruta_dest)
            print(f"✅ PDF '{pdf_gen}' capturado correctamente.")
        else:
            print(f"⚠️ No se generó PDF nuevo en este paso.")

    # --- Unión Final ---
    if len(pdfs_para_unir) > 1:
        print("\n🔄 Uniendo reportes...")
        writer = PdfWriter()
        pdfs_para_unir.sort(key=natural_sort_key) # Asegúrate de tener esta función definida arriba
        
        for p_path in pdfs_para_unir:
            if os.path.exists(p_path) and os.path.getsize(p_path) > 100:
                reader = PdfReader(p_path)
                for page in reader.pages: writer.add_page(page)
                
        output_name = f"Informe_ABP_{equipo_canonico.replace(' ', '_')}_J{jornada_inicio}_J{jornada_fin}.pdf"
        with open(output_name, "wb") as f: writer.write(f)
        print(f"\n✅ GENERADO: {output_name}")
    
    # Limpieza final
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()