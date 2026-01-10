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
import textwrap

# Intenta importar las librerías necesarias y da un aviso si faltan.
try:
    from PyPDF2 import PdfMerger
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
    
    patron = re.compile(r'^tactic[0-9].*\.py$')
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

# Lista de equipos en el orden que usan los SCRIPTS DE OPTA
EQUIPOS_OPTA = sorted(list(TEAM_NAME_MAPPING.keys()))

# Lista de equipos en el orden que usan los SCRIPTS DE MEDIACOACH
# ¡¡IMPORTANTE!! Este orden debe coincidir EXACTAMENTE con el menú de los scripts de mediacoach.
EQUIPOS_MEDIACOACH = [
    'Athletic Club', 'Atlético de Madrid', 'CA Osasuna', 'Deportivo Alavés',
    'Elche CF', 'FC Barcelona', 'Getafe CF', 'Girona FC', 'Levante UD',
    'RC Celta', 'RCD Espanyol', 'RCD Mallorca', 'Rayo Vallecano', 'Real Betis',
    'Real Madrid', 'Real Oviedo', 'Real Sociedad', 'Sevilla FC', 'Valencia CF', 'Villarreal CF'
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
        ruta_portada = "assets/portada_tactico_informe.png"
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
        
        # 2. TÍTULO "INFORME TACTICO" - MÁS ARRIBA (0.92 en lugar de 0.88)
        ax.text(0.5, 0.92, 'INFORME SITUACIONES DE JUEGO',
                ha='center', va='center',
                fontsize=42, fontweight='bold',
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
    import sys
    print("=" * 70)
    print("📋 CONFIGURACIÓN DEL REPORTE TÁCTICO")
    print("=" * 70)

    limpiar_pdfs_antiguos()
    equipos_disponibles = EQUIPOS_OPTA

    if len(sys.argv) > 2:
        # Dash pasa el NOMBRE del equipo, no el índice
        nombre_recibido = sys.argv[1]
        jornada = sys.argv[2]
        
        # LIMPIEZA: Quitamos el "1. ", "20. ", etc. para que coincida con el mapeo
        nombre_recibido = re.sub(r'^\d+\.\s*', '', nombre_sucio).strip()

        # Buscar el equipo por nombre
        equipo_canonico = None
        for key in TEAM_NAME_MAPPING.keys():
            if nombre_recibido.lower() == key.lower():
                equipo_canonico = key
                break
        
        if not equipo_canonico:
            print(f"❌ Error: No se encontró '{nombre_recibido}'")
            sys.exit(1)
        
        print(f"✅ Ejecución desde Dash: {equipo_canonico} - Jornada {jornada}")
        indice = EQUIPOS_OPTA.index(equipo_canonico)
    else:
        for i, equipo in enumerate(equipos_disponibles, 1): print(f"{i:2d}. {equipo}")
        indice = int(input("\n➤ Selecciona un equipo: ")) - 1
        equipo_canonico = equipos_disponibles[indice]
        jornada = input("➤ Introduce la jornada: ").strip()

    nombre_mediacoach = TEAM_NAME_MAPPING[equipo_canonico]['mediacoach']
    indice_mediacoach = EQUIPOS_MEDIACOACH.index(nombre_mediacoach)
    seleccion_equipo_mediacoach_str = str(indice_mediacoach + 1)
    seleccion_equipo_opta_str = str(indice + 1)

    temp_dir = "reportes_temporales"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    ruta_portada = os.path.join(temp_dir, "00_portada.pdf")
    generar_portada_temporal(equipo_canonico, int(jornada), ruta_portada)
    
    pdfs_para_unir = [ruta_portada]
    
    # --- 4. EJECUTAR SCRIPTS INDIVIDUALES ---
    for i, script_py in enumerate(REPORTES_A_GENERAR, 1):
        print(f"\n--- [{i}/{len(REPORTES_A_GENERAR)}] Ejecutando: {script_py} ---")

        if not os.path.exists(script_py):
            print(f"⚠️  AVISO: El script '{script_py}' no se encontró. Saltando...")
            continue
        
        # NUEVO: Limpiar cualquier PDF existente ANTES de ejecutar el script
        pdfs_antes = set(f for f in os.listdir('.') if f.endswith(".pdf"))
        
        # Verificar si el script usa datos de OPTA aunque tenga "mediacoach" en el nombre
        usa_datos_opta = False
        if 'mediacoach' in script_py.lower():
            # Leer el script para ver si usa datos de OPTA
            with open(script_py, 'r', encoding='utf-8') as f:
                contenido = f.read()
                if 'datos_opta_parquet' in contenido or 'abp_events.parquet' in contenido:
                    usa_datos_opta = True

        if 'mediacoach' in script_py.lower() and not usa_datos_opta:
            nombre_equipo_para_script = TEAM_NAME_MAPPING[equipo_canonico]['mediacoach']
            respuestas_para_script = f"{seleccion_equipo_mediacoach_str}\n{jornada}\n"
            modo = "MediaCoach"
            seleccion_enviada = seleccion_equipo_mediacoach_str
        else:
            nombre_equipo_para_script = TEAM_NAME_MAPPING[equipo_canonico]['opta']
            respuestas_para_script = f"{seleccion_equipo_opta_str}\n{jornada}\n"
            modo = "Opta"
            seleccion_enviada = seleccion_equipo_opta_str

        print(f"    |-> Enviando respuestas automáticas (Modo {modo})...")
        print(f"    |-> Selección de Equipo: '{seleccion_enviada}' (para {nombre_equipo_para_script})")
        print(f"    |-> Jornada: '{jornada}'")

        j_inicio_val = int(sys.argv[2]) if len(sys.argv) > 3 else 1
        j_fin_val = int(sys.argv[3]) if len(sys.argv) > 3 else int(jornada)

        injected_code = "import matplotlib, pandas as pd, sys, numpy as np, re; " \
                        "matplotlib.use('Agg'); " \
                        "_orig = pd.read_parquet; " \
                        "def _patched(*args, **kwargs): " \
                        "    df = _orig(*args, **kwargs); " \
                        "    pats = ['jornada', 'week', 'matchday', 'match_week', 'stg', 'fecha']; " \
                        "    for col in df.columns: " \
                        "        if any(p in col.lower() for p in pats): " \
                        "            try: " \
                        "                sv = df[col].astype(str).str.lower().str.replace('j', '').str.strip(); " \
                        "                nv = pd.to_numeric(sv, errors='coerce'); " \
                        "                if nv.notna().any(): " \
                        "                    df = df[(nv >= " + str(j_inicio_val) + ") & (nv <= " + str(j_fin_val) + ")]; " \
                        "            except: pass; " \
                        "    return df; " \
                        "pd.read_parquet = _patched; " \
                        "exec(open('" + script_py + "', encoding='utf-8').read())"

        
        comando = ["python3", "-c", injected_code]
        
        my_env = os.environ.copy()
        
        try:
            proceso = subprocess.Popen(
                comando,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                env=my_env,
                bufsize=1
            )

            try:
                stdout, _ = proceso.communicate(input=respuestas_para_script, timeout=180)
                
                print("--- SALIDA DEL SCRIPT ---")
                lineas_salida = stdout.splitlines()
                for linea in lineas_salida[-20:]:
                    print(f"    | {linea.strip()}")
                
                if any('guardado' in linea.lower() or 'éxito' in linea.lower() for linea in lineas_salida):
                    print("    | ✅ El script reportó éxito")
                else:
                    print("    | ⚠️  El script no reportó éxito claramente")
                print("-------------------------")

            except subprocess.TimeoutExpired:
                proceso.kill()
                stdout, _ = proceso.communicate()
                print("--- SALIDA DEL SCRIPT (INCOMPLETA por timeout) ---")
                for linea in stdout.splitlines()[-10:]:
                    print(f"    | {linea.strip()}")
                print("---------------------------------------")

            # NUEVO: Detectar el PDF recién generado comparando antes/después
            pdfs_despues = set(f for f in os.listdir('.') if f.endswith(".pdf"))
            pdfs_nuevos = pdfs_despues - pdfs_antes
            
            if pdfs_nuevos:
                pdf_generado = list(pdfs_nuevos)[0]  # Tomamos el nuevo PDF
                print(f"    |-> ✅ PDF detectado: '{pdf_generado}'")
                
                # Mover inmediatamente sin validación del nombre
                ruta_origen = pdf_generado
                script_base = script_py.replace('.py', '')
                nuevo_nombre = f"{i:02d}_{script_base}_{os.path.basename(pdf_generado)}"
                ruta_destino = os.path.join(temp_dir, nuevo_nombre)
                shutil.move(ruta_origen, ruta_destino)
                pdfs_para_unir.append(ruta_destino)
                print(f"✅ Éxito. PDF '{os.path.basename(pdf_generado)}' movido a carpeta temporal.")
            else:
                print(f"⚠️  AVISO: No se detectó PDF nuevo para {script_py}")

        except Exception as e:
            print(f"❌ Error ejecutando {script_py}: {e}")

    # --- 5. UNIR TODOS LOS PDFS RECOLECTADOS ---
    if len(pdfs_para_unir) > 1:
        print("\n" + "=" * 70)
        print("🔄 Uniendo todos los reportes en el informe final...")
        
        merger = PdfMerger()
        pdfs_para_unir.sort(key=natural_sort_key) 

        for pdf_path in pdfs_para_unir:
            print(f"   -> Añadiendo {os.path.basename(pdf_path)}")
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 100:
                try:
                    merger.append(pdf_path)
                except Exception as e:
                    print(f"   -> ❌ ERROR al procesar '{os.path.basename(pdf_path)}': {e}. El fichero podría estar corrupto y será omitido.")
            else:
                print(f"   -> ⚠️  AVISO: El fichero '{os.path.basename(pdf_path)}' está vacío o no existe y será omitido.")
            
        fecha_actual = datetime.now().strftime("%Y%m%d_%H%M")
        output_final_pdf = f"Informe_Situaciones_juego_Completo_{equipo_canonico.replace(' ', '_')}_J{jornada}_{fecha_actual}.pdf"
        
        merger.write(output_final_pdf)
        merger.close()
        
        print(f"\n✅ ¡REPORTE FINAL GENERADO! -> {output_final_pdf}")
    else:
        print("\n⚠️ No se generaron suficientes reportes para crear un PDF final.")
        
    print("🧹 Limpiando archivos y carpetas temporales...")
    shutil.rmtree(temp_dir)
    print("✨ Proceso completado.")

if __name__ == "__main__":
    main()