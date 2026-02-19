#!/usr/bin/env python3
import os
import sys
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as patheffects
import subprocess
import textwrap
import json
import tempfile
from datetime import datetime
from difflib import SequenceMatcher
import unicodedata
import re
import gc
import warnings
from PyPDF2 import PdfReader, PdfWriter

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE SCRIPTS ---
# Ordenado alfabéticamente para una ejecución predecible.
SCRIPT_CONFIG = {
    'fisico1_mediacoach_minutos_jugados': {'func': 'generar_reporte_personalizado', 'tipo_datos': 'ultimas_5', 'params_map': {'equipo': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico2_mediacoach_distancias_recorridas': {'func': 'generar_reporte_distancias_personalizado', 'tipo_datos': 'ultimas_5', 'params_map': {'equipo': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico3_mediacoach_distancias_recorridas_villarrealcf': {'func': 'generar_reporte_villarreal_personalizado', 'tipo_datos': 'ultimas_5', 'equipo_fijo': 'Villarreal CF', 'params_map': {'jornadas': 'jornadas_vcf_a_pasar'}},
    'fisico4_mediacoach_distancias_por_zonas': {'func': 'generar_reporte_zonas_personalizado', 'tipo_datos': 'ultimas_5', 'params_map': {'equipo': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico5_mediacoach_sprints': {'func': 'generar_reporte_sprints_personalizado', 'tipo_datos': 'ultimas_5', 'params_map': {'equipo': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico6_mediacoach_sprints_villarrealcf': {'func': 'generar_reporte_sprints_villarreal_personalizado', 'tipo_datos': 'ultimas_5', 'equipo_fijo': 'Villarreal CF', 'params_map': {'jornadas': 'jornadas_vcf_a_pasar'}},
    'fisico7_mediacoach_comparativa_sprints': {'func': 'generar_comparativa_sprints_personalizada', 'tipo_datos': 'ultimas_5_comparativa', 'params_map': {'equipo_rival': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico8_mediacoach_10jugadores_mas_rapidos': {'func': 'generar_reporte_velocidades_personalizado', 'tipo_datos': 'ultimas_5', 'params_map': {'equipo': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico9_mediacoach_datos_promedio': {'func': 'generar_reporte_campo_personalizado', 'tipo_datos': 'temporada_completa_comparativa', 'params_map': {'equipo_rival': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico10_mediacoach_datos_comparacion': {'func': 'generar_reporte_graficos_personalizado', 'tipo_datos': 'temporada_completa_comparativa', 'params_map': {'equipo_rival': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico11_mediacoach_comparativa_vmax': {'func': 'generar_reporte_barras_personalizado', 'tipo_datos': 'temporada_completa_comparativa', 'params_map': {'equipo_rival': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico12_mediacoach_datos_maximos': {'func': 'generar_reporte_campo_maximos', 'tipo_datos': 'temporada_completa_comparativa', 'params_map': {'equipo_rival': 'equipo_principal', 'jornadas': 'jornadas_a_pasar'}},
    'fisico13_ultimos4partidos': {'func': 'generar_4_campos_coordenadas_fijas', 'tipo_datos': 'ultimas_5', 'params_map': {'equipo': 'equipo_principal', 'jornada_maxima': 'jornada_num_ref', 'tipo_partido_filter': 'tipo_partido'}},
    'fisico14_mediacoach_posible_11': {'func': 'generar_posible_11_personalizado', 'tipo_datos': 'ultimas_5', 'params_map': {'equipo': 'equipo_principal', 'jornada': 'jornada_str_ref'}},
}

class GeneradorMaestro:
    def __init__(self):
        self._read_parquet_original = pd.read_parquet
        self.df_para_script = None
        self.main_data_path = "extraccion_mediacoach/data/rendimiento_fisico.parquet"
        print("🚀 Inicializando Generador Maestro de Informes...")
        try:
            self.df_mediacoach_original = pd.read_parquet(self.main_data_path)
            self.df_mediacoach_original['Fecha'] = pd.to_datetime(self.df_mediacoach_original['Fecha'])
            self.df_mediacoach_original['Jornada_num'] = self.df_mediacoach_original['Jornada'].apply(
                lambda x: int(str(x).lower().replace('j', '')) if pd.notna(x) else 0
            )
            print("✅ Datos de Mediacoach cargados y preparados en memoria.")
        except Exception as e:
            print(f"❌ Error crítico: No se pudieron cargar los datos de Mediacoach: {e}")
            sys.exit(1)

    def similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def normalize_text(self, text):
        if not isinstance(text, str): return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_team_logo(self, equipo):
        """
        Carga el escudo del equipo con una potente lógica de búsqueda jerárquica.
        1. Mapeo Manual: Anula todo lo demás para casos conflictivos.
        2. Coincidencia Exacta: Busca una correspondencia perfecta entre nombres normalizados.
        3. Coincidencia de Palabra Larga: Busca si una palabra significativa (>4 letras) coincide.
        4. Similitud: Como último recurso, busca nombres con una alta similitud de texto.
        """
        escudos_dir = "assets/escudos"
        if not os.path.exists(escudos_dir):
            print(f"Directorio de escudos no encontrado: {escudos_dir}")
            return None

        # --- Nivel 1: MAPEO MANUAL (Máxima Prioridad) ---
        # Edita esta sección para forzar la correspondencia de equipos problemáticos.
        # ¡Asegúrate de que este diccionario esté igual en todos tus scripts!
        TEAM_LOGO_MAP = {
            self.normalize_text("Athletic Club"): "Athletic",
            self.normalize_text("Atletico de Madrid"): "Atlético",
            # self.normalize_text("Real Betis Balompie"): "betis", # Ejemplo
        }
        
        equipo_norm = self.normalize_text(equipo)
        if equipo_norm in TEAM_LOGO_MAP:
            logo_filename = TEAM_LOGO_MAP[equipo_norm]
            for ext in ['.png', '.jpg', '.jpeg']:
                logo_path = os.path.join(escudos_dir, f"{logo_filename}{ext}")
                if os.path.exists(logo_path):
                    print(f"✅ Escudo encontrado por Mapeo Manual: {logo_path}")
                    try:
                        return plt.imread(logo_path)
                    except Exception as e:
                        print(f"Error al cargar escudo mapeado: {e}")
            print(f"⚠️ Advertencia: El archivo mapeado '{logo_filename}' no fue encontrado.")

        # --- Búsqueda Automática ---
        available_files = [f for f in os.listdir(escudos_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # --- Nivel 2: COINCIDENCIA EXACTA ---
        for filename in available_files:
            file_base_norm = self.normalize_text(os.path.splitext(filename)[0])
            if file_base_norm == equipo_norm:
                logo_path = os.path.join(escudos_dir, filename)
                print(f"✅ Escudo encontrado por Coincidencia Exacta: {logo_path}")
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    print(f"Error al cargar escudo por coincidencia exacta: {e}")

        # --- Nivel 3: COINCIDENCIA DE PALABRA LARGA ---
        MIN_WORD_LENGTH = 4 # Busca palabras con 5 o más letras
        team_long_words = {word for word in equipo_norm.split() if len(word) > MIN_WORD_LENGTH}

        if team_long_words:
            # Crea un diccionario de búsqueda de archivos normalizados
            normalized_files = {self.normalize_text(os.path.splitext(f)[0]): f for f in available_files}

            for file_norm, original_filename in normalized_files.items():
                file_words = set(file_norm.split())
                
                # Comprueba si alguna palabra larga del equipo está en las palabras del nombre del archivo
                if not team_long_words.isdisjoint(file_words):
                    logo_path = os.path.join(escudos_dir, original_filename)
                    print(f"✅ Escudo encontrado por Palabra Larga Común ({team_long_words.intersection(file_words)}): {logo_path}")
                    try:
                        return plt.imread(logo_path)
                    except Exception as e:
                        print(f"Error al cargar escudo por palabra larga: {e}")

        # --- Nivel 4: BÚSQUEDA POR SIMILITUD (Último Recurso) ---
        best_match_file = None
        best_similarity = 0.88  # Umbral alto para evitar errores

        for filename in available_files:
            file_base_norm = self.normalize_text(os.path.splitext(filename)[0])
            similarity = self.similarity(equipo_norm, file_base_norm)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_file = filename
        
        if best_match_file:
            logo_path = os.path.join(escudos_dir, best_match_file)
            print(f"✅ Escudo encontrado por Similitud (score: {best_similarity:.2f}): {logo_path}")
            try:
                return plt.imread(logo_path)
            except Exception as e:
                print(f"Error al cargar escudo por similitud: {e}")

        print(f"❌ No se encontró un escudo definitivo para: {equipo} (normalizado como: {equipo_norm})")
        return None

    def unificar_equipos(self, lista_equipos):
        team_mapping = {}
        processed = set()
        for team in lista_equipos:
            if team in processed: continue
            similar = [t for t in lista_equipos if self.similarity(t, team) > 0.85]
            canonical = max(similar, key=len)
            for s in similar:
                team_mapping[s] = canonical
                processed.add(s)
        lista_limpia = sorted(list(set(team_mapping.values())))
        print(f"✅ Se han unificado {len(lista_equipos)} nombres en {len(lista_limpia)} equipos únicos.")
        return lista_limpia, team_mapping

    def recopilar_parametros(self):
        print("\n" + "="*60 + "\n🎯 CONFIGURACIÓN DE REPORTES FÍSICOS\n" + "="*60)
        
        equipos_raw = sorted(self.df_mediacoach_original['Equipo'].unique())
        self.equipos_unificados, self.mapeo_equipos = self.unificar_equipos(equipos_raw)

        if len(sys.argv) > 3:
            try:
                # Dash pasa el NOMBRE del equipo + rango de jornadas
                nombre_recibido = sys.argv[1]
                jornada_inicio = int(sys.argv[2])
                jornada_fin = int(sys.argv[3])
                
                # Buscar equipo por nombre
                equipo_encontrado = None
                for eq in self.equipos_unificados:
                    if nombre_recibido.lower() == eq.lower():
                        equipo_encontrado = eq
                        break
                
                if not equipo_encontrado:
                    print(f"❌ Error: No se encontró el equipo '{nombre_recibido}'")
                    return False
                
                self.equipo_principal = equipo_encontrado
                self.jornada_max_seleccionada = jornada_fin
                self.jornada_inicio = jornada_inicio

                
                # Filtrar por RANGO de jornadas
                self.df_mediacoach_original = self.df_mediacoach_original[
                    (self.df_mediacoach_original['Jornada_num'] >= jornada_inicio) &
                    (self.df_mediacoach_original['Jornada_num'] <= jornada_fin)
                ].copy()
                
                # Actualizamos la fecha límite a la del último partido disponible tras el filtrado
                self.fecha_limite = self.df_mediacoach_original['Fecha'].max()
                self.tipo_partido = None 
                
                print(f"✂️ Filtrando datos de jornada {jornada_inicio} a {jornada_fin}")
                print(f"✅ Dash detectado: {self.equipo_principal} (Jornadas {jornada_inicio}-{jornada_fin})")
                
            except Exception as e:
                print(f"❌ Error filtrando jornadas en Físico: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            # Lógica manual (input consola)
            for i, equipo in enumerate(self.equipos_unificados, 1): print(f"  {i:2d}. {equipo}")
            sel = input(f"\n➤ Selecciona el EQUIPO PRINCIPAL: ").strip()
            self.equipo_principal = self.equipos_unificados[int(sel) - 1]
            fechas_disp = sorted(self.df_mediacoach_original['Fecha'].dt.date.unique(), reverse=True)
            for i, fecha in enumerate(fechas_disp[:10], 1): print(f"  {i:2d}. {fecha}")
            sel_f = input(f"➤ Selecciona la FECHA LÍMITE: ").strip()
            self.fecha_limite = pd.to_datetime(fechas_disp[int(sel_f) - 1])
            self.tipo_partido = None

        self.preparar_dataframes_filtrados()
        return True

    def preparar_dataframes_filtrados(self):
        df_hasta_fecha = self.df_mediacoach_original[self.df_mediacoach_original['Fecha'] <= self.fecha_limite].copy()
        nombres_orig_principal = [k for k, v in self.mapeo_equipos.items() if v == self.equipo_principal]
        df_equipo_principal = df_hasta_fecha[df_hasta_fecha['Equipo'].isin(nombres_orig_principal)].copy()
        self.df_principal_temporada_completa = df_equipo_principal
        ultimas_5_jornadas_num = sorted(df_equipo_principal['Jornada_num'].unique())[-5:]
        self.df_principal_ultimas_5 = df_equipo_principal[df_equipo_principal['Jornada_num'].isin(ultimas_5_jornadas_num)].copy()
        self.jornadas_a_pasar = [f"J{j}" for j in ultimas_5_jornadas_num]
        self.jornada_num_ref = ultimas_5_jornadas_num[-1] if ultimas_5_jornadas_num else 0
        self.jornada_str_ref = f"J{self.jornada_num_ref}"
        print(f"📊 Datos del equipo principal: {len(self.df_principal_temporada_completa)} filas hasta la fecha.")
        print(f"📋 Últimas 5 jornadas para {self.equipo_principal}: {self.jornadas_a_pasar}")
        df_villarreal = df_hasta_fecha[df_hasta_fecha['Equipo'] == 'Villarreal CF'].copy()
        ultimas_5_vcf_num = sorted(df_villarreal['Jornada_num'].unique())[-5:]
        self.df_villarreal_ultimas_5 = df_villarreal[df_villarreal['Jornada_num'].isin(ultimas_5_vcf_num)].copy()
        self.jornadas_vcf_a_pasar = [f"J{j}" for j in ultimas_5_vcf_num]
        self.df_comparativo_principal_vs_vcf = pd.concat([self.df_principal_temporada_completa, df_villarreal]).drop_duplicates().copy()
        print(f"📊 Datos para comparativas (temporada completa vs VCF): {len(self.df_comparativo_principal_vs_vcf)} filas.")
        self.df_comparativo_sprints_ultimas_5 = pd.concat([self.df_principal_ultimas_5, self.df_villarreal_ultimas_5]).drop_duplicates().copy()
        print(f"📊 Datos para comparativa de sprints ('ultimas_5' vs VCF): {len(self.df_comparativo_sprints_ultimas_5)} filas.")
        # Liberar DataFrame intermedio que ya no se necesita
        del df_hasta_fecha
        gc.collect()

    def generar_portada(self):
        print("🎨 Generando portada...")
        fig = plt.figure(figsize=(11.69, 8.27)) # A4 Landscape
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1], zorder=1)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
        try:
            fondo_img = plt.imread("assets/fondo_informes_titulos.png")
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=0)
            ax_bg.imshow(fondo_img, aspect='auto', extent=[0, 100, 0, 100]); ax_bg.axis('off')
        except Exception: print("⚠️ No se encontró fondo para la portada.")
        logo_img = self.load_team_logo(self.equipo_principal)
        if logo_img is not None:
            try:
                imagebox = OffsetImage(logo_img, zoom=0.8)
                ab = AnnotationBbox(imagebox, (50, 62), frameon=False, pad=0)
                ax.add_artist(ab)
            except Exception as e: print(f"❌ Error al colocar el escudo en la portada: {e}")
        ax.text(50, 92, 'INFORME DE RENDIMIENTO FÍSICO', fontsize=40, weight='bold', ha='center', color='#1e3d59', family='serif', path_effects=[patheffects.withStroke(linewidth=4, foreground='white', alpha=0.8)])
        ax.text(50, 45, self.equipo_principal.upper(), fontsize=60, weight='bold', ha='center', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='#1e3d59'), patheffects.SimplePatchShadow(offset=(3, -3), shadow_rgbFace='black', alpha=0.5)])
        ax.text(50, 15, f"Análisis hasta la fecha: {self.fecha_limite.strftime('%d-%m-%Y')}", fontsize=16, ha='center', color='#34495e', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        return fig

    def _read_parquet_simulado(self, path, **kwargs):
        if os.path.normpath(self.main_data_path) in os.path.normpath(path):
            print(f"✔️ Interceptando lectura de '{os.path.basename(path)}'. Devolviendo datos pre-filtrados en memoria.")
            return self.df_para_script
        else:
            print(f"✔️ Lectura normal permitida para: '{os.path.basename(path)}'")
            return self._read_parquet_original(path, **kwargs)

    def _ejecutar_script_subprocess(self, script_name, config, temp_pdf_path, j_inicio, j_fin):
        """
        Ejecuta un script fisico individual en un subprocess aislado.

        Cuando el subprocess termina, el SO recupera toda la RAM utilizada.

        Args:
            script_name (str): Nombre del módulo (sin .py)
            config (dict): Configuración del script desde SCRIPT_CONFIG
            temp_pdf_path (str): Ruta donde el subprocess guardará el PDF
            j_inicio (int): Jornada inicial
            j_fin (int): Jornada final

        Returns:
            bool: True si se generó el PDF correctamente
        """
        # Construir los argumentos de la función
        func_args = {'mostrar': False, 'guardar': False}
        for func_param_name, master_var_name in config['params_map'].items():
            value = getattr(self, master_var_name)
            func_args[func_param_name] = value

        # Escribir argumentos en fichero temporal (evita problemas de escape)
        args_file = temp_pdf_path + ".args.json"

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(args_file, 'w', encoding='utf-8') as f:
            json.dump(func_args, f, ensure_ascii=False, cls=_NumpyEncoder)

        injected_code = textwrap.dedent(f"""\
            import matplotlib, pandas as pd, sys, json, os
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Monkey-patch pd.read_parquet con DuckDB pushdown de predicados
            _j0, _j1 = {j_inicio}, {j_fin}
            _orig_rp = pd.read_parquet
            def _r(path, *a, **kw):
                try:
                    import duckdb as _ddb
                    if not (isinstance(path, str) and path.endswith('.parquet')):
                        return _orig_rp(path, *a, **kw)
                    _con = _ddb.connect()
                    _safe = str(path).replace("'", "\\'")
                    _info = _con.execute("DESCRIBE SELECT * FROM read_parquet('" + _safe + "') LIMIT 0").df()
                    _jrow = [(r['column_name'], r['column_type']) for _, r in _info.iterrows() if any(_x in r['column_name'].lower() for _x in ['jornada','week','matchday','semana'])]
                    _jcol = _jrow[0][0] if _jrow else None
                    _jtype = _jrow[0][1] if _jrow else ''
                    if _jcol:
                        if any(_t in _jtype.upper() for _t in ['INT','BIGINT','SMALLINT','HUGEINT','DOUBLE','FLOAT','DECIMAL']):
                            _sql = "SELECT * FROM read_parquet(?) WHERE " + _jcol + " BETWEEN ? AND ?"
                        else:
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
                    if any(x in c.lower() for x in ['jornada', 'week', 'matchday']):
                        try:
                            s = df[c].astype(str).str.lower().str.replace('j', '').str.strip()
                            v = pd.to_numeric(s, errors='coerce')
                            if v.notna().any():
                                df = df[(v >= _j0) & (v <= _j1)]
                                break
                        except:
                            pass
                return df
            pd.read_parquet = _r

            # Leer argumentos del fichero temporal
            with open('{args_file}', 'r', encoding='utf-8') as f:
                args = json.load(f)

            # Importar el módulo y ejecutar la función
            import importlib
            module = importlib.import_module('{script_name}')
            func = getattr(module, '{config["func"]}')
            fig = func(**args)

            if fig and isinstance(fig, plt.Figure):
                fig.set_size_inches(11.69, 8.27, forward=True)
                try:
                    fig.tight_layout(pad=0.5)
                except:
                    pass
                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages('{temp_pdf_path}') as pdf:
                    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                print("PDF_SAVED_OK")
            else:
                print("NO_FIGURE_RETURNED")
        """)

        try:
            proceso = subprocess.Popen(
                [sys.executable, "-u", "-c", injected_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            stdout, _ = proceso.communicate(timeout=300)

            if stdout:
                lineas = stdout.strip().split('\n')
                for linea in lineas[-5:]:
                    print(f"   {linea}")

            if proceso.returncode != 0:
                print(f"   ⚠️ {script_name} terminó con código {proceso.returncode}")

            return os.path.exists(temp_pdf_path)

        except subprocess.TimeoutExpired:
            print(f"   ❌ Timeout ejecutando {script_name} (>300s)")
            proceso.kill()
            proceso.wait()
            return False
        except Exception as e:
            print(f"   ❌ Error ejecutando {script_name}: {e}")
            return False

    def ejecutar_todos(self):
        if not self.recopilar_parametros():
            return

        j_inicio = getattr(self, 'jornada_inicio', 1)
        j_fin = getattr(self, 'jornada_max_seleccionada', 99)

        output_pdf_path = f"Informe_Fisico_{self.equipo_principal.replace(' ', '_')}_J{j_inicio}_J{j_fin}.pdf"

        # Crear directorio temporal para PDFs intermedios
        temp_dir = tempfile.mkdtemp(prefix="fisico_")

        try:
            # 1. Generar portada en el proceso master (ligera, no necesita subprocess)
            portada_path = os.path.join(temp_dir, "00_portada.pdf")
            fig_portada = self.generar_portada()
            with PdfPages(portada_path) as pdf:
                pdf.savefig(fig_portada, bbox_inches='tight', pad_inches=0)
            plt.close(fig_portada)
            del fig_portada
            gc.collect()

            print("\n" + "="*60 + f"\n🚀 INICIANDO EJECUCIÓN EN MODO SUBPROCESS (J{j_inicio}-J{j_fin})\n" + "="*60)

            # 2. Ejecutar cada script en un subprocess aislado
            pdf_parciales = [portada_path]

            for i, (script_name, config) in enumerate(SCRIPT_CONFIG.items(), 1):
                script_base_name = script_name.replace('.py', '')
                print(f"\n--- [{i}/{len(SCRIPT_CONFIG)}] Ejecutando: {script_base_name} ---")

                temp_pdf = os.path.join(temp_dir, f"{i:02d}_{script_base_name}.pdf")
                exito = self._ejecutar_script_subprocess(
                    script_base_name, config, temp_pdf, j_inicio, j_fin
                )

                if exito:
                    pdf_parciales.append(temp_pdf)
                    print(f"✅ Página añadida correctamente.")
                else:
                    print(f"⚠️  El script no generó una figura válida.")

            # 3. Fusionar todos los PDFs parciales en el PDF final
            print(f"\n📄 Fusionando {len(pdf_parciales)} páginas...")
            writer = PdfWriter()
            for pdf_path in pdf_parciales:
                try:
                    reader = PdfReader(pdf_path)
                    for page in reader.pages:
                        writer.add_page(page)
                except Exception as e:
                    print(f"   ⚠️ Error leyendo {os.path.basename(pdf_path)}: {e}")

            with open(output_pdf_path, 'wb') as f:
                writer.write(f)

            print("\n" + "="*60 + f"\n🎉 PROCESO FINALIZADO\n✅ Informe guardado como: {output_pdf_path}\n" + "="*60)

        finally:
            # Limpiar archivos temporales
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

if __name__ == "__main__":
    generador = GeneradorMaestro()
    generador.ejecutar_todos()