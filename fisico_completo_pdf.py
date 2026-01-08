#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as patheffects
import importlib
import traceback
from datetime import datetime
from difflib import SequenceMatcher
import unicodedata
import re
import warnings

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
        df_hasta_fecha = self.df_mediacoach_original[self.df_mediacoach_original['Fecha'] <= self.fecha_limite]
        nombres_orig_principal = [k for k, v in self.mapeo_equipos.items() if v == self.equipo_principal]
        df_equipo_principal = df_hasta_fecha[df_hasta_fecha['Equipo'].isin(nombres_orig_principal)]
        self.df_principal_temporada_completa = df_equipo_principal
        ultimas_5_jornadas_num = sorted(df_equipo_principal['Jornada_num'].unique())[-5:]
        self.df_principal_ultimas_5 = df_equipo_principal[df_equipo_principal['Jornada_num'].isin(ultimas_5_jornadas_num)]
        self.jornadas_a_pasar = [f"J{j}" for j in ultimas_5_jornadas_num]
        self.jornada_num_ref = ultimas_5_jornadas_num[-1] if ultimas_5_jornadas_num else 0
        self.jornada_str_ref = f"J{self.jornada_num_ref}"
        print(f"📊 Datos del equipo principal: {len(self.df_principal_temporada_completa)} filas hasta la fecha.")
        print(f"📋 Últimas 5 jornadas para {self.equipo_principal}: {self.jornadas_a_pasar}")
        df_villarreal = df_hasta_fecha[df_hasta_fecha['Equipo'] == 'Villarreal CF']
        ultimas_5_vcf_num = sorted(df_villarreal['Jornada_num'].unique())[-5:]
        self.df_villarreal_ultimas_5 = df_villarreal[df_villarreal['Jornada_num'].isin(ultimas_5_vcf_num)]
        self.jornadas_vcf_a_pasar = [f"J{j}" for j in ultimas_5_vcf_num]
        self.df_comparativo_principal_vs_vcf = pd.concat([self.df_principal_temporada_completa, df_villarreal]).drop_duplicates()
        print(f"📊 Datos para comparativas (temporada completa vs VCF): {len(self.df_comparativo_principal_vs_vcf)} filas.")
        # 🔥 NUEVO: Crear un DataFrame combinado para la comparativa de sprints (ultimas 5 jornadas)
        self.df_comparativo_sprints_ultimas_5 = pd.concat([self.df_principal_ultimas_5, self.df_villarreal_ultimas_5]).drop_duplicates()
        print(f"📊 Datos para comparativa de sprints ('ultimas_5' vs VCF): {len(self.df_comparativo_sprints_ultimas_5)} filas.")

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

    def ejecutar_todos(self):
        if not self.recopilar_parametros(): 
            return

        # 1. Definir la jornada tope (capturada en recopilar_parametros desde Dash)
        # Si no existe (ejecución manual), usamos 99 para no filtrar nada.
        limit_j = getattr(self, 'jornada_max_seleccionada', 99)

        # 2. Guardar la función original de pandas para poder usarla dentro del parche
        _original_pd_read = pd.read_parquet

        # 3. Definir el Parche de Fuerza Bruta para filtrar CUALQUIER parquet que se abra
        def _patched_read_filtered(path, **kwargs):
            # Lógica de intercepción de datos pre-filtrados en memoria (lo que ya tenías para rendimiento_fisico)
            if os.path.normpath(self.main_data_path) in os.path.normpath(path):
                print(f"✔️  [PARCHE] Interceptando lectura de '{os.path.basename(path)}'. Usando datos de memoria.")
                df = self.df_para_script
            else:
                # Lectura normal del disco para otros archivos (como player_stats, etc)
                df = _original_pd_read(path, **kwargs)

            # --- APLICAR TOPE DE JORNADA A CUALQUIER COLUMNA QUE PAREZCA UNA JORNADA ---
            # Buscamos en todas las columnas nombres como 'jornada', 'week', 'match_week', etc.
            patterns = ['jornada', 'week', 'matchday', 'match_week', 'matchWeek']
            for col in df.columns:
                if any(p in col.lower() for p in patterns):
                    try:
                        # Limpiamos el rastro de letras 'J' o 'j' (ej: 'J14' -> '14')
                        s_vals = df[col].astype(str).str.lower().str.replace('j', '').str.strip()
                        # Convertimos a número
                        n_vals = pd.to_numeric(s_vals, errors='coerce')

                        # Si la columna realmente contiene números de jornada
                        if n_vals.notna().any():
                            # BORRADO PROVISIONAL: Solo mantenemos filas donde jornada <= tope
                            df = df[n_vals <= limit_j]
                    except:
                        pass
            return df

        # 4. Activar el parche globalmente sustituyendo la función de pandas
        pd.read_parquet = _patched_read_filtered

        # 5. Configurar el archivo de salida
        output_pdf_path = f"Informe_Físico_{self.equipo_principal.replace(' ', '_')}.pdf"
        
        try:
            with PdfPages(output_pdf_path) as pdf:
                # Generar y añadir la portada
                pdf.savefig(self.generar_portada(), bbox_inches='tight', pad_inches=0)
                plt.close('all')
                
                print("\n" + "="*60 + f"\n🚀 INICIANDO EJECUCIÓN (TOPE: J{limit_j})\n" + "="*60)
                
                # Ejecutar cada script de la configuración
                for i, (script_name, config) in enumerate(SCRIPT_CONFIG.items(), 1):
                    script_base_name = script_name.replace('.py', '')
                    print(f"\n--- [{i}/{len(SCRIPT_CONFIG)}] Ejecutando: {script_base_name} ---")
                    
                    try:
                        # Seleccionar el DataFrame correcto según el tipo de datos del script
                        if config['tipo_datos'] == 'temporada_completa_comparativa':
                            self.df_para_script = self.df_comparativo_principal_vs_vcf
                        elif config['tipo_datos'] == 'ultimas_5_comparativa':
                            self.df_para_script = self.df_comparativo_sprints_ultimas_5
                        elif config['tipo_datos'] == 'ultimas_5':
                            self.df_para_script = self.df_principal_ultimas_5
                            if config.get('equipo_fijo') == 'Villarreal CF':
                                self.df_para_script = self.df_villarreal_ultimas_5
                        else:
                            self.df_para_script = self.df_principal_temporada_completa

                        # Construir los argumentos para la función del script (mostar/guardar siempre False)
                        func_args = {'mostrar': False, 'guardar': False}
                        for func_param_name, master_var_name in config['params_map'].items():
                            func_args[func_param_name] = getattr(self, master_var_name)

                        # Importar el módulo del script dinámicamente y ejecutar su función
                        module = importlib.import_module(script_base_name)
                        importlib.reload(module)
                        generar_func = getattr(module, config['func'])
                        
                        figura_reporte = generar_func(**func_args)

                        # Si el script devuelve una figura de Matplotlib, la añadimos al PDF
                        if figura_reporte and isinstance(figura_reporte, plt.Figure):
                            # Forzar tamaño A4 horizontal
                            figura_reporte.set_size_inches(11.69, 8.27, forward=True)
                            try:
                                figura_reporte.tight_layout(pad=0.5)
                            except:
                                pass
                            
                            pdf.savefig(figura_reporte, bbox_inches='tight', pad_inches=0.1)
                            print(f"✅ Página añadida correctamente.")
                        else:
                            print(f"⚠️  El script no devolvió una figura válida.")
                    
                    except Exception as e:
                        print(f"❌ ERROR GRAVE EN {script_base_name}: {e}")
                        traceback.print_exc()
                    
                    finally:
                        plt.close('all')

            print("\n" + "="*60 + f"\n🎉 PROCESO FINALIZADO\n✅ Informe guardado como: {output_pdf_path}\n" + "="*60)

        finally:
            # 6. RESTAURAR PANDAS: Muy importante devolver pd.read_parquet a su estado original
            pd.read_parquet = _original_pd_read

        print("\n" + "="*60 + "\n🎉 PROCESO FINALIZADO\n" + f"✅ Informe maestro guardado como: {output_pdf_path}\n" + "="*60)

if __name__ == "__main__":
    generador = GeneradorMaestro()
    generador.ejecutar_todos()