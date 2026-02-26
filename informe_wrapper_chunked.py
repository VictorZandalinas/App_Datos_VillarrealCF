#!/usr/bin/env python3
"""
WRAPPER DE GENERACIÓN DE INFORMES POR CHUNKS
Ejecuta scripts de análisis en grupos pequeños con limpieza de memoria entre chunks
"""

import os
import sys
import re
import subprocess
import textwrap
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from memory_utils import clean_memory_agresivo, get_memory_info, check_memory_threshold, clear_all_script_caches
from pdf_fusion import fusionar_pdfs_incremental
from memory_monitor import MemoryMonitor

# Funciones auxiliares para portadas
def generar_portada(tipo_informe, equipo, jornada, output_path):
    """
    Genera la portada del informe según el tipo.

    Args:
        tipo_informe (str): 'ABP', 'TACTIC', etc.
        equipo (str): Nombre del equipo
        jornada (int): Número de jornada
        output_path (Path): Ruta donde guardar la portada

    Returns:
        bool: True si se generó correctamente, False si hubo error
    """
    try:
        from matplotlib import patheffects
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from difflib import SequenceMatcher

        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        efecto_relieve = [
            patheffects.withStroke(linewidth=4, foreground='white', alpha=0.9),
            patheffects.Normal()
        ]

        # Determinar imagen de fondo según tipo
        if tipo_informe in ['TACTIC', 'TACTICO']:
            ruta_portada = "assets/portada_tactico_informe.png"
            titulo = 'INFORME SITUACIONES DE JUEGO'
        elif tipo_informe == 'ABP':
            ruta_portada = "assets/portada_abp_informe.png"
            titulo = 'INFORME BALÓN PARADO'
        else:
            ruta_portada = None
            titulo = f'INFORME {tipo_informe}'

        # Fondo
        if ruta_portada and os.path.exists(ruta_portada):
            try:
                img_fondo = plt.imread(ruta_portada)
                ax.imshow(img_fondo, aspect='auto', extent=[0, 1, 0, 1])
            except:
                ax.set_facecolor('#1e3d59')
        else:
            ax.set_facecolor('#1e3d59')

        # Título
        ax.text(0.5, 0.92, titulo, ha='center', va='center',
                fontsize=42, fontweight='bold', color='#1e3d59',
                family='serif', path_effects=efecto_relieve)

        # Nombre del equipo
        ax.text(0.20, 0.76, equipo.upper(), ha='center', va='center',
                fontsize=32, fontweight='bold', color='#e74c3c',
                family='serif', path_effects=efecto_relieve)

        # Escudo del equipo
        if os.path.exists('assets/escudos'):
            equipo_clean = equipo.lower().replace(' ', '').replace('cf', '').replace('fc', '')
            best_match, best_sim = None, 0

            for f in os.listdir('assets/escudos'):
                if not f.endswith('.png'):
                    continue
                name = f.replace('.png', '').lower().replace('_', '').replace('cf', '').replace('fc', '')
                sim = SequenceMatcher(None, equipo_clean, name).ratio()
                if sim > best_sim and sim > 0.4:
                    best_sim = sim
                    best_match = f

            if best_match:
                try:
                    logo_img = plt.imread(f"assets/escudos/{best_match}")
                    imagebox = OffsetImage(logo_img, zoom=0.6)
                    ab = AnnotationBbox(imagebox, (0.20, 0.58), frameon=False)
                    ax.add_artist(ab)
                except:
                    pass

        # Jornada
        ax.text(0.5, 0.18, f'Jornada: {jornada}', ha='center', va='center',
                fontsize=16, color='#34495e', family='serif',
                path_effects=efecto_relieve)

        # Guardar
        fig.savefig(str(output_path), format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return True

    except Exception as e:
        print(f"❌ Error generando portada: {e}")
        return False


class InformeGeneratorChunked:
    """
    Generador de informes con procesamiento por chunks.

    Divide la ejecución de scripts en grupos pequeños (chunks) y libera
    memoria agresivamente entre chunks para evitar OOM en servidores limitados.
    """

    # Scripts que necesitan datos de TODA la liga (carpeta general, no equipo específico)
    SCRIPTS_LIGA = [
        'tactic1_opta_clasificacion_liga.py',
        'tactic1.1_mediacoach_resumen_con_balon.py',
        'tactic1.2_mediacoach_resumen_sin_balon.py',
        'tactic1.3_mediacoach_evolucion_resumen_general.py',
        'tactic1.4_opta_xT.py',
    ]

    def __init__(self, tipo_informe, chunk_size=6, team_mappings=None, equipos_opta=None, equipos_mediacoach=None):
        """
        Inicializa el generador de informes.

        Args:
            tipo_informe (str): 'ABP' o 'TACTIC'
            chunk_size (int): Número de scripts por chunk (6-8 recomendado para 4GB RAM)
            team_mappings (dict): Mapeo de nombres de equipos entre fuentes
            equipos_opta (list): Lista de equipos en orden de Opta
            equipos_mediacoach (list): Lista de equipos en orden de MediaCoach
        """
        self.tipo = tipo_informe.upper()
        self.chunk_size = chunk_size
        self.team_mappings = team_mappings or {}
        self.equipos_opta = equipos_opta or []
        self.equipos_mediacoach = equipos_mediacoach or []
        self.scripts = self._discover_scripts()
        self.chunks = self._create_chunks()
        self.temp_dir = Path("reportes_temporales")

        # Crear directorio temporal
        if not self.temp_dir.exists():
            self.temp_dir.mkdir()

        print(f"🔧 InformeGeneratorChunked inicializado:")
        print(f"   Tipo: {self.tipo}")
        print(f"   Scripts encontrados: {len(self.scripts)}")
        print(f"   Chunks: {len(self.chunks)} (tamaño: {self.chunk_size})")

    def _discover_scripts(self):
        """
        Descubre scripts del tipo de informe especificado.

        Returns:
            list: Lista ordenada de nombres de scripts
        """
        # Patrón para buscar scripts (abp1, abp2.1, tactic1, etc.)
        patron = re.compile(f'^{self.tipo.lower()}[0-9].*\\.py$', re.IGNORECASE)

        scripts = [
            f for f in os.listdir('.')
            if os.path.isfile(f) and patron.match(f)
        ]

        # Ordenar naturalmente (abp1, abp2, abp2.1, abp3, ...)
        scripts.sort(key=self._natural_sort_key)

        return scripts

    def _natural_sort_key(self, s):
        """
        Clave de ordenamiento natural para scripts.
        Maneja correctamente: abp1, abp2, abp2.1, abp3, abp10

        Args:
            s (str): Nombre del archivo

        Returns:
            list: Lista de componentes para ordenamiento
        """
        filename = os.path.basename(s)
        parts = re.split(r'(\d+\.?\d*)', filename)

        key_parts = []
        for part in parts:
            if not part:
                continue
            # Si es número, convertir a float
            if re.match(r'^\d+\.?\d*$', part):
                key_parts.append(float(part))
            else:
                key_parts.append(part.lower())

        return key_parts

    def _create_chunks(self):
        """
        Divide scripts en chunks de tamaño chunk_size.

        Returns:
            list: Lista de listas, cada una conteniendo scripts del chunk
        """
        chunks = []
        for i in range(0, len(self.scripts), self.chunk_size):
            chunk = self.scripts[i:i+self.chunk_size]
            chunks.append(chunk)

        return chunks

    def ejecutar(self, equipo, j_inicio, j_fin):
        """
        Ejecuta todos los chunks con limpieza de memoria entre ellos.

        Args:
            equipo (str): Nombre canónico del equipo
            j_inicio (int): Jornada inicial
            j_fin (int): Jornada final

        Returns:
            Path: Ruta al PDF generado, o None si error
        """
        pdfs_generados = []
        monitor = MemoryMonitor(threshold_mb=2500, interval=5)
        monitor.start()

        try:
            # Generar portada primero
            print(f"\n{'='*70}")
            print("🎨 Generando portada...")
            print(f"{'='*70}")

            portada_path = self.temp_dir / "00_portada.pdf"
            if generar_portada(self.tipo, equipo, j_fin, portada_path):
                pdfs_generados.append(portada_path)
                print("✅ Portada generada")
            else:
                print("⚠️ No se pudo generar portada, continuando sin ella...")

            # Ejecutar cada chunk
            for chunk_id, chunk_scripts in enumerate(self.chunks, 1):
                print(f"\n{'='*70}")
                print(f"🔄 [CHUNK {chunk_id}/{len(self.chunks)}] Procesando {len(chunk_scripts)} scripts")
                print(f"{'='*70}")

                # Ejecutar scripts del chunk
                chunk_pdfs = self._ejecutar_chunk(
                    chunk_scripts,
                    equipo,
                    j_inicio,
                    j_fin,
                    chunk_id
                )

                pdfs_generados.extend(chunk_pdfs)

                # Limpieza agresiva entre chunks
                mem_antes = get_memory_info()
                print(f"\n🧹 Liberando memoria post-chunk...")

                # Limpiar cachés de clase de scripts de análisis
                clear_all_script_caches()

                check_memory_threshold(threshold_mb=2500, auto_clean=True)
                clean_memory_agresivo()

                mem_despues = get_memory_info()
                if mem_despues['available']:
                    freed = mem_antes['rss_mb'] - mem_despues['rss_mb']
                    print(f"📊 [MEMORIA] Chunk {chunk_id} completado: {mem_despues['rss_mb']:.0f}MB "
                          f"({mem_despues['percent']:.1f}% | Liberados: {freed:.0f}MB)")

            # Fusión incremental final
            print(f"\n{'='*70}")
            print("🔄 FASE FINAL: Fusionando PDFs...")
            print(f"   Total de PDFs a fusionar: {len(pdfs_generados)}")
            print(f"{'='*70}")

            if len(pdfs_generados) == 0:
                print("❌ No se generaron PDFs para fusionar")
                return None

            output_path = self._generar_nombre_output(equipo, j_inicio, j_fin)
            print(f"📄 Nombre del archivo final: {output_path}")
            print(f"📂 Directorio de trabajo: {os.getcwd()}")

            resultado = fusionar_pdfs_incremental(pdfs_generados, output_path, self.temp_dir)

            if resultado and resultado.exists():
                print(f"✅ PDF final generado: {resultado}")
                print(f"   Tamaño: {resultado.stat().st_size / 1024 / 1024:.2f} MB")
                print(f"   Ruta absoluta: {resultado.absolute()}")
            else:
                print(f"❌ Error: No se pudo generar el PDF final")

            return resultado

        except Exception as e:
            print(f"❌ Error crítico en ejecución: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            monitor.stop()
            stats = monitor.get_stats()
            if stats:
                print(f"\n📊 [RESUMEN MEMORIA]")
                print(f"   Pico: {stats['peak_mb']:.0f}MB")
                print(f"   Alertas: {stats['alert_count']}")

    def _ejecutar_chunk(self, chunk_scripts, equipo, j_inicio, j_fin, chunk_id):
        """
        Ejecuta un chunk de scripts con manejo de errores individual.

        Args:
            chunk_scripts (list): Lista de scripts del chunk
            equipo (str): Nombre del equipo
            j_inicio (int): Jornada inicial
            j_fin (int): Jornada final
            chunk_id (int): ID del chunk (para logging)

        Returns:
            list: Lista de rutas Path a PDFs generados
        """
        chunk_pdfs = []

        for script in chunk_scripts:
            try:
                # Índice global para mantener compatibilidad con app.py
                script_idx = self.scripts.index(script) + 1
                total = len(self.scripts)

                # Formato compatible con regex de app.py: "[i/total] Ejecutando: nombre"
                print(f"\n--- [{script_idx}/{total}] Ejecutando: {script} ---")

                # Ejecutar script individual
                pdf_path = self._ejecutar_script_individual(
                    script,
                    equipo,
                    j_inicio,
                    j_fin
                )

                if pdf_path:
                    chunk_pdfs.append(pdf_path)
                    print(f"✅ PDF generado: {pdf_path.name}")
                else:
                    print(f"⚠️ {script} no generó PDF")

                # Log de memoria si está alta
                mem = get_memory_info()
                if mem['available'] and mem['rss_mb'] > 2000:
                    print(f"📊 [MEMORIA] Tras {script}: {mem['rss_mb']:.0f}MB ({mem['percent']:.1f}%)")

            except MemoryError as e:
                print(f"❌ [CHUNK {chunk_id}] OOM en {script}: {e}")
                clean_memory_agresivo()
                # Continuar con siguiente script

            except Exception as e:
                print(f"⚠️ [CHUNK {chunk_id}] Error en {script}: {e}")
                # Continuar con siguiente script

        return chunk_pdfs

    def _ejecutar_script_individual(self, script, equipo, j_inicio, j_fin):
        """
        Ejecuta un script individual en un subprocess aislado.

        Usa subprocess.Popen en lugar de exec() para que al terminar el
        subproceso, el SO recupere toda la RAM utilizada por el script.

        Args:
            script (str): Nombre del script
            equipo (str): Nombre del equipo
            j_inicio (int): Jornada inicial
            j_fin (int): Jornada final

        Returns:
            Path: Ruta al PDF generado, o None si no se generó
        """
        # Detectar PDFs antes de ejecutar
        pdfs_antes = set(Path('.').glob("*.pdf"))

        # Preparar inputs automáticos
        respuestas = self._preparar_inputs(script, equipo, j_fin)

        # Determinar si es un script de "liga" (datos de toda la liga) o de "equipo"
        is_script_liga = script in self.SCRIPTS_LIGA

        # Código inyectado: monkey-patch pd.read_parquet (DuckDB pushdown) + ejecutar script
        # - Para scripts de equipo: merge rival folder + Villarreal folder
        # - Para scripts de liga: usar carpeta general directamente (sin merge)
        injected_code = textwrap.dedent(f"""\
            import matplotlib, pandas as pd, sys, numpy as np
            import os as _os
            matplotlib.use('Agg')
            _j0, _j1 = {j_inicio}, {j_fin}
            _equipo_key = '{equipo}'
            _is_liga = {is_script_liga}
            def _norm(s):
                for x in [' cf',' fc',' rc',' rcd',' ca',' ud',' ','-']:
                    s = s.lower().replace(x,'')
                return s
            def _find_folder(name):
                _b = 'data_por_equipos'
                if not _os.path.exists(_b): return None
                _t = _norm(name)
                for _d in _os.listdir(_b):
                    if _os.path.isdir(_os.path.join(_b,_d)):
                        _dn = _norm(_d)
                        if _t in _dn or _dn in _t: return _d
                return None
            _equipo_folder = _find_folder(_equipo_key)
            _villa_folder = _find_folder('Villarreal')
            if _villa_folder == _equipo_folder:
                _villa_folder = None
            def _to_path(orig, folder):
                if folder is None: return None
                c = orig[2:] if orig.startswith('./') else orig
                t = _os.path.join('data_por_equipos', folder, c)
                if _os.path.exists(t): return t
                _idx = c.find('/')
                if _idx > 0:
                    t2 = _os.path.join('data_por_equipos', folder, c[_idx+1:])
                    if _os.path.exists(t2): return t2
                return None
            _orig_rp = pd.read_parquet
            def _read_one(path):
                try:
                    import duckdb as _ddb
                    _con = _ddb.connect()
                    _safe = str(path).replace("'", "\\'")
                    _info = _con.execute("DESCRIBE SELECT * FROM read_parquet('" + _safe + "') LIMIT 0").df()
                    _jrow = [(r['column_name'], r['column_type']) for _, r in _info.iterrows() if any(_x in r['column_name'].lower() for _x in ['jornada','week','semana','matchday'])]
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
                df = _orig_rp(path)
                for _c in df.columns:
                    if any(x in _c.lower() for x in ['jornada', 'week', 'semana']):
                        try:
                            s = df[_c].astype(str).str.lower().str.replace('j', '').str.replace('w', '').str.strip()
                            v = pd.to_numeric(s, errors='coerce')
                            if v.notna().any():
                                df = df[(v >= _j0) & (v <= _j1)]
                                break
                        except: pass
                return df
            def _r(path, *a, **kw):
                if not (isinstance(path, str) and path.endswith('.parquet')):
                    return _orig_rp(path, *a, **kw)
                # Para scripts de liga: usar ruta original (carpeta general)
                if _is_liga:
                    return _read_one(path)
                # Para scripts de equipo: redirigir a carpetas de equipo
                p1 = _to_path(path, _equipo_folder)
                p2 = _to_path(path, _villa_folder)
                if p1 and p2:
                    return pd.concat([_read_one(p1), _read_one(p2)], ignore_index=True).drop_duplicates()
                elif p1:
                    return _read_one(p1)
                elif p2:
                    return _read_one(p2)
                else:
                    return _read_one(path)
            pd.read_parquet = _r
            exec(open('{script}', encoding='utf-8').read())
        """)

        try:
            proceso = subprocess.Popen(
                [sys.executable, "-u", "-c", injected_code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            stdout, _ = proceso.communicate(input=respuestas, timeout=300)

            # Mostrar últimas líneas de salida del subprocess
            if stdout:
                lineas = stdout.strip().split('\n')
                # Mostrar últimas 5 líneas para contexto
                for linea in lineas[-5:]:
                    print(f"   {linea}")

            if proceso.returncode != 0:
                print(f"   ⚠️ {script} terminó con código {proceso.returncode}")

        except subprocess.TimeoutExpired:
            print(f"   ❌ Timeout ejecutando {script} (>300s)")
            proceso.kill()
            proceso.wait()
        except Exception as e:
            print(f"   ❌ Error ejecutando {script}: {e}")

        # Detectar PDF generado
        pdfs_despues = set(Path('.').glob("*.pdf"))
        nuevos = pdfs_despues - pdfs_antes

        if nuevos:
            pdf_path = list(nuevos)[0]

            # Mover a temp_dir con índice
            idx = self.scripts.index(script)
            dest = self.temp_dir / f"{idx:02d}_{pdf_path.name}"

            try:
                pdf_path.rename(dest)
                return dest
            except Exception as e:
                print(f"   ⚠️ Error moviendo PDF {pdf_path} a {dest}: {e}")
                return None
        else:
            return None

    def _preparar_inputs(self, script, equipo, jornada):
        """
        Prepara inputs automáticos para el script.

        Args:
            script (str): Nombre del script
            equipo (str): Nombre canónico del equipo (ej: 'Villarreal', 'Barcelona')
            jornada (int): Jornada final

        Returns:
            str: String con inputs simulados (separados por \n)
        """
        # Ajustar para scripts específicos
        if 'mediacoach' in script.lower():
            # MediaCoach: necesita índice del equipo en lista mediacoach
            if self.team_mappings and equipo in self.team_mappings:
                nombre_mc = self.team_mappings[equipo].get('mediacoach', equipo)
                try:
                    idx_mc = self.equipos_mediacoach.index(nombre_mc)
                    respuestas = f"{idx_mc + 1}\n{jornada}\n{jornada}\n"
                except (ValueError, AttributeError):
                    # Fallback: Villarreal por defecto
                    respuestas = f"19\n{jornada}\n{jornada}\n"
            else:
                # Sin mappings, asumir Villarreal
                respuestas = f"19\n{jornada}\n{jornada}\n"

        elif 'sportian' in script.lower():
            # Sportian espera nombre directo
            respuestas = f"{equipo}\n"

        else:
            # Opta: necesita índice del equipo en lista opta
            if self.equipos_opta and equipo in self.equipos_opta:
                try:
                    idx_opta = self.equipos_opta.index(equipo)
                    respuestas = f"{idx_opta + 1}\n{jornada}\n"
                except ValueError:
                    # Fallback
                    respuestas = f"1\n{jornada}\n"
            else:
                # Sin lista, usar índice 1 por defecto
                respuestas = f"1\n{jornada}\n"

        return respuestas

    def _generar_nombre_output(self, equipo, j_inicio, j_fin):
        """
        Genera nombre del archivo de salida.

        Args:
            equipo (str): Nombre del equipo
            j_inicio (int): Jornada inicial
            j_fin (int): Jornada final

        Returns:
            str: Nombre del archivo PDF
        """
        equipo_clean = equipo.replace(' ', '_')

        if self.tipo == 'ABP':
            return f"Informe_ABP_{equipo_clean}_J{j_inicio}_J{j_fin}.pdf"
        elif self.tipo == 'TACTIC' or self.tipo == 'TACTICO':
            return f"Informe_Situaciones_Juego_{equipo_clean}_J{j_inicio}_J{j_fin}.pdf"
        else:
            return f"Informe_{self.tipo}_{equipo_clean}_J{j_inicio}_J{j_fin}.pdf"


# === FUNCIONES DE UTILIDAD ===

def test_chunked_system():
    """Test básico del sistema de chunks"""
    print("=== Test del Sistema de Chunks ===\n")

    # Verificar imports
    print("✅ Imports verificados")

    # Crear instancia de prueba
    try:
        wrapper = InformeGeneratorChunked(tipo_informe='ABP', chunk_size=6)
        print(f"✅ Wrapper creado: {len(wrapper.scripts)} scripts, {len(wrapper.chunks)} chunks")
    except Exception as e:
        print(f"❌ Error creando wrapper: {e}")
        return

    # Mostrar estructura de chunks
    print("\nEstructura de chunks:")
    for i, chunk in enumerate(wrapper.chunks, 1):
        print(f"  Chunk {i}: {len(chunk)} scripts")
        for script in chunk[:2]:  # Mostrar solo primeros 2
            print(f"    - {script}")
        if len(chunk) > 2:
            print(f"    ... y {len(chunk)-2} más")

    print("\n✅ Test básico completado")


if __name__ == "__main__":
    # Ejecutar test si se llama directamente
    test_chunked_system()
