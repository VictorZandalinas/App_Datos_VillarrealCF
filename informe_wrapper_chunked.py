#!/usr/bin/env python3
"""
WRAPPER DE GENERACIÓN DE INFORMES POR CHUNKS
Ejecuta scripts de análisis en grupos pequeños con limpieza de memoria entre chunks
"""

import os
import sys
import io
import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from memory_utils import clean_memory_agresivo, get_memory_info, check_memory_threshold
from pdf_fusion import fusionar_pdfs_incremental
from memory_monitor import MemoryMonitor


class InformeGeneratorChunked:
    """
    Generador de informes con procesamiento por chunks.

    Divide la ejecución de scripts en grupos pequeños (chunks) y libera
    memoria agresivamente entre chunks para evitar OOM en servidores limitados.
    """

    def __init__(self, tipo_informe, chunk_size=6):
        """
        Inicializa el generador de informes.

        Args:
            tipo_informe (str): 'ABP' o 'TACTICO'
            chunk_size (int): Número de scripts por chunk (6-8 recomendado para 4GB RAM)
        """
        self.tipo = tipo_informe.upper()
        self.chunk_size = chunk_size
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
            print(f"{'='*70}")

            output_path = self._generar_nombre_output(equipo, j_inicio, j_fin)
            resultado = fusionar_pdfs_incremental(pdfs_generados, output_path, self.temp_dir)

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
        Ejecuta un script individual usando exec() (modelo de tactic_informe_todo.py).

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
        # Nota: Esta es una versión simplificada - en producción debe usar
        # los mismos mappings y lógica que abp_informe_todo.py
        respuestas = self._preparar_inputs(script, equipo, j_fin)

        # Configurar variables de entorno para filtrado de jornadas
        os.environ[f'{self.tipo}_J_INI'] = str(j_inicio)
        os.environ[f'{self.tipo}_J_FIN'] = str(j_fin)

        # Redirigir stdin
        stdin_original = sys.stdin
        sys.stdin = io.StringIO(respuestas)

        try:
            # Leer código del script
            with open(script, 'r', encoding='utf-8') as f:
                codigo = f.read()

            # Crear scope aislado con pandas y matplotlib
            scope = {
                '__builtins__': __builtins__,
                '__name__': '__main__',
                '__file__': script,
                'pd': pd,
                'plt': plt,
                'sys': sys,
                'os': os
            }

            # Ejecutar script
            exec(codigo, scope, scope)

        finally:
            # Restaurar stdin y limpiar
            sys.stdin = stdin_original
            scope.clear()
            del scope
            clean_memory_agresivo()

        # Detectar PDF generado
        pdfs_despues = set(Path('.').glob("*.pdf"))
        nuevos = pdfs_despues - pdfs_antes

        if nuevos:
            pdf_path = list(nuevos)[0]

            # Mover a temp_dir con índice
            idx = self.scripts.index(script)
            dest = self.temp_dir / f"{idx:02d}_{pdf_path.name}"
            pdf_path.rename(dest)

            return dest

        return None

    def _preparar_inputs(self, script, equipo, jornada):
        """
        Prepara inputs automáticos para el script.

        IMPORTANTE: Versión simplificada para demostración.
        En producción, debe usar los mismos TEAM_NAME_MAPPING y EQUIPOS_* que abp_informe_todo.py

        Args:
            script (str): Nombre del script
            equipo (str): Nombre del equipo
            jornada (int): Jornada final

        Returns:
            str: String con inputs simulados (separados por \n)
        """
        # Por defecto, asumir scripts de Opta que esperan: índice equipo, jornada
        respuestas = f"1\n{jornada}\n"

        # Ajustar para scripts específicos si es necesario
        if 'mediacoach' in script.lower():
            # MediaCoach suele pedir: índice, jornada, jornada
            respuestas = f"19\n{jornada}\n{jornada}\n"  # 19 = Villarreal por defecto
        elif 'sportian' in script.lower():
            # Sportian espera nombre directo
            respuestas = f"{equipo}\n"

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
        elif self.tipo == 'TACTICO':
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
