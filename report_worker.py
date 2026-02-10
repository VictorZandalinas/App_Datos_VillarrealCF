#!/usr/bin/env python3
"""
Worker independiente para generación de informes.
Procesa jobs de la cola (carpeta jobs/) de forma desacoplada del servidor web.

Ejecutar como servicio systemd o con cron cada 30 segundos.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import glob
from datetime import datetime
from pathlib import Path

# Directorios de trabajo
BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"
PENDING_DIR = JOBS_DIR / "pending"
PROCESSING_DIR = JOBS_DIR / "processing"
COMPLETED_DIR = JOBS_DIR / "completed"
FAILED_DIR = JOBS_DIR / "failed"
INFORMES_DIR = BASE_DIR / "informes_generados"

# Timeouts
TIMEOUT_GLOBAL = 3600  # 1 hora max por informe (ABP requiere ~35 scripts)
TIMEOUT_INACTIVIDAD = 600  # 10 minutos (scripts tácticos pueden tardar sin output)

# Crear directorios si no existen
for d in [PENDING_DIR, PROCESSING_DIR, COMPLETED_DIR, FAILED_DIR, INFORMES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg):
    """Log con timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def actualizar_estado(job_path, **updates):
    """Actualiza el estado de un job de forma atómica"""
    try:
        with open(job_path, 'r') as f:
            job = json.load(f)

        job.update(updates)
        job['updated_at'] = datetime.now().isoformat()

        # Escribir a archivo temporal y luego mover (atómico)
        temp_path = str(job_path) + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(job, f, indent=2)
        shutil.move(temp_path, job_path)

        return job
    except Exception as e:
        log(f"Error actualizando estado: {e}")
        return None


def procesar_job(job_path):
    """Procesa un job de generación de informe"""

    # Leer el job
    with open(job_path, 'r') as f:
        job = json.load(f)

    job_id = job['id']
    script = job['script']
    equipo = job['equipo']
    j_inicio = job['j_inicio']
    j_fin = job['j_fin']

    log(f"Procesando job {job_id}: {script} - {equipo} J{j_inicio}-J{j_fin}")

    # Mover a processing
    processing_path = PROCESSING_DIR / f"{job_id}.json"
    shutil.move(job_path, processing_path)

    actualizar_estado(processing_path,
                      status='processing',
                      progress=5,
                      message='Iniciando generador...')

    process = None
    try:
        # Limpiar PDFs antiguos del directorio de trabajo
        for pdf in glob.glob(str(BASE_DIR / "*.pdf")):
            try:
                if time.time() - os.path.getmtime(pdf) > 3600:  # 1 hora
                    os.remove(pdf)
            except:
                pass

        # Ejecutar el script de generación
        cmd = [sys.executable, "-u", script, equipo, str(j_inicio), str(j_fin)]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR)
        )

        tiempo_inicio = time.time()
        ultimo_output = time.time()

        import select

        while True:
            # Verificar timeouts
            elapsed = time.time() - tiempo_inicio
            inactivity = time.time() - ultimo_output

            if elapsed > TIMEOUT_GLOBAL:
                raise TimeoutError(f"Timeout global: {TIMEOUT_GLOBAL}s excedido")

            if inactivity > TIMEOUT_INACTIVIDAD:
                raise TimeoutError(f"Timeout inactividad: {TIMEOUT_INACTIVIDAD}s sin output")

            # Verificar si terminó
            if process.poll() is not None:
                # Leer output restante
                remaining = process.stdout.read()
                if remaining:
                    log(f"Output final: {remaining[-200:]}")
                break

            # Leer con timeout
            ready, _, _ = select.select([process.stdout], [], [], 1.0)

            if ready:
                line = process.stdout.readline()
                if not line:
                    break

                ultimo_output = time.time()
                line = line.strip()

                if not line:
                    continue

                # Parsear progreso
                import re
                match = re.search(r"\[(\d+)/(\d+)\] Ejecutando: (.*) ---", line)

                if match:
                    pag_actual = int(match.group(1))
                    pag_total = int(match.group(2))
                    nombre_pag = match.group(3)
                    porcentaje = int((pag_actual / pag_total) * 90) + 5

                    actualizar_estado(processing_path,
                                      progress=porcentaje,
                                      message=f"Generando {pag_actual}/{pag_total}: {nombre_pag}")

                elif "Uniendo" in line:
                    actualizar_estado(processing_path,
                                      progress=95,
                                      message="Uniendo páginas en PDF final...")

        # Buscar el PDF generado
        actualizar_estado(processing_path, progress=97, message="Buscando PDF generado...")

        # Buscar PDFs recientes en el directorio base
        pdfs = glob.glob(str(BASE_DIR / "*.pdf"))
        if pdfs:
            pdf_reciente = max(pdfs, key=os.path.getctime)
            nombre_pdf = os.path.basename(pdf_reciente)
            destino = INFORMES_DIR / nombre_pdf

            shutil.move(pdf_reciente, destino)
            log(f"PDF movido a: {destino}")

            # Mover job a completed
            completed_path = COMPLETED_DIR / f"{job_id}.json"
            shutil.move(processing_path, completed_path)

            actualizar_estado(completed_path,
                              status='completed',
                              progress=100,
                              message='Informe listo',
                              pdf_path=str(destino),
                              pdf_name=nombre_pdf)

            log(f"Job {job_id} completado: {nombre_pdf}")
            return True
        else:
            raise FileNotFoundError("No se generó ningún PDF")

    except Exception as e:
        log(f"Error en job {job_id}: {e}")

        # Matar proceso si sigue vivo
        if process and process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=5)
            except:
                pass

        # Mover a failed
        failed_path = FAILED_DIR / f"{job_id}.json"
        if processing_path.exists():
            shutil.move(processing_path, failed_path)

        actualizar_estado(failed_path,
                          status='failed',
                          progress=100,
                          message=f'Error: {str(e)}',
                          error=str(e))

        return False


def limpiar_jobs_antiguos():
    """Limpia jobs completados/fallidos de más de 24 horas"""
    ahora = time.time()
    for carpeta in [COMPLETED_DIR, FAILED_DIR]:
        for archivo in carpeta.glob("*.json"):
            try:
                if ahora - archivo.stat().st_mtime > 86400:  # 24 horas
                    # También borrar el PDF asociado si existe
                    with open(archivo) as f:
                        job = json.load(f)
                    if 'pdf_path' in job and os.path.exists(job['pdf_path']):
                        os.remove(job['pdf_path'])
                    archivo.unlink()
                    log(f"Limpiado job antiguo: {archivo.name}")
            except Exception as e:
                log(f"Error limpiando {archivo}: {e}")


def run_once():
    """Ejecuta una iteración del worker"""

    # Limpiar jobs antiguos
    limpiar_jobs_antiguos()

    # Verificar si hay un job en processing (recuperar de crash)
    processing_jobs = list(PROCESSING_DIR.glob("*.json"))
    if processing_jobs:
        log(f"Recuperando job en processing: {processing_jobs[0].name}")
        # Mover de vuelta a pending para reprocesar
        for job_file in processing_jobs:
            try:
                with open(job_file) as f:
                    job = json.load(f)
                # Si lleva más de 25 minutos en processing, es un zombie
                updated = datetime.fromisoformat(job.get('updated_at', job['created_at']))
                if (datetime.now() - updated).total_seconds() > 1500:  # 25 min
                    shutil.move(job_file, PENDING_DIR / job_file.name)
                    log(f"Job zombie recuperado: {job_file.name}")
            except Exception as e:
                log(f"Error recuperando job: {e}")

    # Buscar jobs pendientes (ordenados por fecha)
    pending_jobs = sorted(PENDING_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)

    if not pending_jobs:
        return False

    # Procesar el más antiguo
    job_path = pending_jobs[0]
    procesar_job(job_path)
    return True


def run_daemon():
    """Ejecuta el worker como daemon (bucle infinito)"""
    log("Worker iniciado en modo daemon")

    while True:
        try:
            had_work = run_once()

            if not had_work:
                # Sin trabajo, esperar 5 segundos
                time.sleep(5)
            else:
                # Hubo trabajo, revisar inmediatamente si hay más
                time.sleep(1)

        except KeyboardInterrupt:
            log("Worker detenido por usuario")
            break
        except Exception as e:
            log(f"Error en worker: {e}")
            time.sleep(10)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Modo cron: ejecutar una vez
        run_once()
    else:
        # Modo daemon: bucle infinito
        run_daemon()
