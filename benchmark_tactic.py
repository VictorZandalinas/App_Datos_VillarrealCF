#!/usr/bin/env python3
"""Benchmark de tiempos para scripts tácticos"""
import time
import subprocess
import sys
from datetime import datetime

SCRIPTS = [
    'tactic1_opta_clasificacion_liga.py',
    'tactic1.1_mediacoach_resumen_con_balon.py',
    'tactic1.2_mediacoach_resumen_sin_balon.py',
    'tactic1.3_mediacoach_evolucion_resumen_general.py',
    'tactic1.4_opta_xT.py',
    'tactic2_opta_mapa_pases.py',
    'tactic2.1_opta_mapa_pases_campo_contrario.py',
    'tactic2.2_opta_contraataque.py',
    'tactic2.2.1_opta_evolutivo_contraataque.py',
    'tactic3_opta_saque_puerta_secuencias.py',
    'tactic3.1_opta_inicios_blocaje_portero.py',
    'tactic4_saques_banda_ofensivos.py',
    'tactic5_opta_perdidas.py',
]

EQUIPO = "Villarreal"
JORNADA = "15"

print(f"=== BENCHMARK TÁCTICO ===")
print(f"Equipo: {EQUIPO}, Jornada: {JORNADA}")
print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

resultados = []

for i, script in enumerate(SCRIPTS, 1):
    print(f"\n[{i}/{len(SCRIPTS)}] Probando: {script}")

    inicio = time.time()

    try:
        # Ejecutar con timeout de 5 minutos
        proc = subprocess.run(
            ['python', script],
            input=f"{EQUIPO}\n{JORNADA}\n",
            capture_output=True,
            text=True,
            timeout=300
        )

        duracion = time.time() - inicio

        if proc.returncode == 0:
            estado = "✅ OK"
        else:
            estado = f"❌ ERROR (code {proc.returncode})"

        print(f"   {estado} | Tiempo: {duracion:.2f}s")

        # Ver si hay stderr relevante
        if proc.stderr and "error" in proc.stderr.lower():
            print(f"   STDERR: {proc.stderr[:200]}")

        resultados.append({
            'script': script,
            'tiempo': duracion,
            'estado': 'OK' if proc.returncode == 0 else 'ERROR'
        })

    except subprocess.TimeoutExpired:
        duracion = 300
        print(f"   ⚠️ TIMEOUT (>5min)")
        resultados.append({
            'script': script,
            'tiempo': duracion,
            'estado': 'TIMEOUT'
        })
    except Exception as e:
        duracion = 0
        print(f"   ❌ EXCEPCIÓN: {e}")
        resultados.append({
            'script': script,
            'tiempo': duracion,
            'estado': f'EXCEPCION: {e}'
        })

# Ordenar por tiempo (mayor a menor)
resultados.sort(key=lambda x: x['tiempo'], reverse=True)

print("\n" + "=" * 60)
print("=== RANKING POR TIEMPO (mayor a menor) ===")
print("=" * 60)

for i, r in enumerate(resultados, 1):
    barra = "█" * int(r['tiempo'] / 10) if r['tiempo'] > 0 else ""
    print(f"{i:2}. {r['tiempo']:7.2f}s | {r['estado']:15} | {r['script']} {barra}")

# Guardar resultados
with open('benchmark_resultados.txt', 'w') as f:
    f.write(f"Benchmark realizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Equipo: {EQUIPO}, Jornada: {JORNADA}\n\n")
    for r in resultados:
        f.write(f"{r['tiempo']:.2f}s | {r['estado']} | {r['script']}\n")

print(f"\n📄 Resultados guardados en: benchmark_resultados.txt")
