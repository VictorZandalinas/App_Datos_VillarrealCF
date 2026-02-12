#!/usr/bin/env python3
"""
UTILIDADES DE GESTIÓN DE MEMORIA
Limpieza y monitoreo de memoria para evitar OOM en servidores con RAM limitada
"""

import gc
import matplotlib.pyplot as plt

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil no disponible - monitoreo de memoria limitado")


def clean_memory_agresivo():
    """
    Limpieza crítica de memoria entre chunks.
    Cierra figuras matplotlib y fuerza triple recolección de basura.
    """
    # Cerrar todas las figuras matplotlib abiertas
    plt.close('all')

    # Triple recolección para asegurar liberación efectiva
    # (necesario en Python por referencias circulares)
    gc.collect()
    gc.collect()
    gc.collect()


def get_memory_info():
    """
    Retorna información del uso de memoria del proceso actual.

    Returns:
        dict: Diccionario con 'rss_mb' (memoria residente) y 'percent' (porcentaje)
              o valores por defecto si psutil no está disponible
    """
    if not PSUTIL_AVAILABLE:
        return {
            'rss_mb': 0,
            'percent': 0,
            'available': False
        }

    try:
        process = psutil.Process()
        mem_info = process.memory_info()

        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Convertir bytes a MB
            'percent': process.memory_percent(),
            'available': True
        }
    except Exception as e:
        print(f"⚠️ Error obteniendo info de memoria: {e}")
        return {
            'rss_mb': 0,
            'percent': 0,
            'available': False
        }


def check_memory_threshold(threshold_mb=2500, auto_clean=True):
    """
    Verifica si el uso de memoria supera un umbral y opcionalmente limpia.

    Args:
        threshold_mb (int): Umbral de memoria en MB (default: 2500)
        auto_clean (bool): Si True, limpia memoria automáticamente al superar umbral

    Returns:
        bool: True si se superó el umbral (y se limpió si auto_clean=True), False si no
    """
    mem_info = get_memory_info()

    if not mem_info['available']:
        # Sin psutil, solo hacer limpieza preventiva
        if auto_clean:
            clean_memory_agresivo()
        return False

    if mem_info['rss_mb'] > threshold_mb:
        print(f"⚠️ Memoria alta: {mem_info['rss_mb']:.0f}MB (>{threshold_mb}MB) - ", end='')

        if auto_clean:
            print("Liberando...")
            clean_memory_agresivo()

            # Verificar cuánto liberamos
            new_mem = get_memory_info()
            freed_mb = mem_info['rss_mb'] - new_mem['rss_mb']
            if freed_mb > 0:
                print(f"   ✅ Liberados {freed_mb:.0f}MB (ahora: {new_mem['rss_mb']:.0f}MB)")
            else:
                print(f"   ⚠️ Memoria no liberada significativamente ({new_mem['rss_mb']:.0f}MB)")
        else:
            print("Advertencia emitida")

        return True

    return False


def format_memory_mb(bytes_value):
    """
    Formatea un valor en bytes a megabytes con formato legible.

    Args:
        bytes_value (int): Valor en bytes

    Returns:
        str: String formateado (ej: "1024.5 MB")
    """
    mb_value = bytes_value / 1024 / 1024
    return f"{mb_value:.1f} MB"


def get_system_memory_info():
    """
    Obtiene información de memoria del sistema (no solo del proceso).

    Returns:
        dict: Información de memoria del sistema o None si no disponible
    """
    if not PSUTIL_AVAILABLE:
        return None

    try:
        mem = psutil.virtual_memory()
        return {
            'total_mb': mem.total / 1024 / 1024,
            'available_mb': mem.available / 1024 / 1024,
            'percent_used': mem.percent,
            'used_mb': mem.used / 1024 / 1024
        }
    except Exception as e:
        print(f"⚠️ Error obteniendo info del sistema: {e}")
        return None


if __name__ == "__main__":
    # Test básico del módulo
    print("=== Test de memory_utils ===\n")

    print("1. Info de memoria del proceso:")
    mem = get_memory_info()
    if mem['available']:
        print(f"   RSS: {mem['rss_mb']:.2f} MB")
        print(f"   Percent: {mem['percent']:.2f}%")
    else:
        print("   No disponible (instalar psutil)")

    print("\n2. Info de memoria del sistema:")
    sys_mem = get_system_memory_info()
    if sys_mem:
        print(f"   Total: {sys_mem['total_mb']:.0f} MB")
        print(f"   Usada: {sys_mem['used_mb']:.0f} MB ({sys_mem['percent_used']:.1f}%)")
        print(f"   Disponible: {sys_mem['available_mb']:.0f} MB")
    else:
        print("   No disponible")

    print("\n3. Test de limpieza:")
    clean_memory_agresivo()
    print("   ✅ Limpieza ejecutada")

    print("\n4. Test de threshold (2000 MB):")
    exceeded = check_memory_threshold(threshold_mb=2000, auto_clean=True)
    print(f"   Umbral superado: {exceeded}")
