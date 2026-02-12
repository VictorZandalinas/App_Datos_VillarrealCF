#!/usr/bin/env python3
"""
MONITOR DE MEMORIA EN BACKGROUND
Thread daemon que monitorea uso de memoria y registra estadísticas
"""

import threading
import time
from collections import deque

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryMonitor:
    """
    Monitor de memoria que corre en un thread separado.
    No bloquea la ejecución principal y permite rastrear picos de memoria.
    """

    def __init__(self, threshold_mb=3000, interval=5, max_history=60):
        """
        Inicializa el monitor de memoria.

        Args:
            threshold_mb (int): Umbral de memoria para alertas (MB)
            interval (int): Intervalo entre mediciones (segundos)
            max_history (int): Máximo de entradas en historial
        """
        self.threshold_mb = threshold_mb
        self.interval = interval
        self.history = deque(maxlen=max_history)
        self.running = False
        self.thread = None
        self.peak_memory = 0
        self.alert_count = 0
        self.psutil_available = PSUTIL_AVAILABLE

        if not self.psutil_available:
            print("⚠️ MemoryMonitor: psutil no disponible - monitoreo desactivado")

    def start(self):
        """Inicia el thread de monitoreo"""
        if not self.psutil_available:
            return

        if self.running:
            print("⚠️ MemoryMonitor ya está corriendo")
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"✅ MemoryMonitor iniciado (umbral: {self.threshold_mb}MB, intervalo: {self.interval}s)")

    def stop(self):
        """Detiene el thread de monitoreo"""
        if not self.running:
            return

        self.running = False

        if self.thread:
            self.thread.join(timeout=10)

            if self.thread.is_alive():
                print("⚠️ MemoryMonitor no se detuvo en 10s")
            else:
                print("✅ MemoryMonitor detenido")

    def _monitor_loop(self):
        """Loop principal del monitor (corre en thread separado)"""
        try:
            process = psutil.Process()
        except Exception as e:
            print(f"❌ MemoryMonitor: No se pudo obtener proceso: {e}")
            return

        while self.running:
            try:
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                mem_percent = process.memory_percent()

                # Actualizar pico
                if mem_mb > self.peak_memory:
                    self.peak_memory = mem_mb

                # Registrar en historial
                self.history.append({
                    'timestamp': time.time(),
                    'rss_mb': mem_mb,
                    'percent': mem_percent
                })

                # Alertar si supera umbral
                if mem_mb > self.threshold_mb:
                    self.alert_count += 1
                    print(f"⚠️ [MONITOR] Memoria crítica: {mem_mb:.0f}MB (>{self.threshold_mb}MB)")

                time.sleep(self.interval)

            except Exception as e:
                print(f"⚠️ [MONITOR] Error en loop: {e}")
                time.sleep(self.interval)

    def get_stats(self):
        """
        Obtiene estadísticas del monitor.

        Returns:
            dict: Estadísticas de memoria o dict vacío si no hay datos
        """
        if not self.psutil_available:
            return {}

        if not self.history:
            return {
                'peak_mb': self.peak_memory,
                'alert_count': self.alert_count,
                'history_size': 0
            }

        recent = list(self.history)[-12:]  # Último minuto (12 * 5s)

        return {
            'current_mb': recent[-1]['rss_mb'] if recent else 0,
            'peak_mb': self.peak_memory,
            'avg_last_minute': sum(h['rss_mb'] for h in recent) / len(recent) if recent else 0,
            'alert_count': self.alert_count,
            'history_size': len(self.history)
        }

    def get_history(self):
        """
        Obtiene el historial completo.

        Returns:
            list: Lista de mediciones
        """
        return list(self.history)

    def reset_stats(self):
        """Resetea estadísticas (útil para tests múltiples)"""
        self.peak_memory = 0
        self.alert_count = 0
        self.history.clear()

    def __enter__(self):
        """Context manager: inicia monitor automáticamente"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: detiene monitor automáticamente"""
        self.stop()
        return False


if __name__ == "__main__":
    # Test básico del módulo
    print("=== Test de MemoryMonitor ===\n")

    if not PSUTIL_AVAILABLE:
        print("❌ psutil no disponible - instalar con: pip install psutil")
        exit(1)

    print("Iniciando monitor de memoria por 15 segundos...\n")

    # Test usando context manager
    with MemoryMonitor(threshold_mb=1000, interval=2) as monitor:
        # Simular trabajo
        for i in range(7):
            print(f"Tick {i+1}/7")
            time.sleep(2)

            # Mostrar stats cada 6 segundos
            if (i + 1) % 3 == 0:
                stats = monitor.get_stats()
                print(f"   Stats: Current={stats['current_mb']:.0f}MB, "
                      f"Peak={stats['peak_mb']:.0f}MB, "
                      f"Avg={stats['avg_last_minute']:.0f}MB")

    # Al salir del 'with', el monitor se detiene automáticamente
    print("\n✅ Test completado")

    # Mostrar stats finales
    final_stats = monitor.get_stats()
    print(f"\nEstadísticas finales:")
    print(f"  Pico de memoria: {final_stats['peak_mb']:.2f} MB")
    print(f"  Alertas: {final_stats['alert_count']}")
    print(f"  Muestras: {final_stats['history_size']}")
