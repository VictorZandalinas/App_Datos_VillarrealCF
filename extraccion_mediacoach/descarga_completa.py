#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para procesar archivos de MediaCoach secuencialmente.
Adaptado para ejecución automática desde Dash (Villarreal CF).
"""

import subprocess
import sys
import time
import os
import argparse
from datetime import datetime
import logging

class ProcesadorSecuencial:
    def __init__(self, liga=None, temporada=None, j_inicio=None, j_fin=None):
        # Configurar logging
        log_filename = os.path.join(os.path.dirname(__file__), "proceso_mediacoach.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Guardamos los parámetros recibidos
        self.args_descarga = [liga, temporada, j_inicio, j_fin] if liga else None
        
        # Ruta base donde están los scripts (la propia carpeta de este archivo)
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        # Lista de archivos en orden de ejecución
        self.archivos = [
            {
                'nombre': 'extraccion_nueva_mediacoach_todas_ligas.py',  # ✅ CAMBIO 1: nombre correcto
                'tipo': 'python',
                'descripcion': 'Extracción inicial de MediaCoach',
                'interactivo': True  # Requiere input del usuario
            },
            {
                'nombre': '1.carpetas_con_jornadas.py',  # ✅ CAMBIO 2: .py en lugar de .ipynb
                'tipo': 'python',                        # ✅ CAMBIO 3: python en lugar de notebook
                'descripcion': 'Procesamiento de carpetas con jornada'
            },
            {
                'nombre': '2.extraer_eventos_xml.py',
                'tipo': 'python',
                'descripcion': 'Extracción de eventos XML'
            },
            {
                'nombre': '3.extraer_rendimiento_xlsx.js',
                'tipo': 'javascript',
                'descripcion': 'Extracción de rendimiento XLSX'
            },
            {
                'nombre': '3.2_fusionar.py',
                'tipo': 'python',
                'descripcion': 'Fusión de datos'
            },
            {
                'nombre': '4.extraer_maxima_exigencia.js',
                'tipo': 'javascript',
                'descripcion': 'Extracción de máxima exigencia'
            },
            {
                'nombre': '5.extraer_estadisticas_csv.py',
                'tipo': 'python',
                'descripcion': 'Extracción de estadísticas CSV'
            }
        ]
        
        self.total_archivos = len(self.archivos)
        self.tiempo_inicio = None

    def ejecutar_python(self, nombre_archivo, interactivo=False):
        """Ejecuta un archivo Python transmitiendo la salida en tiempo real"""
        cmd = [sys.executable, nombre_archivo]
        if interactivo and self.args_descarga:
            cmd.extend(self.args_descarga)
            
        # Usamos Popen en lugar de run para transmitir en vivo
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            cwd=self.base_path
        )
        
        # Leemos y re-imprimimos cada línea que suelte el script hijo
        for line in process.stdout:
            print(line.strip())
            sys.stdout.flush()
            
        process.wait()
        return process

    def ejecutar_javascript(self, nombre_archivo):
        """Ejecuta un archivo JavaScript transmitiendo la salida en tiempo real"""
        cmd = ['node', nombre_archivo]
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            cwd=self.base_path
        )
        
        for line in process.stdout:
            print(line.strip())
            sys.stdout.flush()
            
        process.wait()
        return process

    def ejecutar_archivo(self, archivo_info):
        """Ejecuta un archivo según su tipo"""
        nombre = archivo_info['nombre']
        tipo = archivo_info['tipo']
        interactivo = archivo_info.get('interactivo', False)
        
        self.logger.info(f"🚀 INICIANDO: {nombre} ({archivo_info['descripcion']})")
        tiempo_inicio_archivo = time.time()
        
        try:
            if tipo == 'python':
                resultado = self.ejecutar_python(nombre, interactivo)
            elif tipo == 'javascript':
                resultado = self.ejecutar_javascript(nombre)
            else:
                return False, f"Tipo de archivo no soportado: {tipo}"
                
            tiempo_ejecucion = time.time() - tiempo_inicio_archivo
            
            if resultado.returncode == 0:
                self.logger.info(f"✅ COMPLETADO: {nombre} (⏱️ {tiempo_ejecucion:.2f}s)")
                return True, f"Éxito"
            else:
                self.logger.error(f"❌ ERROR en {nombre}")
                self.logger.error(f"Detalle: {resultado.stderr}")
                return False, f"Error en ejecución"
                
        except Exception as e:
            return False, f"Excepción: {str(e)}"

    def procesar_todos(self):
        """Procesa todos los archivos secuencialmente sin interrupciones"""
        self.tiempo_inicio = time.time()
        resultados = []
        
        print(f"🎬 Iniciando procesamiento secuencial para {self.total_archivos} módulos.")
        
        for i, archivo_info in enumerate(self.archivos, 1):
            # En Dash esto se leerá como progreso
            print(f"PROGRESS:{int((i/self.total_archivos)*100)} - Procesando {archivo_info['descripcion']}...")
            
            exito, mensaje = self.ejecutar_archivo(archivo_info)
            
            resultados.append({'archivo': archivo_info['nombre'], 'exito': exito, 'mensaje': mensaje})
            
            if not exito:
                self.logger.error(f"🛑 Proceso detenido por error en {archivo_info['nombre']}")
                break
            
            # Pausa breve de cortesía
            time.sleep(1)
        
        # Resumen final
        tiempo_total = time.time() - self.tiempo_inicio
        exitosos = sum(1 for r in resultados if r['exito'])
        self.logger.info(f"🏁 Finalizado. Exitosos: {exitosos}/{len(resultados)}. Tiempo total: {tiempo_total:.2f}s")

if __name__ == "__main__":
    # Configuración de argumentos para recibir desde Dash
    parser = argparse.ArgumentParser(description='Orquestador de descarga MediaCoach')
    parser.add_argument('--liga', type=str, help='Nombre de la liga')
    parser.add_argument('--temporada', type=str, help='Temporada (ej: 2024-2025)')
    parser.add_argument('--j_inicio', type=str, help='Jornada inicial')
    parser.add_argument('--j_fin', type=str, help='Jornada final')
    
    args = parser.parse_args()

    # Si se pasan argumentos, lanzamos el procesador con ellos
    if args.liga:
        procesador = ProcesadorSecuencial(
            liga=args.liga, 
            temporada=args.temporada, 
            j_inicio=args.j_inicio, 
            j_fin=args.j_fin
        )
    else:
        # Fallback por si se lanza a mano sin argumentos
        procesador = ProcesadorSecuencial()
        
    procesador.procesar_todos()