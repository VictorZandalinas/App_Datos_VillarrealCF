import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptaParquetAnalyzer:
    def __init__(self, file_paths):
        """
        Inicializa el analizador con las rutas de los archivos parquet
        """
        self.file_paths = file_paths
        self.dataframes = {}
        self.analysis_results = {}
        
    def load_parquet_files(self):
        """
        Carga todos los archivos parquet
        """
        print("Cargando archivos parquet...")
        for file_path in self.file_paths:
            try:
                filename = os.path.basename(file_path).replace('.parquet', '')
                df = pd.read_parquet(file_path)
                self.dataframes[filename] = df
                print(f"✓ Cargado: {filename} - {df.shape[0]} filas, {df.shape[1]} columnas")
            except Exception as e:
                print(f"✗ Error cargando {file_path}: {str(e)}")
    
    def analyze_dataframe(self, df, filename):
        """
        Analiza un DataFrame específico
        """
        analysis = {
            'filename': filename,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_analysis': {},
            'categorical_analysis': {},
            'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
        }
        
        # Análisis por columna
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Análisis numérico
                non_null_data = df[col].dropna()
                if len(non_null_data) > 0:
                    analysis['numeric_analysis'][col] = {
                        'min': float(non_null_data.min()),
                        'max': float(non_null_data.max()),
                        'mean': float(non_null_data.mean()),
                        'median': float(non_null_data.median()),
                        'std': float(non_null_data.std()) if len(non_null_data) > 1 else 0,
                        'unique_count': int(non_null_data.nunique()),
                        'sorted_unique': sorted(non_null_data.unique().tolist())[:20]  # Primeros 20 valores
                    }
            else:
                # Análisis categórico
                non_null_data = df[col].dropna()
                if len(non_null_data) > 0:
                    value_counts = non_null_data.value_counts()
                    analysis['categorical_analysis'][col] = {
                        'unique_count': int(non_null_data.nunique()),
                        'unique_values': non_null_data.unique().tolist()[:50],  # Primeros 50 valores
                        'top_values': value_counts.head(10).to_dict(),
                        'sample_values': non_null_data.head(20).tolist()
                    }
        
        return analysis
    
    def perform_analysis(self):
        """
        Realiza el análisis completo de todos los DataFrames
        """
        print("\nAnalizando archivos...")
        for filename, df in self.dataframes.items():
            print(f"Analizando: {filename}")
            self.analysis_results[filename] = self.analyze_dataframe(df, filename)
    
    def create_pdf_report(self, output_filename="datos_API_Opta.pdf"):
        """
        Crea un PDF con todos los análisis
        """
        print(f"\nGenerando PDF: {output_filename}")
        
        # Configuración del documento
        doc = SimpleDocTemplate(output_filename, pagesize=A4, 
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=1*inch, bottomMargin=1*inch)
        
        # Estilos
        styles = getSampleStyleSheet()
        
        # Estilos personalizados
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=HexColor('#2E4057')
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=HexColor('#048A81')
        )
        
        section_style = ParagraphStyle(
            'CustomSection',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            textColor=HexColor('#2E4057')
        )
        
        # Contenido del PDF
        story = []
        
        # Título principal
        story.append(Paragraph("Análisis de Datos - API de Opta", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Resumen general
        story.append(Paragraph("Resumen General", subtitle_style))
        
        summary_data = []
        summary_data.append(['Archivo', 'Filas', 'Columnas', 'Tamaño total'])
        
        total_rows = 0
        total_cols = 0
        
        for filename, analysis in self.analysis_results.items():
            rows, cols = analysis['shape']
            total_rows += rows
            total_cols += cols
            summary_data.append([filename, f"{rows:,}", str(cols), f"{rows * cols:,} celdas"])
        
        summary_data.append(['TOTAL', f"{total_rows:,}", str(total_cols), f"{total_rows * total_cols:,} celdas"])
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#048A81')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#E8F4FD')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        story.append(summary_table)
        story.append(PageBreak())
        
        # Análisis detallado por archivo
        for filename, analysis in self.analysis_results.items():
            story.append(Paragraph(f"Análisis Detallado: {filename}", subtitle_style))
            
            # Información básica
            story.append(Paragraph("Información Básica", section_style))
            basic_info = f"""
            <b>Dimensiones:</b> {analysis['shape'][0]:,} filas × {analysis['shape'][1]} columnas<br/>
            <b>Columnas:</b> {', '.join(analysis['columns'][:10])}{'...' if len(analysis['columns']) > 10 else ''}<br/>
            <b>Total de columnas:</b> {len(analysis['columns'])}
            """
            story.append(Paragraph(basic_info, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Análisis de columnas numéricas
            if analysis['numeric_analysis']:
                story.append(Paragraph("Columnas Numéricas", section_style))
                
                for col, num_data in analysis['numeric_analysis'].items():
                    story.append(Paragraph(f"<b>{col}</b>", styles['Heading4']))
                    
                    # Tabla con estadísticas
                    stats_data = [
                        ['Estadística', 'Valor'],
                        ['Mínimo', f"{num_data['min']:,.2f}"],
                        ['Máximo', f"{num_data['max']:,.2f}"],
                        ['Media', f"{num_data['mean']:,.2f}"],
                        ['Mediana', f"{num_data['median']:,.2f}"],
                        ['Desv. Estándar', f"{num_data['std']:,.2f}"],
                        ['Valores únicos', f"{num_data['unique_count']:,}"]
                    ]
                    
                    stats_table = Table(stats_data, colWidths=[2*inch, 2*inch])
                    stats_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#E8F4FD')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 1, black)
                    ]))
                    
                    story.append(stats_table)
                    
                    # Valores únicos ordenados (si no son muchos)
                    if num_data['unique_count'] <= 20:
                        sorted_vals = ', '.join([str(x) for x in num_data['sorted_unique']])
                        story.append(Paragraph(f"<b>Valores únicos ordenados:</b> {sorted_vals}", styles['Normal']))
                    else:
                        sample_vals = ', '.join([str(x) for x in num_data['sorted_unique']])
                        story.append(Paragraph(f"<b>Muestra de valores ordenados:</b> {sample_vals}...", styles['Normal']))
                    
                    story.append(Spacer(1, 0.15*inch))
            
            # Análisis de columnas categóricas
            if analysis['categorical_analysis']:
                story.append(Paragraph("Columnas Categóricas/Texto", section_style))
                
                for col, cat_data in analysis['categorical_analysis'].items():
                    story.append(Paragraph(f"<b>{col}</b>", styles['Heading4']))
                    
                    # Información básica
                    story.append(Paragraph(f"<b>Valores únicos:</b> {cat_data['unique_count']:,}", styles['Normal']))
                    
                    # Valores más frecuentes
                    if cat_data['top_values']:
                        story.append(Paragraph("<b>Valores más frecuentes:</b>", styles['Normal']))
                        
                        top_data = [['Valor', 'Frecuencia']]
                        for value, count in cat_data['top_values'].items():
                            # Truncar valores muy largos
                            display_value = str(value)[:50] + '...' if len(str(value)) > 50 else str(value)
                            top_data.append([display_value, f"{count:,}"])
                        
                        if len(top_data) > 1:  # Solo crear tabla si hay datos
                            top_table = Table(top_data, colWidths=[3*inch, 1*inch])
                            top_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#E8F4FD')),
                                ('TEXTCOLOR', (0, 0), (-1, 0), black),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 9),
                                ('GRID', (0, 0), (-1, -1), 1, black)
                            ]))
                            story.append(top_table)
                    
                    # Muestra de valores únicos
                    if len(cat_data['unique_values']) <= 20:
                        unique_vals = ', '.join([str(x)[:30] for x in cat_data['unique_values']])
                        story.append(Paragraph(f"<b>Todos los valores únicos:</b> {unique_vals}", styles['Normal']))
                    else:
                        sample_vals = ', '.join([str(x)[:30] for x in cat_data['unique_values'][:20]])
                        story.append(Paragraph(f"<b>Muestra de valores únicos:</b> {sample_vals}...", styles['Normal']))
                    
                    story.append(Spacer(1, 0.15*inch))
            
            # Información sobre valores nulos
            null_info = [col for col, count in analysis['null_counts'].items() if count > 0]
            if null_info:
                story.append(Paragraph("Valores Nulos", section_style))
                null_text = "Columnas con valores nulos: " + ', '.join([f"{col} ({analysis['null_counts'][col]:,})" for col in null_info])
                story.append(Paragraph(null_text, styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
            
            story.append(PageBreak())
        
        # Pie del documento
        story.append(Paragraph(f"Reporte generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles['Normal']))
        
        # Construir PDF
        doc.build(story)
        print(f"✓ PDF generado exitosamente: {output_filename}")

def main():
    """
    Función principal para ejecutar el análisis
    """
    # Lista de archivos parquet (ajusta las rutas según tu ubicación)
    parquet_files = [
        'abp_events.parquet',
        'player_stats.parquet',
        'player_xg_stats.parquet',
        'team_officials.parquet',
        'team_stats.parquet',
        'xg_events.parquet'
    ]
    
    # Verificar que los archivos existen
    existing_files = []
    for file in parquet_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print(f"⚠️  Archivo no encontrado: {file}")
    
    if not existing_files:
        print("❌ No se encontraron archivos parquet.")
        return
    
    # Crear analizador y ejecutar
    analyzer = OptaParquetAnalyzer(existing_files)
    analyzer.load_parquet_files()
    analyzer.perform_analysis()
    analyzer.create_pdf_report()
    
    print("\n🎉 Análisis completado exitosamente!")

if __name__ == "__main__":
    main()