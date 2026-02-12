#!/usr/bin/env python3
"""
FUSIÓN INCREMENTAL DE PDFs
Fusiona PDFs de 2 en 2 (merge sort) para minimizar uso de memoria.
Nunca carga más de 2 PDFs simultáneamente en RAM.
"""

import os
import shutil
import gc
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter


def fusionar_pdfs_incremental(pdfs_list, output_path, temp_dir):
    """
    Fusiona una lista de PDFs usando estrategia de merge sort.

    En lugar de cargar todos los PDFs en memoria (consumo lineal O(n)),
    los fusiona de 2 en 2 en niveles sucesivos (consumo constante O(1)).

    Ejemplo con 8 PDFs:
        Nivel 0: [1,2,3,4,5,6,7,8]
        Fusiona (1,2)→A, (3,4)→B, (5,6)→C, (7,8)→D

        Nivel 1: [A,B,C,D]
        Fusiona (A,B)→E, (C,D)→F

        Nivel 2: [E,F]
        Fusiona (E,F)→FINAL

    Args:
        pdfs_list (list): Lista de rutas Path a los PDFs a fusionar
        output_path (str/Path): Ruta de salida del PDF final
        temp_dir (Path): Directorio para archivos temporales intermedios

    Returns:
        Path: Ruta al PDF generado, o None si error
    """
    if not pdfs_list:
        print("⚠️ Lista de PDFs vacía")
        return None

    if len(pdfs_list) == 1:
        # Caso trivial: solo un PDF
        shutil.copy(pdfs_list[0], output_path)
        return Path(output_path)

    # Convertir a Path objects
    current_level = [Path(p) for p in pdfs_list]
    level = 0

    print(f"🔄 Fusionando {len(current_level)} PDFs incrementalmente...")

    while len(current_level) > 1:
        next_level = []
        pairs_in_level = (len(current_level) + 1) // 2

        print(f"   Nivel {level}: {len(current_level)} PDFs → {pairs_in_level} fusiones")

        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                # Fusionar par
                merged_path = temp_dir / f"merged_L{level}_P{i//2}.pdf"
                _merge_two_pdfs(current_level[i], current_level[i+1], merged_path)
                next_level.append(merged_path)

                # Liberar PDFs originales si son temporales
                if current_level[i].parent == temp_dir:
                    try:
                        os.remove(current_level[i])
                    except Exception as e:
                        print(f"   ⚠️ No se pudo borrar {current_level[i].name}: {e}")

                if current_level[i+1].parent == temp_dir:
                    try:
                        os.remove(current_level[i+1])
                    except Exception as e:
                        print(f"   ⚠️ No se pudo borrar {current_level[i+1].name}: {e}")

                # Forzar liberación de memoria
                gc.collect()
            else:
                # PDF impar sin pareja - pasa al siguiente nivel
                next_level.append(current_level[i])

        current_level = next_level
        level += 1

    # Copiar resultado final
    final_pdf = current_level[0]
    shutil.copy(final_pdf, output_path)

    # Limpiar último temporal si es necesario
    if final_pdf.parent == temp_dir:
        try:
            os.remove(final_pdf)
        except:
            pass

    print(f"✅ Fusión completada: {Path(output_path).name}")
    return Path(output_path)


def _merge_two_pdfs(pdf1_path, pdf2_path, output_path):
    """
    Fusiona exactamente 2 PDFs en uno solo.

    Args:
        pdf1_path (Path): Ruta al primer PDF
        pdf2_path (Path): Ruta al segundo PDF
        output_path (Path): Ruta de salida
    """
    writer = PdfWriter()

    # Abrir y leer primer PDF
    with open(pdf1_path, 'rb') as f1:
        reader1 = PdfReader(f1)
        for page in reader1.pages:
            writer.add_page(page)

    # Abrir y leer segundo PDF
    with open(pdf2_path, 'rb') as f2:
        reader2 = PdfReader(f2)
        for page in reader2.pages:
            writer.add_page(page)

    # Escribir resultado
    with open(output_path, 'wb') as out:
        writer.write(out)


def fusionar_pdfs_tradicional(pdfs_list, output_path):
    """
    Fusión tradicional (carga todos en memoria).
    Solo para comparación - NO usar en producción con RAM limitada.

    Args:
        pdfs_list (list): Lista de rutas a PDFs
        output_path (str/Path): Ruta de salida

    Returns:
        Path: Ruta al PDF generado
    """
    print(f"⚠️ ADVERTENCIA: Fusión tradicional (alto uso de memoria)")

    writer = PdfWriter()

    for pdf_path in pdfs_list:
        if not os.path.exists(pdf_path):
            print(f"   ⚠️ Saltando {pdf_path} (no existe)")
            continue

        if os.path.getsize(pdf_path) < 100:
            print(f"   ⚠️ Saltando {pdf_path} (archivo vacío)")
            continue

        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                writer.add_page(page)

    with open(output_path, 'wb') as out:
        writer.write(out)

    return Path(output_path)


def get_pdf_page_count(pdf_path):
    """
    Obtiene el número de páginas de un PDF sin cargarlo completamente.

    Args:
        pdf_path (str/Path): Ruta al PDF

    Returns:
        int: Número de páginas, o 0 si error
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except Exception as e:
        print(f"⚠️ Error leyendo {pdf_path}: {e}")
        return 0


if __name__ == "__main__":
    # Test básico del módulo
    print("=== Test de pdf_fusion ===\n")

    # Verificar que PyPDF2 está disponible
    print("✅ PyPDF2 importado correctamente")

    # Mostrar ejemplo de uso
    print("\nEjemplo de uso:")
    print("""
    from pathlib import Path
    import pdf_fusion

    pdfs = [Path('reporte1.pdf'), Path('reporte2.pdf'), Path('reporte3.pdf')]
    temp_dir = Path('reportes_temporales')

    pdf_fusion.fusionar_pdfs_incremental(
        pdfs_list=pdfs,
        output_path='informe_completo.pdf',
        temp_dir=temp_dir
    )
    """)

    print("\n⚠️ Para test real, ejecutar con PDFs válidos")
