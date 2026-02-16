#!/usr/bin/env python3
"""
Script to fix syntax errors in sportian ABP files where filters were incorrectly placed outside try blocks
"""
import re

def fix_file_syntax(filepath):
    """
    Fixes the syntax where filter code was placed outside try blocks
    Pattern:
        try:
            df = pd.read_parquet(...)
        # Filtrar...  <-- WRONG (outside try)

    Should be:
        try:
            df = pd.read_parquet(...)
            # Filtrar...  <-- CORRECT (inside try)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to find misplaced filter code
    # Looks for pd.read_parquet followed by filter code that's not indented enough
    pattern = r'(        (\w+) = pd\.read_parquet\([^)]+\))\n        # Filtrar por jornadas'
    replacement = r'\1\n\n            # Filtrar por jornadas'

    content = re.sub(pattern, replacement, content)

    # Fix the indentation of the entire filter block when it's misplaced
    # Pattern: find filter blocks that start at column 8 when they should be at column 12
    filter_block_pattern = r'^        # Filtrar por jornadas si las variables de entorno están definidas\n        j_ini = os\.environ\.get'

    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line is a misplaced filter comment (8 spaces instead of 12)
        if line == '        # Filtrar por jornadas si las variables de entorno están definidas':
            # Look back to see if previous line was a pd.read_parquet
            if i > 0 and 'pd.read_parquet' in lines[i-1]:
                # Add blank line and properly indented filter
                fixed_lines.append('')
                fixed_lines.append('            # Filtrar por jornadas si las variables de entorno están definidas')
                i += 1

                # Fix indentation for the rest of the filter block (next ~15 lines)
                indent_count = 0
                while i < len(lines) and indent_count < 20:
                    current_line = lines[i]
                    if current_line.startswith('        ') and not current_line.startswith('            '):
                        # Add 4 more spaces
                        fixed_lines.append('    ' + current_line)
                    else:
                        # Either already correctly indented or end of filter block
                        if current_line.startswith('            ') or current_line.strip() == '':
                            fixed_lines.append(current_line)
                        else:
                            # End of filter block
                            fixed_lines.append(current_line)
                            break
                    i += 1
                    indent_count += 1
                continue

        fixed_lines.append(line)
        i += 1

    content = '\n'.join(fixed_lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Fixed {filepath}")

if __name__ == '__main__':
    import sys
    import os

    base_dir = '/Users/imac/Programas/App_Datos_VillarrealCF'
    os.chdir(base_dir)

    files_to_fix = [
        'abp9.1.1_sportian_corners_ofensivo_posicionamiento_abierto.py',
        'abp9.1.2_sportian_corners_ofensivo_posicionamiento_cerrado.py',
    ]

    for filename in files_to_fix:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            fix_file_syntax(filepath)
        else:
            print(f"❌ Not found: {filepath}")
