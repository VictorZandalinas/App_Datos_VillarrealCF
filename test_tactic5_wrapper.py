#!/usr/bin/env python3
"""Test para debuggear tactic5 con el wrapper"""
import subprocess
import sys
import textwrap

injected_code = textwrap.dedent("""\
    import matplotlib, pandas as pd, sys, numpy as np
    import os as _os
    matplotlib.use('Agg')
    _j0, _j1 = 1, 15
    _equipo_key = 'Villarreal'
    _is_liga = False
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
        except Exception as e:
            print(f'_read_one error: {e}')
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
        if _is_liga:
            return _read_one(path)
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
    # Auto-selección de equipo
    import builtins as _builtins, sys as _sys
    _orig_input = _builtins.input
    _last_teams_cache = [None]
    _rp_orig_tracking = pd.read_parquet
    def _rp_tracking(path, *a, **kw):
        result = _rp_orig_tracking(path, *a, **kw)
        for _tc in result.columns:
            if 'team' in _tc.lower() and ('name' in _tc.lower() or _tc.lower() == 'team'):
                _last_teams_cache[0] = sorted(result[_tc].dropna().unique().tolist())
                print(f'[DEBUG] Equipos cache: {_last_teams_cache[0]}', file=sys.stderr)
                break
        return result
    pd.read_parquet = _rp_tracking
    _stdin_lines = _sys.stdin.read().splitlines()
    print(f'[DEBUG] Stdin: {_stdin_lines}', file=sys.stderr)
    _stdin_idx = [0]
    def _auto_input(prompt=''):
        queued = _stdin_lines[_stdin_idx[0]] if _stdin_idx[0] < len(_stdin_lines) else ''
        _stdin_idx[0] += 1
        print(f'[DEBUG] input() -> {queued!r}', file=sys.stderr)
        if 'equipo' in str(prompt).lower() and _last_teams_cache[0] is not None:
            try:
                _idx = _last_teams_cache[0].index(_equipo_key) + 1
                print(f'[DEBUG] Auto-idx: {_idx}', file=sys.stderr)
                return str(_idx)
            except (ValueError, TypeError):
                pass
        return queued
    _builtins.input = _auto_input
    exec(open('tactic5_opta_perdidas.py', encoding='utf-8').read())
""")

proc = subprocess.Popen(
    [sys.executable, '-u', '-c', injected_code],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)
out, _ = proc.communicate(input='20\n15\n', timeout=120)
print('=== SALIDA ===')
print(out)
