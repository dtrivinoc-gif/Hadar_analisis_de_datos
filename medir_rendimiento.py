"""
medir_rendimiento.py — Mide tiempo y memoria de las partes que podrían ser lentas. NO es un test: solo imprime números.
Uso (con los .py del proyecto en la carpeta de arriba):   python herramientas/medir_rendimiento.py [--grande]
Con --grande también prueba entrenar() con 500.000 filas (puede tardar varios minutos).
"""
import os, sys, time, tracemalloc, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import prediccion as P, series_tiempo as st

def medir(nombre, fn):
    tracemalloc.start(); t0 = time.perf_counter()
    r = fn(); dt = time.perf_counter() - t0
    _, pico = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"{nombre:<58s} {dt:8.2f} s   pico de memoria {pico / 1e6:8.1f} MB")
    return r

def tabla(n, n_cat=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    cat = rng.integers(0, n_cat, n).astype(str)
    y = (x + rng.normal(0, 1, n) > 0).astype(int)
    return pd.DataFrame({"x": x, "x2": rng.normal(size=n), "cat": cat, "cat2": rng.choice(list("abcde"), n), "y": y})

if __name__ == "__main__":
    grande = "--grande" in sys.argv
    print("== Preprocesador.transformar (columna de texto con muchas categorías → frecuencia) ==")
    for n in (100_000, 300_000):
        df = tabla(n, n_cat=n // 2)
        pre = P.Preprocesador(["x", "x2", "cat", "cat2"]).ajustar(df)
        medir(f"transformar, {n:,} filas, {n // 2:,} categorías", lambda: pre.transformar(df))
    print("\n== Holt (varias combinaciones × orígenes) ==")
    for n in (200, 800, 3000):
        y = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
        f = pd.date_range("2020-01-01", periods=n, freq="D")
        s = st.SerieTiempo(f, y, "D", "suma", 7, "x", 0.0)
        medir(f"proyectar, serie de {n} períodos, h=30", lambda: st.proyectar(s, 30, 0.8))
    print("\n== Agotamiento con muchos elementos ==")
    for k in (300, 3000):
        d = pd.date_range("2024-01-01", periods=120, freq="D"); rng = np.random.default_rng(0)
        df = pd.DataFrame({"f": np.tile(d, k), "c": rng.poisson(5, 120 * k) * 1.0,
                           "e": np.repeat([f"E{i}" for i in range(k)], 120), "s": np.repeat(rng.uniform(100, 900, k), 120)})
        from pronosticos import ConfigPronostico as C
        cfg = C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", columna_entidad="e", columna_stock="s")
        medir(f"analizar_agotamiento, {k:,} elementos", lambda: st.analizar_agotamiento(df, cfg))
    print("\n== entrenar() ==")
    for n in ((150_000, 500_000) if grande else (50_000, 150_000)):
        df = tabla(n)
        cfg = P.ConfigPrediccion(objetivo="y", tipo="clasificacion", pliegues=1)
        medir(f"entrenar, {n:,} filas, 4 columnas", lambda: P.entrenar(df, cfg, ["x", "x2", "cat", "cat2"]))
