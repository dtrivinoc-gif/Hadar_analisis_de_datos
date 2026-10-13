import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import numpy as np, pandas as pd
import series_tiempo as st, eventos as ev, pronosticos as pr
from pronosticos import ConfigPronostico as C

def serie(y, freq="D", m=7):
    f = pd.date_range("2022-01-01", periods=len(y), freq=freq)
    return st.SerieTiempo(f, np.asarray(y, float), freq, "suma", m, "x", 0.0)

def gen(kind, n, rng):
    t = np.arange(n)
    if kind == "rw": return 100 + np.cumsum(rng.normal(0, 1, n))
    if kind == "estacional": return 50 + 10*np.sin(2*np.pi*t/7) + rng.normal(0, 2, n)
    if kind == "tendencia": return 20 + 0.5*t + rng.normal(0, 3, n)

import pronostico_base as _pb
_pb.set_formato_numerico("estandar")   # los tests comparan textos con el formato estándar (1,234.5)

# Pruebas estadísticas con muchas repeticiones (lentas por naturaleza). Se corren con:  HADAR_LENTOS=1 python -m unittest discover -s tests -t .
LENTO = unittest.skipUnless(os.environ.get("HADAR_LENTOS"), "prueba estadística lenta: corre con HADAR_LENTOS=1")