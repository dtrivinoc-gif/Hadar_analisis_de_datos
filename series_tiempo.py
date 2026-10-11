"""
series_tiempo.py — Palabras clave que trabajan sobre UNA serie en el tiempo
(sin Qt, se puede probar sin abrir la ventana):

  proyección, tendencia, estacionalidad, umbral, saturación, agotamiento, escenario.

Todas parten igual: tomar una tabla con una columna de fecha (y opcionalmente
un valor y una "entidad": máquina, producto, sucursal…), juntar las filas por
período (hora, día, semana o mes, detectado solo) y trabajar sobre esa serie.
No asume ningún dominio: un congelador, un router de casino o las ventas de
una tienda son, para Hadar, una serie de fechas y valores.

Cómo se proyecta (sin librerías nuevas, solo numpy/pandas):
  1. Se prueban varios métodos simples (repetir lo último, promedio reciente,
     tendencia, tendencia suavizada, con o sin patrón estacional).
  2. Se comparan «a ciegas»: se simula que estamos en el pasado, se predice y
     se compara contra lo que de verdad pasó (varios puntos del historial).
  3. Gana el más simple entre los que rinden parecido. Se mide cuánto mejora
     frente a «repetir lo último que pasó» (la respuesta tonta de siempre).
  4. El rango sale de cuánto se equivocó ese método en las pruebas pasadas,
     no de una fórmula teórica. Se informa qué fracción de los casos pasados
     cayó dentro del rango (si es baja, se avisa).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

try:
    from .prediccion import _a_datetime, _tokens, es_columna_id, normalizar_clave, tipo_columna
    from .pronostico_base import (DIAS_SEMANA, MESES, UNIDADES, ResultadoPronostico, confianza_por_datos,
                                  dias_txt, duracion_txt, fecha_txt, leer_fechas, num_txt, pct_auto, pct_txt)
except ImportError:  # pruebas sueltas fuera del paquete
    from prediccion import _a_datetime, _tokens, es_columna_id, normalizar_clave, tipo_columna
    from pronostico_base import (DIAS_SEMANA, MESES, UNIDADES, ResultadoPronostico, confianza_por_datos,
                                 dias_txt, duracion_txt, fecha_txt, leer_fechas, num_txt, pct_auto, pct_txt)

MIN_PERIODOS = 12
MAX_PERIODOS = {"H": 2000, "D": 3000, "W": 1000, "M": 600}
MAX_AJUSTE_HOLT = 600          # puntos con que se ajusta la tendencia suavizada
K_ORIGENES = 16                # cuántos «puntos del pasado» se usan para probar
MIN_ORIGENES_HONESTOS = 8      # con menos, elegir y evaluar se hace con las mismas pruebas (se avisa)
MIN_PERIODOS_ESTACION = {"H": 72, "D": 21, "W": 104, "M": 24}
BASE_ESTACION = {"H": 24, "D": 7, "W": 52, "M": 12}
HORIZONTE_DEFECTO = {"H": 48, "D": 30, "W": 12, "M": 6}
HORIZONTE_UMBRAL = {"H": 72, "D": 90, "W": 26, "M": 12}
HORIZONTE_AGOTAMIENTO = {"H": 24 * 60, "D": 365, "W": 104, "M": 36}
VENTANA_CONSUMO = {"H": 168, "D": 56, "W": 13, "M": 12}
MAX_ENTIDADES = 3000
N_SIMULACIONES = 600

_PISTAS_NIVEL = {"stock", "nivel", "saldo", "temperatura", "temp", "precio", "promedio", "tasa", "porcentaje",
                 "humedad", "presion", "presión", "voltaje", "latencia", "ocupacion", "ocupación", "saturacion"}
_PISTAS_PICO = {"uso", "carga", "ocupacion", "ocupación", "trafico", "tráfico", "conexiones", "usuarios",
                "latencia", "cpu", "memoria", "ancho", "banda", "aforo", "cola"}

CANDIDATOS = [
    ("ultimo", None), ("promedio", None), ("tendencia", None), ("tendencia_larga", None), ("holt", None),
    ("estacional", None),
    ("ultimo", "aditiva"), ("promedio", "aditiva"), ("tendencia", "aditiva"), ("tendencia_larga", "aditiva"),
    ("holt", "aditiva"),
    ("ultimo", "multiplicativa"), ("promedio", "multiplicativa"), ("tendencia", "multiplicativa"),
    ("tendencia_larga", "multiplicativa"), ("holt", "multiplicativa"),
]
_NOMBRE_BASE = {
    "ultimo": "repetir el último valor", "promedio": "promedio reciente",
    "tendencia": "tendencia lineal reciente", "tendencia_larga": "tendencia lineal de todo el historial",
    "holt": "tendencia suavizada",
    "estacional": "repetir el mismo período del ciclo anterior",
}


def _g(cfg, nombre, defecto=None):
    v = getattr(cfg, nombre, None)
    return defecto if v is None else v


def nombre_modelo(base, est) -> str:
    n = _NOMBRE_BASE[base]
    if est:
        n += " + patrón estacional"
    return n


# ----------------------------------------------------------------------
# Preparar la serie (juntar filas por período)
# ----------------------------------------------------------------------
@dataclass
class SerieTiempo:
    fechas: pd.DatetimeIndex
    y: np.ndarray
    freq: str                    # H | D | W | M
    agregacion: str              # suma | promedio | ultimo | max | min | cuenta
    m: int | None                # largo del ciclo usable (7 días, 12 meses…), None si no hay datos para verlo
    nombre: str
    relleno: float               # fracción de períodos sin datos que se rellenó
    advertencias: list = field(default_factory=list)
    explicacion: str = ""


def detectar_frecuencia(dt: pd.Series) -> str:
    f = dt.dropna()
    if len(f) < 3:
        return "D"
    span = (f.max() - f.min()).total_seconds() / 86400.0
    if span < 3:
        return "H"
    dias = f.dt.normalize().drop_duplicates().sort_values()
    if len(dias) / max(span, 1.0) >= 0.5:
        return "D"
    d = np.diff(dias.to_numpy("datetime64[s]")).astype("float64") / 86400.0
    med = float(np.median(d)) if len(d) else 1.0
    if med < 3.5:
        return "D"
    if med < 20:
        return "W"
    return "M"


def _bucket(dt: pd.Series, freq: str) -> pd.Series:
    if freq == "H":
        return dt.dt.floor("h")
    n = dt.dt.normalize()
    if freq == "D":
        return n
    if freq == "W":
        return n - pd.to_timedelta(dt.dt.dayofweek, unit="D")
    return dt.dt.to_period("M").dt.to_timestamp()


def _paso(freq: str):
    return {"H": pd.Timedelta(hours=1), "D": pd.Timedelta(days=1), "W": pd.Timedelta(days=7)}.get(freq)


def _indice(inicio, fin, freq: str) -> pd.DatetimeIndex:
    if freq == "M":
        return pd.date_range(inicio, fin, freq="MS")
    return pd.date_range(inicio, fin, freq=_paso(freq))


def _futuras(ultimo, h: int, freq: str) -> pd.DatetimeIndex:
    if freq == "M":
        return pd.date_range(ultimo + pd.DateOffset(months=1), periods=h, freq="MS")
    paso = _paso(freq)
    return pd.date_range(ultimo + paso, periods=h, freq=paso)


def _siguiente(t, freq: str):
    return t + pd.DateOffset(months=1) if freq == "M" else t + _paso(freq)


def _agregar(dt: pd.Series, vals, freq: str, agg: str) -> pd.Series:
    orden = np.argsort(dt.to_numpy(), kind="stable")
    dt = dt.iloc[orden]
    clave = _bucket(dt, freq).to_numpy()
    if agg == "cuenta" or vals is None:
        s = pd.Series(1.0, index=np.arange(len(dt))).groupby(clave).sum()
    else:
        v = pd.Series(np.asarray(vals)[orden], index=np.arange(len(dt)))
        fn = {"suma": "sum", "promedio": "mean", "ultimo": "last", "max": "max", "min": "min"}[agg]
        s = v.groupby(clave).agg(fn)
    s.index = pd.DatetimeIndex(s.index)
    return s.astype("float64")


def _rellenar(s: pd.Series, freq: str, agg: str, con_filas=None):
    """Completa los períodos que faltan. Un período donde NO hay ninguna fila vale cero si se suma o se cuenta
    (no hubo ventas ese día); uno donde hay filas pero sin valor legible es un dato faltante y se interpola."""
    idx = _indice(s.index.min(), s.index.max(), freq)
    s = s.reindex(idx)
    vacio_con_filas = np.zeros(len(idx), dtype=bool)
    if con_filas is not None:
        vacio_con_filas = idx.isin(con_filas) & s.isna().to_numpy()
    faltan = int(s.isna().sum())
    if agg in ("suma", "cuenta"):
        s = s.where(~(s.isna() & ~vacio_con_filas), 0.0)
        s[vacio_con_filas] = np.nan
        s = s.interpolate(limit_area="inside")
    elif agg == "ultimo":
        s = s.ffill()
    else:
        s = s.interpolate(limit_area="inside")
    s = s.ffill().bfill()
    return s, faltan / max(len(idx), 1)


def agregacion_auto(col_valor, defecto: str = "suma") -> str:
    if col_valor is None:
        return "cuenta"
    toks = set(_tokens(col_valor))
    if toks & _PISTAS_NIVEL:
        return "promedio"
    if toks & _PISTAS_PICO:
        return "max"
    return defecto


def periodo_estacional(freq: str, n: int):
    base = BASE_ESTACION[freq]
    return base if n >= MIN_PERIODOS_ESTACION[freq] else None


def _filtrar_entidad(df, col_entidad, entidad):
    if col_entidad and col_entidad in df.columns and entidad not in (None, ""):
        ok = (normalizar_clave(df[col_entidad]) == str(entidad)).fillna(False).to_numpy(dtype=bool)
        if not ok.any():
            raise ValueError(f"No encontré «{entidad}» en la columna «{col_entidad}».")
        df = df[ok]
    return df


_VERBO = {"suma": "Se sumó", "promedio": "Se promedió", "ultimo": "Se tomó el último valor de",
          "max": "Se tomó el máximo de", "min": "Se tomó el mínimo de", "cuenta": "Se contó la cantidad de filas"}


def preparar_serie(df: pd.DataFrame, col_fecha, col_valor=None, agregacion: str = "auto",
                   frecuencia: str = "auto", col_entidad=None, entidad=None,
                   agregacion_defecto: str = "suma") -> SerieTiempo:
    if not col_fecha or col_fecha not in df.columns:
        raise ValueError("Elige la columna de fecha.")
    if col_valor is not None and col_valor not in df.columns:
        raise ValueError(f"No existe la columna «{col_valor}».")
    df = _filtrar_entidad(df, col_entidad, entidad)
    dt = leer_fechas(df[col_fecha], col_fecha)
    ok = dt.notna().to_numpy()
    avisos_ini = []
    freq_pre = None
    if not ok.all():
        avisos_ini.append(f"{(~ok).sum()} filas ({(~ok).mean() * 100:.0f}%) no tenían una fecha legible y no se usaron.")
    dt = dt[ok].reset_index(drop=True)
    dt_todas = dt
    vals = None
    if col_valor is not None:
        v = pd.to_numeric(df[col_valor], errors="coerce").to_numpy(dtype="float64")[ok]
        vals = v
    agg = agregacion_auto(col_valor, agregacion_defecto) if agregacion in (None, "auto") else agregacion
    if col_valor is None:
        agg = "cuenta"
    if vals is not None:
        buenos = np.isfinite(vals)
        if not buenos.any():
            raise ValueError(f"«{col_valor}» no tiene números.")
        if (~buenos).mean() > 0.1:
            avisos_ini.append(f"{(~buenos).mean() * 100:.0f}% de los valores de «{col_valor}» no eran números y no se usaron.")
        dt, vals = dt[buenos].reset_index(drop=True), vals[buenos]
    freq = detectar_frecuencia(dt) if frecuencia in (None, "auto") else frecuencia
    con_filas = pd.DatetimeIndex(_bucket(dt_todas, freq).unique())
    s = _agregar(dt, vals, freq, agg)
    avisos = list(avisos_ini)
    # último período a medio llenar (solo importa si se suma o se cuenta)
    if agg in ("suma", "cuenta") and freq in ("W", "M") and len(s) > 1:
        fin = _siguiente(s.index[-1], freq)
        if dt.max().normalize() < fin - pd.Timedelta(days=3):
            s = s.iloc[:-1]
            avisos.append("Se dejó fuera el último período porque está incompleto (sumarlo daría un valor engañosamente bajo).")
    s = s.iloc[-MAX_PERIODOS[freq]:]
    s, relleno = _rellenar(s, freq, agg, con_filas)
    if len(s) < MIN_PERIODOS:
        uno, varios = UNIDADES[freq]
        raise ValueError(f"Hay solo {len(s)} {varios} de historial; se necesitan al menos {MIN_PERIODOS} "
                         f"para poder decir algo con sentido.")
    if relleno > 0.2:
        avisos.append(f"El {relleno * 100:.0f}% de los períodos no tenía datos y se rellenó: "
                      f"los resultados pueden ser poco fiables.")
    y = s.to_numpy(dtype="float64")
    m = periodo_estacional(freq, len(y))
    nombre = str(col_valor) if col_valor is not None else "cantidad de registros"
    uno, varios = UNIDADES[freq]
    expl = f"{_VERBO[agg]} «{col_valor}» por {uno}." if col_valor is not None else f"{_VERBO[agg]} por {uno}."
    return SerieTiempo(fechas=s.index, y=y, freq=freq, agregacion=agg, m=m, nombre=nombre,
                       relleno=relleno, advertencias=avisos, explicacion=expl)


def serie_desde_cfg(df, cfg, agregacion_defecto: str = "suma") -> SerieTiempo:
    return preparar_serie(df, _g(cfg, "columna_fecha"), _g(cfg, "columna_valor"), _g(cfg, "agregacion", "auto"),
                          _g(cfg, "frecuencia", "auto"), _g(cfg, "columna_entidad"), _g(cfg, "entidad"),
                          agregacion_defecto=agregacion_defecto)


# ----------------------------------------------------------------------
# Estacionalidad
# ----------------------------------------------------------------------
def _tendencia_centrada(y: np.ndarray, m: int) -> np.ndarray:
    return pd.Series(y).rolling(m, center=True, min_periods=max(2, m // 2)).mean().to_numpy()


def fuerza_estacional(y: np.ndarray, m) -> float:
    """0 = sin patrón repetitivo, 1 = el patrón explica casi todo lo que sube y baja."""
    if not m or len(y) < 2 * m:
        return 0.0
    resto = y - _tendencia_centrada(y, m)
    ok = np.isfinite(resto)
    if ok.sum() < 2 * m:
        return 0.0
    pos = np.arange(len(y)) % m
    med = np.zeros(m)
    for p in range(m):
        sel = ok & (pos == p)
        if sel.any():
            med[p] = resto[sel].mean()
    r = resto - med[pos]
    vt = float(np.var(resto[ok]))
    if vt <= 0:
        return 0.0
    return float(np.clip(1 - np.var(r[ok]) / vt, 0.0, 1.0))


def _indices(y: np.ndarray, m: int, mult: bool):
    if mult and (np.any(y <= 0)):
        return None
    tend = _tendencia_centrada(y, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = y / tend if mult else y - tend
    pos = np.arange(len(y)) % m
    idx = np.full(m, 1.0 if mult else 0.0)
    for p in range(m):
        sel = (pos == p) & np.isfinite(d)
        if sel.any():
            idx[p] = d[sel].mean()
    if not np.all(np.isfinite(idx)):
        return None
    if mult:
        mu = idx.mean()
        if mu <= 0:
            return None
        idx = idx / mu
        if np.any(idx <= 0.05):
            return None
    else:
        idx = idx - idx.mean()
    return idx


# ----------------------------------------------------------------------
# Métodos de proyección
# ----------------------------------------------------------------------
_GRID_HOLT = [(a, b, p) for a in (0.05, 0.15, 0.4, 0.8) for b in (0.005, 0.03, 0.15) for p in (0.9, 0.98, 1.0)]


def _holt_filtrar(yl: list, alpha: float, beta: float, phi: float):
    n = len(yl)
    k = min(3, n - 1)
    nivel = yl[0]
    pend = (yl[k] - yl[0]) / max(k, 1)
    sse = 0.0
    for t in range(1, n):
        pred = nivel + phi * pend
        e = yl[t] - pred
        sse += e * e
        nuevo = alpha * yl[t] + (1 - alpha) * pred
        pend = beta * (nuevo - nivel) + (1 - beta) * phi * pend
        nivel = nuevo
    return nivel, pend, sse


def _holt(yy: np.ndarray, h: int) -> np.ndarray:
    yl = yy[-MAX_AJUSTE_HOLT:].tolist()
    if len(yl) < 4:
        return np.repeat(yy[-1], h)
    mejor = None
    for alpha, beta, phi in _GRID_HOLT:
        nivel, pend, sse = _holt_filtrar(yl, alpha, beta, phi)
        if mejor is None or sse < mejor[0]:
            mejor = (sse, nivel, pend, phi)
    _, nivel, pend, phi = mejor
    pot = np.cumsum(phi ** np.arange(1, h + 1))
    return nivel + pend * pot


def _base(yy: np.ndarray, h: int, base: str, m) -> np.ndarray:
    n = len(yy)
    if base == "ultimo":
        return np.repeat(yy[-1], h)
    if base == "promedio":
        k = min(n, max(m or 0, 6))
        return np.repeat(yy[-k:].mean(), h)
    if base in ("tendencia", "tendencia_larga"):
        if base == "tendencia":
            w = min(n, max(14, 4 * (m or 4), n // 3))
        else:
            w = min(n, 500)
        seg = yy[-w:]
        x = np.arange(w, dtype=float)
        xm = x.mean()
        sxx = ((x - xm) ** 2).sum()
        b = ((x - xm) * (seg - seg.mean())).sum() / sxx if sxx > 0 else 0.0
        a = seg.mean() - b * xm
        return (a + b * (w - 1)) + b * np.arange(1, h + 1)
    return _holt(yy, h)


def _pronosticar(y: np.ndarray, h: int, m, base: str, est):
    n = len(y)
    if base == "estacional":
        if not m or n < m:
            return None
        return np.array([y[n - m + (k % m)] for k in range(h)], dtype="float64")
    if est:
        if not m or n < 2 * m:
            return None
        mult = est == "multiplicativa"
        idx = _indices(y, m, mult)
        if idx is None:
            return None
        pos = np.arange(n) % m
        yy = y / idx[pos] if mult else y - idx[pos]
        f = _base(yy, h, base, m)
        posf = (n + np.arange(h)) % m
        return f * idx[posf] if mult else f + idx[posf]
    return _base(y, h, base, m)


@dataclass
class Proyeccion:
    freq: str
    fechas: pd.DatetimeIndex
    central: np.ndarray
    bajo: np.ndarray
    alto: np.ndarray
    sigma: np.ndarray
    sigma_acum: np.ndarray
    modelo: tuple
    modelo_nombre: str
    scores: dict
    mae_modelo: float
    mae_base: float
    mejora: float
    cobertura: float
    n_origenes: int
    h_eval: int
    honesta: bool
    nivel: float
    z: float
    no_negativa: bool


def _backtest(y: np.ndarray, h: int, m):
    n = len(y)
    h_eval = int(max(1, min(h, n // 5)))
    ultimo = n - h_eval
    cand = list(CANDIDATOS)
    usa_est = m is not None
    c_min = 8
    if usa_est:
        c_est = max(8, 2 * m)
        if ultimo - c_est + 1 >= 4:
            c_min = c_est
        else:
            usa_est = False
    if not usa_est:
        cand = [c for c in cand if c[1] is None and c[0] != "estacional"]
    origenes = np.arange(c_min, ultimo + 1)
    if len(origenes) > K_ORIGENES:
        origenes = np.unique(np.linspace(c_min, ultimo, K_ORIGENES).round().astype(int))
    if len(origenes) < 2:
        return None
    errores = {}
    for c in cand:
        E = np.full((len(origenes), h_eval), np.nan)
        for i, o in enumerate(origenes):
            f = _pronosticar(y[:o], h_eval, m if usa_est else None, c[0], c[1])
            if f is not None and np.all(np.isfinite(f)):
                E[i] = y[o:o + h_eval] - f
        if np.all(np.isfinite(E)):
            errores[c] = E
    if not errores:
        return None
    return errores, origenes, h_eval, usa_est


def _ajuste_potencia(valores: np.ndarray, b_min: float, b_max: float):
    """sigma(k) ≈ a·k^b ajustado por mínimos cuadrados en log-log (b acotado). Suaviza el ruido de
    tener pocas pruebas por horizonte. Devuelve una función de k o None si no hay base para ajustar."""
    j = np.arange(1, len(valores) + 1, dtype=float)
    ok = valores > 0
    if ok.sum() < 4:
        return None
    X, Y = np.log(j[ok]), np.log(valores[ok])
    b = float(np.clip(np.polyfit(X, Y, 1)[0], b_min, b_max))
    a = float(np.mean(Y - b * X))
    return lambda k: np.exp(a + b * np.log(np.asarray(k, dtype=float)))


def _factor_t(nivel: float, nu: float) -> float:
    """Cuánto ensanchar el rango porque sigma se estimó con pocas pruebas (distribución t en vez de normal)."""
    z = NormalDist().inv_cdf(0.5 + nivel / 2)
    try:
        from scipy.stats import t as _t
        q = float(_t.ppf(0.5 + nivel / 2, max(nu, 1.0)))
    except Exception:
        nu = max(nu, 1.0)
        q = z + (z ** 3 + z) / (4 * nu)
    return float(np.clip(q / z, 1.0, 2.5))


def proyectar(serie: SerieTiempo, h: int, nivel: float = 0.8) -> Proyeccion:
    y = serie.y
    n = len(y)
    h = int(max(h, 1))
    bt = _backtest(y, h, serie.m)
    z = NormalDist().inv_cdf(0.5 + nivel / 2)
    no_neg = bool(np.all(y >= 0))
    if bt is None:
        # muy pocos datos para probar: repetir lo último con un rango amplio basado en la variación reciente
        c = np.repeat(y[-1], h)
        dif = np.diff(y)
        s1 = float(np.std(dif)) if len(dif) else float(np.std(y))
        sig = s1 * np.sqrt(np.arange(1, h + 1))
        sig_ac = np.cumsum(sig)
        lo, hi = c - z * sig, c + z * sig
        if no_neg:
            lo = np.maximum(lo, 0)
        return Proyeccion(serie.freq, _futuras(serie.fechas[-1], h, serie.freq), c, lo, hi, sig, sig_ac,
                          ("ultimo", None), nombre_modelo("ultimo", None), {}, float("nan"), float("nan"), 0.0,
                          float("nan"), 0, 0, False, nivel, z, no_neg)
    errores, origenes, h_eval, usa_est = bt
    K = len(origenes)
    honesta = K >= MIN_ORIGENES_HONESTOS
    sel = slice(0, K // 2) if honesta else slice(0, K)
    ev = slice(K // 2, K) if honesta else slice(0, K)
    orden = [c for c in CANDIDATOS if c in errores]
    scores = {c: float(np.sqrt(np.mean(errores[c][sel] ** 2))) for c in orden}
    mejor = min(scores.values())
    elegibles = [c for c in orden if scores[c] <= mejor * 1.03 + 1e-12]
    elegido = elegibles[0]                          # el más simple (orden de CANDIDATOS)
    E = errores[elegido][ev]
    mae_modelo = float(np.sqrt(np.mean(E ** 2)))
    base_c = [c for c in (("ultimo", None), ("estacional", None)) if c in errores]
    mae_base = min(float(np.sqrt(np.mean(errores[c][ev] ** 2))) for c in base_c)
    mejora = float(np.clip(1 - mae_modelo / mae_base, 0, 1)) if mae_base > 0 else 0.0
    sig_h = np.sqrt(np.mean(E ** 2, axis=0))
    sig_h = np.maximum.accumulate(np.maximum(sig_h, 1e-12))
    ac = np.sqrt(np.mean(np.cumsum(E, axis=1) ** 2, axis=0))
    ac = np.maximum.accumulate(np.maximum(ac, 1e-12))
    ks = np.arange(1, h + 1)
    K_e = E.shape[0]
    orig_ev = origenes[ev]
    k_eff = max(2.0, min(float(K_e), float(orig_ev[-1] - orig_ev[0]) / h_eval + 1.0)) if len(orig_ev) > 1 else 2.0
    f_sig = _ajuste_potencia(sig_h, 0.0, 0.6) if h_eval >= 5 else None
    f_ac = _ajuste_potencia(ac, 0.5, 1.5) if h_eval >= 5 else None
    if f_sig is not None:
        sigma = np.maximum.accumulate(f_sig(ks))
    else:
        sigma = np.where(ks <= h_eval, sig_h[np.minimum(ks, h_eval) - 1], sig_h[-1] * np.sqrt(ks / h_eval))
    if f_ac is not None:
        sigma_ac = np.maximum.accumulate(f_ac(ks))
    else:
        sigma_ac = np.where(ks <= h_eval, ac[np.minimum(ks, h_eval) - 1], ac[-1] * (ks / h_eval) ** 0.75)
    sigma_ac = sigma_ac * _factor_t(nivel, k_eff) * 1.1   # el total acumulado se equivoca más de lo que muestran pocas pruebas
    cobertura = float(np.mean(np.abs(E) <= z * sigma[np.minimum(np.arange(h_eval), h - 1)][None, :]))
    f = _pronosticar(y, h, serie.m if usa_est else None, elegido[0], elegido[1])
    if f is None or not np.all(np.isfinite(f)):
        elegido = ("ultimo", None)
        f = np.repeat(y[-1], h)
    lo, hi = f - z * sigma, f + z * sigma
    if no_neg:
        f, lo, hi = np.maximum(f, 0), np.maximum(lo, 0), np.maximum(hi, 0)
    nombres = {nombre_modelo(*c): s for c, s in scores.items()}
    return Proyeccion(serie.freq, _futuras(serie.fechas[-1], h, serie.freq), f, lo, hi, sigma, sigma_ac, elegido,
                      nombre_modelo(*elegido), nombres, mae_modelo, mae_base, mejora, cobertura, K, h_eval,
                      honesta, nivel, z, no_neg)


def _confianza_proy(p: Proyeccion, n: int):
    if p.n_origenes < 4 or n < 24:
        return "baja", "hay pocos datos para comprobar el método contra lo que ya pasó."
    if p.mejora >= 0.25 and p.cobertura >= 0.65 and n >= 60:
        return "alta", (f"el método reduce el error en {p.mejora * 100:.0f}% frente a repetir lo último, y el rango "
                        f"acertó en {p.cobertura * 100:.0f}% de las pruebas pasadas.")
    if p.mejora >= 0.08:
        return "media", (f"el método mejora {p.mejora * 100:.0f}% a repetir lo último; el rango acertó en "
                         f"{p.cobertura * 100:.0f}% de las pruebas pasadas.")
    return "baja", ("el método casi no mejora a repetir lo último que pasó: la serie se comporta como ruido "
                    "y el rango es ancho.")


def _avisos_proyeccion(serie: SerieTiempo, p: Proyeccion, h: int) -> list:
    av = list(serie.advertencias)
    if not np.isnan(p.cobertura) and p.cobertura < 0.65:
        av.append(f"El rango acertó solo en {p.cobertura * 100:.0f}% de las pruebas pasadas (debería ser cerca de "
                  f"{p.nivel * 100:.0f}%): tómalo como orientativo.")
    if not p.honesta and p.n_origenes:
        av.append("Hay pocas pruebas pasadas, así que el método se eligió y evaluó con los mismos casos: "
                  "las cifras de calidad pueden ser algo optimistas.")
    if h > len(serie.y) / 2:
        av.append("Estás mirando más lejos que la mitad del historial disponible: a mayor distancia, menos fiable.")
    return av


# ----------------------------------------------------------------------
# Gráfico (datos para que la pantalla dibuje)
# ----------------------------------------------------------------------
def grafico_serie(serie: SerieTiempo, proy: Proyeccion | None = None, umbral=None, ultimos: int | None = None,
                  linea=None) -> dict:
    k = ultimos or min(len(serie.y), max(60, 3 * (len(proy.central) if proy else 20)))
    k = min(k, len(serie.y))
    g = {"tipo": "serie", "x_hist": serie.fechas[-k:].to_numpy(), "y_hist": serie.y[-k:],
         "etiqueta_y": serie.nombre, "umbral": umbral, "freq": serie.freq}
    if proy is not None:
        g.update({"x_fut": proy.fechas.to_numpy(), "y_fut": proy.central, "y_bajo": proy.bajo, "y_alto": proy.alto})
    if linea is not None:
        g["linea"] = linea
    return g


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _tabla_proyeccion(p: Proyeccion) -> pd.DataFrame:
    return pd.DataFrame({"Fecha": p.fechas, "Esperado": p.central, "Mínimo probable": p.bajo,
                         "Máximo probable": p.alto})


# ----------------------------------------------------------------------
# Palabra clave: PROYECCIÓN
# ----------------------------------------------------------------------
def analizar_proyeccion(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    serie = serie_desde_cfg(df, cfg)
    h = int(_g(cfg, "horizonte", 0)) or HORIZONTE_DEFECTO[serie.freq]
    nivel = float(_g(cfg, "nivel_intervalo", 0.8))
    p = proyectar(serie, h, nivel)
    y = serie.y
    acumula = serie.agregacion in ("suma", "cuenta")
    ht = duracion_txt(h, serie.freq)
    rango_pct = f"{nivel * 100:.0f}%"
    if acumula:
        total = float(p.central.sum())
        w = p.z * float(p.sigma_acum[-1])
        lo, hi = total - w, total + w
        if p.no_negativa:
            lo = max(lo, 0.0)
        previo = float(y[-h:].sum()) if len(y) >= h else float("nan")
        titular = (f"«{serie.nombre}» en los próximos {ht}: entre {num_txt(lo)} y {num_txt(hi)} en total "
                   f"(lo más probable: {num_txt(total)}).")
        cambio = (total / previo - 1) if previo and np.isfinite(previo) and previo != 0 else float("nan")
        tarj = [(num_txt(total), "Total esperado", f"suma de los próximos {ht}, rango {num_txt(lo)} a {num_txt(hi)}"),
                (pct_txt(cambio, 0, True), "Frente a igual período anterior",
                 f"comparado con los últimos {ht} reales ({num_txt(previo)})")]
    else:
        fin, lo, hi = float(p.central[-1]), float(p.bajo[-1]), float(p.alto[-1])
        titular = (f"«{serie.nombre}» dentro de {ht}: entre {num_txt(lo)} y {num_txt(hi)} "
                   f"(lo más probable: {num_txt(fin)}).")
        ref = float(y[-1])
        cambio = (fin / ref - 1) if ref not in (0.0,) and np.isfinite(ref) else float("nan")
        tarj = [(num_txt(fin), "Valor esperado", f"dentro de {ht}, rango {num_txt(lo)} a {num_txt(hi)}"),
                (pct_txt(cambio, 0, True), "Frente a hoy", f"el último valor real fue {num_txt(ref)}")]
    conf, motivo = _confianza_proy(p, len(y))
    tarj.append((pct_txt(p.mejora), "Mejora sobre repetir lo último",
                 "cuánto error evita el método frente a la respuesta más simple"))
    det = [serie.explicacion + f" Se usó el método «{p.modelo_nombre}».",
           f"Rango de {rango_pct}: según cuánto se equivocó el método en {p.n_origenes} pruebas hechas sobre el pasado."]
    return ResultadoPronostico(
        "proyeccion", "Proyección", titular, "info", conf, motivo, tarj, det,
        ["El comportamiento reciente se mantiene; no se conocen promociones, cierres, cambios de precio ni otros eventos futuros."],
        _avisos_proyeccion(serie, p, h), _tabla_proyeccion(p), "Proyección período a período",
        grafico_serie(serie, p), {"modelo": p.modelo_nombre, "mejora": p.mejora, "cobertura": p.cobertura,
                                  "error": p.mae_modelo, "error_base": p.mae_base, "periodos": len(y)})


# ----------------------------------------------------------------------
# Palabra clave: TENDENCIA
# ----------------------------------------------------------------------
def _ols(yy: np.ndarray):
    n = len(yy)
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = float(((x - xm) ** 2).sum())
    b = float(((x - xm) * (yy - yy.mean())).sum() / sxx) if sxx > 0 else 0.0
    a = float(yy.mean() - b * xm)
    return a, b, sxx


def analizar_tendencia(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    serie = serie_desde_cfg(df, cfg)
    y, fechas = serie.y, serie.fechas
    ventana = int(_g(cfg, "ventana", 0))
    if 0 < ventana < len(y):
        y, fechas = y[-ventana:], fechas[-ventana:]
    n = len(y)
    if n < MIN_PERIODOS:
        raise ValueError(f"Con {n} períodos no se puede hablar de tendencia; se necesitan al menos {MIN_PERIODOS}.")
    avisos = list(serie.advertencias)
    yy = y.copy()
    m = periodo_estacional(serie.freq, n)
    fe = fuerza_estacional(y, m) if m else 0.0
    sin_est = False
    if m and fe >= 0.15 and n >= 2 * m:
        idx = _indices(y, m, False)
        if idx is not None:
            yy = y - idx[np.arange(n) % m]
            sin_est = True
    a, b, sxx = _ols(yy)
    resid = yy - (a + b * np.arange(n))
    s2 = float((resid ** 2).sum() / max(n - 2, 1))
    se = sqrt(s2 / sxx) if sxx > 0 else float("inf")
    rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if n > 3 and np.std(resid) > 0 else 0.0
    rho = float(np.clip(rho if np.isfinite(rho) else 0.0, 0.0, 0.95))
    se_adj = se * sqrt((1 + rho) / (1 - rho))
    b_lo, b_hi = b - 1.96 * se_adj, b + 1.96 * se_adj
    ref = float(np.mean(np.abs(yy))) or 1.0
    uno, varios = UNIDADES[serie.freq]
    ini, fin = a, a + b * (n - 1)
    cambio_total = (fin - ini) / abs(ini) if abs(ini) > 1e-12 * ref and abs(ini) > 0.05 * ref else float("nan")
    if b_lo > 0:
        sentido = "sube"
    elif b_hi < 0:
        sentido = "baja"
    else:
        sentido = "estable"
    h = int(_g(cfg, "horizonte", 0)) or HORIZONTE_DEFECTO[serie.freq]
    por_per = b / ref
    ver = {"sube": "SUBE", "baja": "BAJA", "estable": "no muestra una tendencia clara"}[sentido]
    if sentido == "estable":
        titular = (f"«{serie.nombre}» {ver}: las variaciones de los últimos {n} {varios} no se distinguen "
                   f"de lo normal (cambio por {uno}: entre {pct_auto(b_lo / ref, True)} y {pct_auto(b_hi / ref, True)}).")
    else:
        titular = (f"«{serie.nombre}» {ver} en promedio {pct_auto(abs(por_per))} por {uno} "
                   f"(entre {pct_auto(min(abs(b_lo), abs(b_hi)) / ref)} y {pct_auto(max(abs(b_lo), abs(b_hi)) / ref)}); "
                   f"en los {n} {varios} analizados pasó de ≈{num_txt(ini)} a ≈{num_txt(fin)}.")
    w = max(min(h, n // 2), 1)
    if m and w >= m:
        w = (w // m) * m          # comparar ciclos completos (semanas enteras, años enteros)
    suma = serie.agregacion in ("suma", "cuenta")
    ult, ant = y[-w:], y[-2 * w:-w]
    if len(ant) == w:
        ref_c = float(ant.sum() if suma else ant.mean())
        act_c = float(ult.sum() if suma else ult.mean())
        rec = act_c / ref_c - 1 if ref_c != 0 else float("nan")
    else:
        rec = float("nan")
    tarj = [(pct_auto(por_per, True), f"Cambio por {uno}", f"ritmo promedio; rango {pct_auto(b_lo / ref, True)} a {pct_auto(b_hi / ref, True)}"),
            (pct_txt(cambio_total, 0, True), "Cambio en el período", f"de ≈{num_txt(ini)} a ≈{num_txt(fin)} según la línea de tendencia"),
            (pct_txt(rec, 0, True), "Últimos vs anteriores", f"últimos {duracion_txt(w, serie.freq)} frente a los {duracion_txt(w, serie.freq)} anteriores")]
    futuro = b * h
    det = [serie.explicacion,
           (f"Si sigue igual, en {duracion_txt(h, serie.freq)} más cambiaría unos {num_txt(futuro)} "
            f"(entre {num_txt(b_lo * h)} y {num_txt(b_hi * h)}).")]
    if sin_est:
        det.append("Antes de medir la tendencia se quitó el patrón repetitivo (estacional) para no confundirlo con un cambio.")
    if rho > 0.5:
        avisos.append("Los valores consecutivos se parecen mucho entre sí, así que el rango se amplió para no sobreestimar la certeza.")
    conf = confianza_por_datos(n, 24, 60)
    xs = np.arange(n)
    linea = (fechas.to_numpy(), a + b * xs)
    g = grafico_serie(SerieTiempo(fechas, y, serie.freq, serie.agregacion, serie.m, serie.nombre, serie.relleno),
                      ultimos=n, linea=linea)
    nivel = "info"
    return ResultadoPronostico(
        "tendencia", "Tendencia", titular, nivel, conf,
        f"se analizaron {n} {varios}; una tendencia se confirma cuando el rango no incluye el cero.", tarj, det,
        ["La tendencia es una línea recta ajustada al período analizado; no anticipa cambios de rumbo."],
        avisos, None, "", g, {"pendiente": b, "pendiente_min": b_lo, "pendiente_max": b_hi, "sentido": sentido,
                                "cambio_por_periodo": por_per, "periodos": n})


# ----------------------------------------------------------------------
# Palabra clave: ESTACIONALIDAD
# ----------------------------------------------------------------------
_CICLOS = {"D": [("semana", 7)], "H": [("día", 24), ("semana", 168)], "W": [("año", 52)], "M": [("año", 12)]}


def _clave_calendario(fechas: pd.DatetimeIndex, freq: str, mm: int):
    if freq == "D":
        return fechas.dayofweek.to_numpy()
    if freq == "H":
        return fechas.hour.to_numpy() if mm == 24 else (fechas.dayofweek * 24 + fechas.hour).to_numpy()
    if freq == "M":
        return (fechas.month - 1).to_numpy()
    return (fechas.isocalendar().week.to_numpy().astype(int) - 1)


def _etiqueta_pos(freq: str, mm: int, k: int) -> str:
    if freq == "D":
        return DIAS_SEMANA[k]
    if freq == "H":
        return f"{k:02d}:00" if mm == 24 else f"{DIAS_SEMANA[k // 24]} {k % 24:02d}:00"
    if freq == "M":
        return MESES[k]
    return f"semana {k + 1}"


def analizar_estacionalidad(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    serie = serie_desde_cfg(df, cfg)
    y, n = serie.y, len(serie.y)
    mejores = None
    for nombre_ciclo, mm in _CICLOS[serie.freq]:
        if n < 2 * mm:
            continue
        f = fuerza_estacional(y, mm)
        if mejores is None or f > mejores[0]:
            mejores = (f, nombre_ciclo, mm)
    if mejores is None:
        uno, varios = UNIDADES[serie.freq]
        raise ValueError(f"Hay {n} {varios} de historial: no alcanza para ver cuántas veces se repite un patrón "
                         f"(se necesitan al menos dos ciclos completos).")
    fuerza, nombre_ciclo, mm = mejores
    mult = bool(np.all(y > 0))
    tend = _tendencia_centrada(y, mm)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = y / tend if mult else y - tend
    clave = _clave_calendario(serie.fechas, serie.freq, mm)
    filas = []
    for k in np.unique(clave):
        sel = (clave == k) & np.isfinite(d)
        if sel.any():
            filas.append((int(k), float(d[sel].mean()), int(sel.sum())))
    if not filas:
        raise ValueError("No se pudo calcular el patrón con estos datos.")
    ind = np.array([f[1] for f in filas])
    ind = ind / ind.mean() if mult else ind - ind.mean()
    tabla = pd.DataFrame({"Posición": [_etiqueta_pos(serie.freq, mm, f[0]) for f in filas],
                          "Efecto": (ind - 1.0) if mult else ind, "Observaciones": [f[2] for f in filas]})
    tabla["Efecto"] = tabla["Efecto"].astype(float)
    alto, bajo = int(np.argmax(ind)), int(np.argmin(ind))
    ef = (lambda v: pct_txt(v - 1.0, 0, True)) if mult else (lambda v: num_txt(v))
    nivel_txt = ("fuerte" if fuerza >= 0.6 else "moderado" if fuerza >= 0.3 else "débil" if fuerza >= 0.1 else "casi inexistente")
    avisos = list(serie.advertencias)
    # ¿el patrón se mantiene en el tiempo? (primera mitad vs segunda mitad)
    estable = None
    if n >= 4 * mm:
        mitad = n // 2
        pats = []
        for sl in (slice(0, mitad), slice(mitad, n)):
            dd, cc = d[sl], clave[sl]
            pats.append([np.nanmean(dd[cc == f[0]]) if np.any((cc == f[0]) & np.isfinite(dd)) else np.nan for f in filas])
        a1, a2 = np.array(pats[0]), np.array(pats[1])
        ok = np.isfinite(a1) & np.isfinite(a2)
        if ok.sum() >= 3 and np.std(a1[ok]) > 0 and np.std(a2[ok]) > 0:
            estable = float(np.corrcoef(a1[ok], a2[ok])[0, 1])
    if fuerza < 0.1:
        titular = f"No se ve un patrón repetitivo claro cada {nombre_ciclo} en «{serie.nombre}»."
    else:
        titular = (f"Hay un patrón {nivel_txt} cada {nombre_ciclo}: lo más alto es {tabla['Posición'][alto]} "
                   f"({ef(ind[alto])}) y lo más bajo {tabla['Posición'][bajo]} ({ef(ind[bajo])}) frente al promedio.")
    tarj = [(pct_txt(fuerza), "Intensidad del patrón", "qué parte de lo que sube y baja se explica por el calendario"),
            (f"{tabla['Posición'][alto]}", "Lo más alto", f"{ef(ind[alto])} frente al promedio"),
            (f"{tabla['Posición'][bajo]}", "Lo más bajo", f"{ef(ind[bajo])} frente al promedio")]
    det = [serie.explicacion, f"Ciclo analizado: cada {nombre_ciclo} ({mm} períodos), sobre {n} períodos de historial."]
    if estable is not None:
        det.append("El patrón se mantiene igual en la primera y la segunda mitad del historial." if estable >= 0.7
                   else "El patrón cambió entre la primera y la segunda mitad del historial: úsalo con cuidado.")
        if estable < 0.7:
            avisos.append("El patrón estacional no es estable en el tiempo.")
    g = {"tipo": "barras", "x": list(tabla["Posición"]), "y": tabla["Efecto"].to_numpy(),
         "etiqueta_y": "efecto frente al promedio", "relativo": mult}
    return ResultadoPronostico(
        "estacionalidad", "Estacionalidad", titular, "info", confianza_por_datos(n // mm, 3, 6),
        f"se vieron {n // mm} ciclos completos; con 6 o más el patrón es sólido.", tarj, det,
        ["El patrón es el promedio de todos los ciclos; un ciclo puntual puede diferir."], avisos,
        tabla, "Efecto de cada posición del ciclo", g, {"fuerza": fuerza, "ciclo": nombre_ciclo, "periodos": n})


# ----------------------------------------------------------------------
# Palabra clave: UMBRAL y SATURACIÓN
# ----------------------------------------------------------------------
def _prob_pasado(central, sigma, limite, sube: bool) -> np.ndarray:
    s = np.maximum(sigma, 1e-12)
    z = (limite - central) / s
    p = np.array([_phi(v) for v in z])
    return (1.0 - p) if sube else p


def _primer_k(mask: np.ndarray):
    return int(np.argmax(mask)) if mask.any() else None


def analizar_umbral(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    serie = serie_desde_cfg(df, cfg)
    lim = _g(cfg, "umbral", None)
    if lim is None:
        raise ValueError("Indica el límite (umbral) que quieres vigilar.")
    lim = float(lim)
    y = serie.y
    h = int(_g(cfg, "horizonte", 0)) or HORIZONTE_UMBRAL[serie.freq]
    nivel = float(_g(cfg, "nivel_intervalo", 0.8))
    p = proyectar(serie, h, nivel)
    direccion = _g(cfg, "direccion", "auto")
    sube = (y[-1] < lim) if direccion == "auto" else direccion == "sube"
    verbo = "supere" if sube else "baje de"
    pasado = (lambda v: v >= lim) if sube else (lambda v: v <= lim)
    actual = float(y[-1])
    ya = bool(pasado(actual))
    uno, varios = UNIDADES[serie.freq]
    conf, motivo = _confianza_proy(p, len(y))
    avisos = _avisos_proyeccion(serie, p, h)
    probs = _prob_pasado(p.central, p.sigma, lim, sube)
    p_final, p_max = float(probs[-1]), float(probs.max())
    if sube:
        k_c, k_pronto, k_tarde = (_primer_k(p.central >= lim), _primer_k(p.alto >= lim), _primer_k(p.bajo >= lim))
    else:
        k_c, k_pronto, k_tarde = (_primer_k(p.central <= lim), _primer_k(p.bajo <= lim), _primer_k(p.alto <= lim))
    f = lambda k: fecha_txt(p.fechas[k]) if k is not None else "—"
    d = lambda k: duracion_txt(k + 1, serie.freq) if k is not None else "—"
    if ya:
        seguidos = 0
        for v in y[::-1]:
            if pasado(v):
                seguidos += 1
            else:
                break
        titular = (f"«{serie.nombre}» ya {'superó' if sube else 'bajó de'} el límite {num_txt(lim)} "
                   f"(valor actual {num_txt(actual)}); lleva {duracion_txt(seguidos, serie.freq)} seguidos así.")
        nivel_u = "alerta"
        tarj = [(num_txt(actual), "Valor actual", f"el límite es {num_txt(lim)}"),
                (duracion_txt(seguidos, serie.freq), "Tiempo pasado", "períodos seguidos más allá del límite"),
                (pct_txt(p_final), "Seguirá así", f"probabilidad de seguir pasado dentro de {duracion_txt(h, serie.freq)}")]
    else:
        if k_c is None and k_pronto is None:
            titular = (f"No se espera que «{serie.nombre}» {verbo} {num_txt(lim)} en los próximos "
                       f"{duracion_txt(h, serie.freq)} (probabilidad más alta: {pct_txt(p_max)}).")
        elif k_c is None:
            titular = (f"Lo más probable es que «{serie.nombre}» NO {verbo} {num_txt(lim)} en {duracion_txt(h, serie.freq)}, "
                       f"pero en el peor caso razonable podría ocurrir desde {f(k_pronto)} (probabilidad máxima {pct_txt(p_max)}).")
        else:
            titular = (f"«{serie.nombre}» {verbo} {num_txt(lim)} entre {f(k_pronto)} y "
                       f"{f(k_tarde) if k_tarde is not None else 'después del horizonte'} "
                       f"(lo más probable: {f(k_c)}, en ≈{d(k_c)}).")
        nivel_u = "alerta" if p_max >= 0.5 else "atencion" if p_max >= 0.2 else "info"
        tarj = [(f(k_c) if k_c is not None else "—", "Fecha más probable",
                 f"cuando el valor esperado cruza {num_txt(lim)}" if k_c is not None else f"no cruza en {duracion_txt(h, serie.freq)}"),
                (f"{f(k_pronto)} – {f(k_tarde) if k_tarde is not None else 'más tarde'}", "Rango de fechas",
                 f"del caso más temprano al más tardío razonable ({nivel * 100:.0f}%)"),
                (pct_txt(p_max), "Probabilidad máxima", f"de estar más allá del límite en algún período de los próximos {duracion_txt(h, serie.freq)}")]
    tabla = pd.DataFrame({"Fecha": p.fechas, "Esperado": p.central, "Mínimo probable": p.bajo,
                          "Máximo probable": p.alto, "Prob. de pasar el límite": probs})
    det = [serie.explicacion + f" Método: «{p.modelo_nombre}».",
           f"La probabilidad de cada período sale de cuánto se equivocó el método en {p.n_origenes} pruebas sobre el pasado."]
    return ResultadoPronostico(
        "umbral", "Umbral", titular, nivel_u, conf, motivo, tarj, det,
        ["La serie sigue comportándose como hasta ahora; no considera eventos futuros ni acciones que se tomen."],
        avisos, tabla, "Probabilidad de pasar el límite, período a período",
        grafico_serie(serie, p, umbral=lim),
        {"umbral": lim, "ya_pasado": ya, "prob_max": p_max, "prob_final": p_final, "modelo": p.modelo_nombre})


def analizar_saturacion(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    cap = _g(cfg, "capacidad", None)
    if cap is None:
        cap = _g(cfg, "umbral", None)
    if cap is None:
        raise ValueError("Indica la capacidad máxima (el valor desde el cual se satura).")
    cap = float(cap)
    serie = serie_desde_cfg(df, cfg, agregacion_defecto="max")
    y = serie.y
    h = int(_g(cfg, "horizonte", 0)) or HORIZONTE_UMBRAL[serie.freq]
    nivel = float(_g(cfg, "nivel_intervalo", 0.8))
    p = proyectar(serie, h, nivel)
    probs = _prob_pasado(p.central, p.sigma, cap, True)
    conf, motivo = _confianza_proy(p, len(y))
    avisos = _avisos_proyeccion(serie, p, h)
    k50 = _primer_k(probs >= 0.5)
    k20 = _primer_k(probs >= 0.2)
    pmax = float(probs.max())
    kmax = int(np.argmax(probs))
    hist_sobre = float(np.mean(y >= cap))
    uso = float(y[-1]) / cap if cap else float("nan")
    ht = duracion_txt(h, serie.freq)
    if float(y[-1]) >= cap:
        titular = (f"Ya se está superando la capacidad: el último valor fue {num_txt(y[-1])} ({pct_txt(uso)} de {num_txt(cap)}) "
                   f"y {pct_txt(hist_sobre)} de los períodos del historial estuvo en o sobre ese nivel.")
        nivel_u = "alerta"
    elif k50 is not None:
        titular = (f"Se espera saturación (≥50% de probabilidad) desde {fecha_txt(p.fechas[k50])}; "
                   f"la capacidad es {num_txt(cap)} y el uso proyectado llega a ≈{num_txt(p.central[kmax])}.")
        nivel_u = "alerta"
    elif k20 is not None:
        titular = (f"Hay riesgo de saturación desde {fecha_txt(p.fechas[k20])} (≥20% de probabilidad); "
                   f"el punto más riesgoso es {fecha_txt(p.fechas[kmax])} con {pct_txt(pmax)}.")
        nivel_u = "atencion"
    else:
        titular = (f"No se espera saturación en los próximos {ht}: la probabilidad más alta es {pct_txt(pmax)} "
                   f"({fecha_txt(p.fechas[kmax])}).")
        nivel_u = "info"
    tarj = [(pct_txt(uso), "Uso actual", f"{num_txt(y[-1])} de {num_txt(cap)} de capacidad"),
            (pct_txt(pmax), "Riesgo máximo", f"probabilidad más alta de pasar la capacidad en {ht}"),
            (pct_txt(hist_sobre), "Veces que ya se saturó", "de los períodos del historial estuvo en o sobre la capacidad")]
    det = [serie.explicacion + f" Método: «{p.modelo_nombre}».",
           f"Para saturación suele ser más útil mirar el PICO de cada período que el promedio (por eso se usa «{serie.agregacion}» salvo que indiques otra cosa)."]
    tabla = pd.DataFrame({"Fecha": p.fechas, "Uso esperado": p.central, "Máximo probable": p.alto,
                          "Prob. de saturar": probs, "% de la capacidad": p.central / cap if cap else np.nan})
    return ResultadoPronostico(
        "saturacion", "Saturación", titular, nivel_u, conf, motivo, tarj, det,
        ["La demanda sigue creciendo o variando como hasta ahora; no considera ampliaciones de capacidad."],
        avisos, tabla, "Probabilidad de saturación, período a período", grafico_serie(serie, p, umbral=cap),
        {"capacidad": cap, "prob_max": pmax, "hist_sobre": hist_sobre, "modelo": p.modelo_nombre})


# ----------------------------------------------------------------------
# Palabra clave: AGOTAMIENTO (stock, batería, presupuesto, espacio…)
# ----------------------------------------------------------------------
def _pos_calendario(fechas, freq: str):
    if freq == "D":
        return pd.DatetimeIndex(fechas).dayofweek.to_numpy(), 7
    if freq == "H":
        return pd.DatetimeIndex(fechas).hour.to_numpy(), 24
    return None, None


def _simular_agotamiento(y: np.ndarray, stock: float, hmax: int, pos_hist, pos_fut, m, rng, n_sim: int):
    """Tiempo (en períodos, con fracción) hasta gastar `stock`, repetido n_sim veces
    sacando consumos al azar de lo ocurrido recientemente (respetando el día de la semana / hora)."""
    if m and pos_hist is not None and len(y) >= 3 * m:
        draws = np.empty((n_sim, hmax))
        pools = {p: y[pos_hist == p] for p in range(m)}
        for k in range(hmax):
            pool = pools.get(int(pos_fut[k]))
            if pool is None or len(pool) == 0:
                pool = y
            draws[:, k] = rng.choice(pool, size=n_sim)
    else:
        draws = rng.choice(y, size=(n_sim, hmax))
    cum = np.cumsum(draws, axis=1)
    llego = cum >= stock
    alguna = llego.any(axis=1)
    j = llego.argmax(axis=1)
    T = np.full(n_sim, np.inf)
    filas = np.flatnonzero(alguna)
    if len(filas):
        jj = j[filas]
        previo = np.where(jj > 0, cum[filas, np.maximum(jj - 1, 0)], 0.0)
        d = np.maximum(draws[filas, jj], 1e-12)
        T[filas] = jj + (stock - previo) / d
    return T


def _cuantil(T: np.ndarray, q: float) -> float:
    Ts = np.sort(T)
    return float(Ts[min(int(np.ceil(q * len(Ts))) - 1, len(Ts) - 1) if q > 0 else 0])


def analizar_agotamiento(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    col_fecha = _g(cfg, "columna_fecha")
    col_cons = _g(cfg, "columna_valor")
    col_stock = _g(cfg, "columna_stock")
    col_ent = _g(cfg, "columna_entidad")
    stock_num = _g(cfg, "stock_actual", None)
    if not col_fecha or col_fecha not in df.columns:
        raise ValueError("Elige la columna de fecha.")
    if col_cons is None and col_stock is None:
        raise ValueError("Elige la columna de consumo (lo que se gasta por período) o la columna de stock.")
    if col_stock is None and stock_num is None:
        raise ValueError("Indica el stock actual (un número) o elige la columna de stock.")
    dt = leer_fechas(df[col_fecha], col_fecha)
    ok = dt.notna().to_numpy()
    base = pd.DataFrame({"dt": dt[ok].to_numpy()})
    base["ent"] = (normalizar_clave(df[col_ent]).astype(object).to_numpy()[ok] if col_ent and col_ent in df.columns else "total")
    if col_stock and col_stock in df.columns:
        base["stock"] = pd.to_numeric(df[col_stock], errors="coerce").to_numpy(dtype="float64")[ok]
    avisos = []
    if col_cons and col_cons in df.columns:
        base["consumo"] = pd.to_numeric(df[col_cons], errors="coerce").to_numpy(dtype="float64")[ok]
        derivado = False
    else:
        if "stock" not in base:
            raise ValueError("La columna de stock no existe.")
        base = base.sort_values(["ent", "dt"], kind="stable")
        dif = base.groupby("ent", sort=False)["stock"].diff()
        base["consumo"] = np.maximum(-dif, 0.0)
        base = base[dif.notna().to_numpy()]
        derivado = True
        avisos.append("No había columna de consumo: se calculó como lo que baja el stock entre registros "
                      "(las subidas se tomaron como reposiciones y no cuentan como consumo).")
    base = base[np.isfinite(base["consumo"].to_numpy())]
    base = base[base["ent"].notna()]
    if base.empty:
        raise ValueError("No quedaron filas con consumo válido.")
    if (base["consumo"] < 0).any():
        avisos.append("Hay consumos negativos (devoluciones o ajustes); se contaron como cero.")
        base["consumo"] = base["consumo"].clip(lower=0)
    dt_all = pd.Series(base["dt"].to_numpy())
    freq = _g(cfg, "frecuencia", "auto")
    freq = detectar_frecuencia(dt_all) if freq in (None, "auto") else freq
    base["per"] = _bucket(pd.Series(base["dt"].to_numpy()), freq).to_numpy()
    fin = base["per"].max()
    k_v = int(_g(cfg, "ventana", 0)) or VENTANA_CONSUMO[freq]
    ini = fin - (k_v - 1) * _paso(freq) if freq != "M" else fin - pd.DateOffset(months=k_v - 1)
    idx_v = _indice(max(ini, base["per"].min()), fin, freq)
    base = base[base["per"] >= idx_v[0]]
    piv = base.groupby(["ent", "per"])["consumo"].sum().unstack(fill_value=0.0).reindex(columns=idx_v, fill_value=0.0)
    # stock actual por entidad
    stocks = {}
    if "stock" in base.columns:
        ult = base.dropna(subset=["stock"]).sort_values("dt", kind="stable").groupby("ent", sort=False)["stock"].last()
        stocks = ult.to_dict()
    if stock_num is not None and (not col_ent or col_ent not in df.columns):
        stocks = {"total": float(stock_num)}
    if not stocks:
        raise ValueError("No encontré el stock actual: revisa la columna de stock.")
    if len(piv) > MAX_ENTIDADES:
        piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index[:MAX_ENTIDADES]]
        avisos.append(f"Había más de {MAX_ENTIDADES} elementos; se analizaron los {MAX_ENTIDADES} con más consumo.")
    # primer período con datos de cada entidad dentro de la ventana
    primer = base.groupby("ent")["per"].min()
    hcap = int(_g(cfg, "horizonte", 0)) or HORIZONTE_AGOTAMIENTO[freq]
    rng = np.random.default_rng(int(_g(cfg, "semilla", 0)))
    pos_h, m = _pos_calendario(idx_v, freq)
    uno, varios = UNIDADES[freq]
    ultima_fecha = _siguiente(pd.Timestamp(fin), freq)   # el stock actual es el de al final del último período
    n_sim = N_SIMULACIONES if len(piv) <= 300 else 300
    filas = []
    for ent, fila in piv.iterrows():
        stock = stocks.get(ent)
        if stock is None or not np.isfinite(stock):
            continue
        p0 = primer.get(ent, idx_v[0])
        sel = np.asarray(idx_v >= p0)
        yv = fila.to_numpy(dtype="float64")[sel]
        pos_v = pos_h[sel] if pos_h is not None else None
        if len(yv) == 0:
            continue
        media = float(yv.mean())
        tr = "estable"
        if len(yv) >= 9:
            t3 = len(yv) // 3
            p1, p3 = yv[:t3], yv[-t3:]
            a, b = float(p1.mean()), float(p3.mean())
            se = sqrt(float(p1.var()) / len(p1) + float(p3.var()) / len(p3))
            claro = abs(b - a) > 2.0 * max(se, 1e-12)
            if claro and a > 0 and b / a > 1.25:
                tr = "sube"
            elif claro and a > 0 and b / a < 0.75:
                tr = "baja"
            elif claro and a == 0 and b > 0:
                tr = "sube"
        reg = {"Elemento": ent, "Stock actual": float(stock), "Consumo medio por " + uno: media,
               "Tendencia del consumo": tr, "Períodos observados": len(yv)}
        if stock <= 0:
            reg.update({"Pronto": 0.0, "Central": 0.0, "Tarde": 0.0, "Cobertura": "ya agotado"})
        elif media <= 0:
            reg.update({"Pronto": np.inf, "Central": np.inf, "Tarde": np.inf, "Cobertura": "sin consumo reciente"})
        elif stock / media > hcap:
            reg.update({"Pronto": np.inf, "Central": np.inf, "Tarde": np.inf, "Cobertura": f"más de {duracion_txt(hcap, freq)}"})
        else:
            hmax = int(min(hcap, np.ceil(3 * stock / media) + 5))
            pos_f = None
            if m:
                fut = _futuras(pd.Timestamp(idx_v[-1]), hmax, freq)
                pos_f = _pos_calendario(fut, freq)[0]
            T = _simular_agotamiento(yv, float(stock), hmax, pos_v, pos_f, m, rng, n_sim)
            reg.update({"Pronto": _cuantil(T, 0.10), "Central": _cuantil(T, 0.50), "Tarde": _cuantil(T, 0.90),
                        "Cobertura": ""})
        filas.append(reg)
    if not filas:
        raise ValueError("No se pudo calcular el agotamiento: faltan datos de stock o de consumo.")
    res = pd.DataFrame(filas).sort_values("Central", kind="stable").reset_index(drop=True)

    def fecha_de(T):
        if not np.isfinite(T):
            return pd.NaT
        return _siguiente_n(ultima_fecha, T, freq)

    res["Fecha pronto"] = [fecha_de(v) for v in res["Pronto"]]
    res["Fecha probable"] = [fecha_de(v) for v in res["Central"]]
    res["Fecha tarde"] = [fecha_de(v) for v in res["Tarde"]]
    n_per = int(res["Períodos observados"].median())
    conf = confianza_por_datos(n_per, 14 if freq == "D" else 6, 42 if freq == "D" else 12)
    motivo = (f"se usó lo que se consumió en los últimos {n_per} {varios}; "
              f"con menos de {14 if freq == 'D' else 6} el consumo típico es incierto.")
    if (res["Tendencia del consumo"] == "sube").any():
        avisos.append("En algunos elementos el consumo viene subiendo: el cálculo asume que seguirá como en el período "
                      "reciente, así que podrían agotarse antes.")
    alerta_w = float(_g(cfg, "ventana_dias", 14))
    # ventana de alerta expresada en períodos
    w_per = {"H": alerta_w * 24, "D": alerta_w, "W": alerta_w / 7, "M": alerta_w / 30.4}[freq]
    urgentes = int((res["Pronto"] <= w_per).sum())
    primero = res.iloc[0]
    if len(res) == 1:
        r = primero
        if r["Cobertura"] == "ya agotado":
            titular = "El stock ya está agotado."
            nivel = "alerta"
        elif r["Cobertura"]:
            titular = f"Con el consumo reciente, el stock {r['Cobertura']} (alcanza para varios períodos): no se agota en el horizonte analizado."
            nivel = "info"
        else:
            titular = (f"Al ritmo reciente, el stock dura entre {duracion_txt(r['Pronto'], freq)} y {duracion_txt(r['Tarde'], freq)} "
                       f"(lo más probable: {duracion_txt(r['Central'], freq)}); se acabaría entre {fecha_txt(r['Fecha pronto'])} y "
                       f"{fecha_txt(r['Fecha tarde'])}.")
            nivel = "alerta" if r["Pronto"] <= w_per else "atencion" if r["Central"] <= 2 * w_per else "info"
        tarj = [(duracion_txt(r["Central"], freq) if np.isfinite(r["Central"]) else "—", "Duración probable", "lo más probable según el consumo reciente"),
                (f"{duracion_txt(r['Pronto'], freq)} – {duracion_txt(r['Tarde'], freq)}" if np.isfinite(r["Pronto"]) and np.isfinite(r["Tarde"]) else "—", "Rango", "del caso más rápido al más lento razonable (80%)"),
                (num_txt(r["Consumo medio por " + uno]), f"Consumo por {uno}", f"con {num_txt(r['Stock actual'])} de stock actual")]
    else:
        if np.isfinite(primero["Central"]):
            titular = (f"Primero se agota «{primero['Elemento']}»: entre {duracion_txt(primero['Pronto'], freq)} y "
                       f"{duracion_txt(primero['Tarde'], freq)}. {urgentes} de {len(res)} podrían agotarse en los próximos "
                       f"{num_txt(alerta_w)} días.")
        else:
            titular = f"Ninguno de los {len(res)} elementos se agota dentro del horizonte analizado."
        nivel = "alerta" if urgentes else "info"
        tarj = [(str(urgentes), "En riesgo pronto", f"podrían agotarse en menos de {num_txt(alerta_w)} días"),
                (str(len(res)), "Elementos analizados", "ordenados del que se acaba primero al último"),
                (duracion_txt(primero["Central"], freq) if np.isfinite(primero["Central"]) else "—", "El más urgente", str(primero["Elemento"]))]
    tabla = res.copy()
    for c in ("Pronto", "Central", "Tarde"):
        tabla = tabla.rename(columns={c: {"Pronto": f"Si se gasta rápido ({varios})", "Central": f"Lo más probable ({varios})",
                                         "Tarde": f"Si se gasta lento ({varios})"}[c]})
    det = [f"Se simuló {n_sim} veces cómo se consumiría el stock repitiendo, al azar, consumos de los últimos {k_v} {varios}"
           f"{' (respetando el día de la semana)' if m == 7 else ' (respetando la hora del día)' if m == 24 else ''}; el rango es del 10% más rápido al 10% más lento."]
    if derivado:
        det.append("El consumo se dedujo de las bajas de stock.")
    return ResultadoPronostico(
        "agotamiento", "Agotamiento", titular, nivel, conf, motivo, tarj, det,
        ["El consumo seguirá parecido al de los últimos períodos y no habrá reposiciones."], avisos, tabla,
        "Cuánto dura cada elemento", None, {"frecuencia": freq, "urgentes": urgentes, "elementos": len(res)})


def _siguiente_n(t: pd.Timestamp, n_periodos: float, freq: str):
    if freq == "M":
        meses = int(np.floor(n_periodos))
        base = t + pd.DateOffset(months=meses)
        return base + pd.Timedelta(days=(n_periodos - meses) * 30.4)
    return t + _paso(freq) * n_periodos


# ----------------------------------------------------------------------
# Palabra clave: ESCENARIO (qué pasa si cambio una variable)
# ----------------------------------------------------------------------
def _boot_bloques(Z, y, beta, resid, idx, B, rng):
    n = len(y)
    rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if n > 3 and np.std(resid) > 0 else 0.0
    rho = float(np.clip(rho if np.isfinite(rho) else 0.0, 0.0, 0.98))
    # bloques largos cuando los residuos son muy persistentes (si no, el rango sale demasiado angosto)
    L = int(np.clip(np.ceil(2.0 / (1.0 - rho)), max(2, round(n ** (1 / 3))), max(n // 4, 3)))
    nb = int(np.ceil(n / L))
    fit = Z @ beta
    out = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - L + 1, size=nb)
        r = np.concatenate([resid[s:s + L] for s in starts])[:n]
        bs = np.linalg.lstsq(Z, fit + r, rcond=None)[0]
        out[b] = bs[idx]
    return out


def analizar_escenario(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    col_x = _g(cfg, "columna_driver")
    if not col_x or col_x not in df.columns:
        raise ValueError("Elige la variable que quieres mover (por ejemplo, el precio o la publicidad).")
    cambio = float(_g(cfg, "cambio_pct", 10.0)) / 100.0
    if cambio <= -1:
        raise ValueError("Un cambio de -100% o menos no tiene sentido.")
    serie = serie_desde_cfg(df, cfg)
    n = len(serie.y)
    sdf = _filtrar_entidad(df, _g(cfg, "columna_entidad"), _g(cfg, "entidad"))
    dt = leer_fechas(sdf[_g(cfg, "columna_fecha")], _g(cfg, "columna_fecha"))
    ok = np.array(dt.notna().to_numpy())
    xv = np.array(pd.to_numeric(sdf[col_x], errors="coerce").to_numpy(dtype="float64"))
    ok &= np.isfinite(xv)
    if not ok.any():
        raise ValueError(f"«{col_x}» no tiene números.")
    agx = _g(cfg, "agregacion_driver", "promedio")
    sx = _agregar(dt[ok].reset_index(drop=True), xv[ok], serie.freq, agx)
    sx = sx.reindex(serie.fechas).interpolate(limit_area="inside").ffill().bfill()
    rez = int(_g(cfg, "rezago", 0))
    x = sx.to_numpy(dtype="float64")
    y = serie.y
    if rez > 0:
        x, y, fechas = x[:-rez], y[rez:], serie.fechas[rez:]
    else:
        fechas = serie.fechas
    n = len(y)
    if n < 20:
        raise ValueError(f"Hay solo {n} períodos para medir el efecto; se necesitan al menos 20.")
    if np.std(x) == 0:
        raise ValueError(f"«{col_x}» nunca cambió en el historial: no se puede saber qué efecto tiene.")
    loglog = bool(np.all(y > 0) and np.all(x > 0))
    yy = np.log(y) if loglog else y
    xx = np.log(x) if loglog else x
    cols = [np.ones(n), xx, (np.arange(n) - np.arange(n).mean()) / max(n, 1)]
    m = periodo_estacional(serie.freq, n)
    est = bool(m and fuerza_estacional(y, m) >= 0.15 and n >= 3 * m)
    if est:
        pos = np.arange(n) % m
        cols.extend([(pos == p).astype(float) for p in range(1, m)])
    Z = np.column_stack(cols)
    beta = np.linalg.lstsq(Z, yy, rcond=None)[0]
    resid = yy - Z @ beta
    rng = np.random.default_rng(int(_g(cfg, "semilla", 0)))
    bs = _boot_bloques(Z, yy, beta, resid, 1, 400, rng)
    m_ref = int(min(n, max(m or 4, 4)))
    x_ref = float(np.mean(x[-m_ref:]))
    y_ref = float(np.mean(y[-m_ref:]))
    if x_ref == 0:
        raise ValueError("El nivel reciente de la variable es cero: el cambio en % no se puede calcular.")
    x_new = x_ref * (1 + cambio)

    def efecto(b):
        if loglog:
            return (1 + cambio) ** b - 1
        return b * (x_new - x_ref) / y_ref if y_ref else float("nan")

    e0 = efecto(float(beta[1]))
    e_lo, e_hi = (float(np.nanpercentile([efecto(b) for b in bs], q)) for q in (2.5, 97.5))
    # cuánto aporta la variable (R² con y sin ella)
    def r2(Zm):
        bb = np.linalg.lstsq(Zm, yy, rcond=None)[0]
        ss = float(((yy - Zm @ bb) ** 2).sum())
        tot = float(((yy - yy.mean()) ** 2).sum())
        return 1 - ss / tot if tot > 0 else 0.0
    aporte = max(r2(Z) - r2(np.delete(Z, 1, axis=1)), 0.0)
    distingue = (e_lo > 0) or (e_hi < 0)
    avisos = list(serie.advertencias)
    avisos.append("Esto mide lo que pasó junto con esa variable en el historial; no prueba que una cosa cause la otra "
                  "(puede haber otros factores, como una temporada o una promoción, que cambiaron a la vez).")
    if x_new < float(np.min(x)) or x_new > float(np.max(x)):
        avisos.append("El escenario queda fuera del rango que esa variable tuvo en el historial: el resultado es una extrapolación.")
    if rez:
        avisos.append(f"Se midió el efecto con {rez} período(s) de retraso.")
    signo = "sube" if cambio >= 0 else "baja"
    if not distingue:
        titular = (f"Si «{col_x}» {signo} {abs(cambio) * 100:.0f}%, el efecto sobre «{serie.nombre}» no se distingue de cero con estos datos "
                   f"(entre {pct_txt(e_lo, 1, True)} y {pct_txt(e_hi, 1, True)}).")
        conf = "baja"
        motivo = "el rango incluye el cero: con el historial disponible no se puede afirmar que la variable tenga efecto."
    else:
        titular = (f"Si «{col_x}» {signo} {abs(cambio) * 100:.0f}%, «{serie.nombre}» cambiaría entre {pct_txt(e_lo, 1, True)} y "
                   f"{pct_txt(e_hi, 1, True)} (lo más probable: {pct_txt(e0, 1, True)}).")
        conf = "media" if n >= 30 else "baja"
        motivo = f"se midió sobre {n} períodos y el rango no incluye el cero; sigue siendo una asociación, no una prueba de causa."
    tarj = [(pct_txt(e0, 1, True), "Efecto más probable", f"sobre «{serie.nombre}» si «{col_x}» {signo} {abs(cambio) * 100:.0f}%"),
            (f"{pct_txt(e_lo, 1, True)} a {pct_txt(e_hi, 1, True)}", "Rango", "95% de los casos plausibles según los datos"),
            (pct_txt(aporte), "Qué tanto explica", "parte de las variaciones que se asocia a esta variable")]
    det = [serie.explicacion, ("Se usó una relación porcentaje-porcentaje (elasticidad)." if loglog
                               else "Se usó una relación lineal (la variable o el valor tienen ceros o negativos)."),
           f"Nivel de partida: «{col_x}» ≈ {num_txt(x_ref)} (promedio reciente) → {num_txt(x_new)}."]
    if loglog:
        det.append(f"Elasticidad estimada: {beta[1]:.2f} (si la variable sube 1%, el valor cambia {beta[1]:.2f}%).")
    return ResultadoPronostico(
        "escenario", "Escenario", titular, "info", conf, motivo, tarj, det,
        ["La relación observada en el pasado se mantiene y el resto de las condiciones no cambia."], avisos,
        None, "", None, {"efecto": e0, "efecto_min": e_lo, "efecto_max": e_hi, "aporte": aporte, "periodos": n,
                         "distingue_de_cero": distingue})
