"""
pronostico_base.py — Piezas comunes de las «palabras clave» de la pestaña
Predicción (proyección, tendencia, agotamiento, vida útil, etc.). Sin Qt.

La idea: todas las palabras clave devuelven el MISMO tipo de resultado
(`ResultadoPronostico`), así la pantalla solo necesita saber mostrar uno:
un titular con rango, tres tarjetas, una tabla, un gráfico y los avisos.

Reglas de honestidad que se mantienen en todas:
- Siempre un RANGO, nunca una cifra exacta ("entre 10 y 18 días").
- Siempre se dice qué se asumió y qué tan confiable es el resultado.
- Si los datos no alcanzan, se dice (no se inventa una respuesta).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from .avisos import ETIQUETA, ordenar_avisos
    from .formato import dec, get_formato_numerico, loc, set_formato_numerico
    from .prediccion import _a_datetime, tipo_columna
except ImportError:
    from avisos import ETIQUETA, ordenar_avisos
    from formato import dec, get_formato_numerico, loc, set_formato_numerico
    from prediccion import _a_datetime, tipo_columna

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

UNIDADES = {
    "H": ("hora", "horas"),
    "D": ("día", "días"),
    "B": ("día hábil", "días hábiles"),
    "W": ("semana", "semanas"),
    "M": ("mes", "meses"),
}


# El formato de los números (chileno / estándar) vive en formato.py: un solo lugar para todo Hadar.
_loc = loc


def fecha_txt(t, con_hora: bool = False) -> str:
    if t is None or (not isinstance(t, pd.Timestamp) and pd.isna(t)):
        return "—"
    t = pd.Timestamp(t)
    if pd.isna(t):
        return "—"
    s = f"{t.day} {MESES[t.month - 1]} {t.year}"
    if con_hora:
        s += f" {t.hour:02d}:{t.minute:02d}"
    return s


def num_txt(x, decimales: int | None = None) -> str:
    """Número legible: separador de miles y pocos decimales si es grande (según el formato elegido)."""
    return _loc(_num_estandar(x, decimales))


def _num_estandar(x, decimales: int | None = None) -> str:
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    if decimales is not None:
        return f"{x:,.{decimales}f}"
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:,.1f}".rstrip("0").rstrip(".")
    if a >= 1:
        return f"{x:,.2f}".rstrip("0").rstrip(".")
    return f"{x:.3g}"


def pct_txt(x, decimales: int = 0, signo: bool = False) -> str:
    """0.153 -> '15%'. Con signo=True: '+15%' / '-15%'."""
    if x is None or not np.isfinite(x):
        return "—"
    v = x * 100
    if round(v, decimales) == 0:
        v = 0.0                       # evita «-0%»
    s = f"{v:+.{decimales}f}%" if (signo and v != 0) else f"{v:.{decimales}f}%"
    return _loc(s)


def pct_auto(x, signo: bool = False) -> str:
    """Porcentaje con los decimales justos: 12% · 1.5% · 0.06%."""
    if x is None or not np.isfinite(x):
        return "—"
    a = abs(x) * 100
    dec = 0 if a >= 10 else 1 if a >= 1 else 2
    return pct_txt(x, dec, signo)


def duracion_txt(n, freq: str = "D") -> str:
    """3 -> '3 días' (según la frecuencia de la serie)."""
    if n is None or not np.isfinite(n):
        return "—"
    uno, varios = UNIDADES.get(freq, UNIDADES["D"])
    n = float(n)
    if abs(n - round(n)) < 0.05 or n >= 10:
        v = int(round(n))
        return f"{v} {uno if v == 1 else varios}"
    return _loc(f"{n:.1f} {varios}")


def dias_txt(d) -> str:
    """Días (número) en palabras simples: 0.5 -> '12 horas', 45 -> '45 días', 400 -> '13 meses'."""
    if d is None or not np.isfinite(d):
        return "—"
    d = float(d)
    if d < 1 / 24:
        mi = max(int(round(d * 1440)), 1)
        return f"{mi} minuto" + ("" if mi == 1 else "s")
    if d < 1:
        h = max(int(round(d * 24)), 1)
        return f"{h} hora" + ("" if h == 1 else "s")
    if d < 120:
        v = int(round(d))
        return f"{v} día" + ("" if v == 1 else "s")
    if d < 730:
        v = int(round(d / 30.4))
        return f"{v} meses"
    return _loc(f"{d / 365.25:.1f} años")


# Qué significa el número que entrega cada palabra clave (texto automático, en palabras simples)
TEXTO_TIPO_SALIDA = {
    "valor": "Qué es el número: un valor pronosticado (una estimación con su rango), no una cifra exacta.",
    "probabilidad": "Qué es el número: una probabilidad calculada con los casos observados (de cada 100 casos como este, "
                    "ocurre en aproximadamente ese número).",
    "ranking": "Qué es el número: un puntaje para ORDENAR de más a menos urgente; no es un porcentaje ni un monto real.",
}


@dataclass
class ResultadoPronostico:
    palabra: str                      # clave de la palabra clave («agotamiento», «umbral»…)
    titulo: str                       # nombre legible
    titular: str                      # la respuesta en una frase, con rango
    nivel: str = "info"               # urgencia de lo que dice: info | atencion | alerta
    confianza: str = "media"          # alta | media | baja
    por_que_confianza: str = ""
    tarjetas: list = field(default_factory=list)      # [(valor, titulo, explicacion)] (3 cifras clave)
    detalle: list = field(default_factory=list)       # frases que explican el resultado
    supuestos: list = field(default_factory=list)     # qué asumió Hadar
    advertencias: list = field(default_factory=list)
    tabla: pd.DataFrame | None = field(default=None, repr=False)
    tabla_titulo: str = ""
    grafico: dict | None = field(default=None, repr=False)   # ver series_tiempo.grafico_serie
    metricas: dict = field(default_factory=dict)
    experto: dict | None = field(default=None, repr=False)   # detalles técnicos (modo experto): filas, métodos, residuos
    tipo_salida: str = ""            # qué es el número que se entrega: valor | probabilidad | ranking (ver TEXTO_TIPO_SALIDA)

    def lineas_resumen(self) -> list:
        lineas = [self.titular]
        lineas.extend(self.detalle)
        if self.por_que_confianza:
            lineas.append(f"Confianza {self.confianza}: {self.por_que_confianza}")
        if self.supuestos:
            lineas.append("Qué asume Hadar: " + " ".join(self.supuestos))
        lineas.extend(f"{ETIQUETA[a.nivel]}: {a}" for a in self.avisos_ordenados())
        return lineas

    def avisos_ordenados(self) -> list:
        """Avisos con su nivel (crítico, importante, informativo), los críticos primero. Ver avisos.py."""
        return ordenar_avisos(self.advertencias)

    def texto(self) -> str:
        return "\n".join(self.lineas_resumen())


def confianza_por_datos(n_eventos_o_periodos: int, minimo: int, bueno: int) -> str:
    """'baja' bajo `minimo`, 'alta' desde `bueno`, 'media' entre medio."""
    if n_eventos_o_periodos < minimo:
        return "baja"
    if n_eventos_o_periodos >= bueno:
        return "alta"
    return "media"


def leer_fechas(serie: pd.Series, nombre: str, exigir_validas: bool = True) -> pd.Series:
    """Convierte una columna a fechas SIN zona horaria. Rechaza columnas numéricas (un número suelto no es una
    fecha: leerlo como tal daría 1970 y resultados absurdos) y columnas donde casi nada se pudo leer."""
    t = tipo_columna(serie)
    if t in ("numerica", "numerica_texto", "booleana"):
        raise ValueError(f"«{nombre}» tiene números, no fechas. Elige una columna de fecha (por ejemplo 2024-05-31).")
    dt = _a_datetime(serie)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    if exigir_validas and len(dt) and float(dt.notna().mean()) < 0.5:
        raise ValueError(f"No pude leer fechas en «{nombre}» (solo se entendió el {dec(dt.notna().mean() * 100, 0)}%).")
    return dt


# ----------------------------------------------------------------------
# Fecha de corte («como estaba el mundo en esa fecha») y aviso de datos antiguos
# ----------------------------------------------------------------------
UMBRAL_DATOS_ANTIGUOS_DIAS = 45


def leer_fecha_corte(txt):
    """Texto -> Timestamp (o None si está vacío). Error claro, en español, si no se entiende."""
    if txt is None or (isinstance(txt, str) and not txt.strip()):
        return None
    try:
        t = pd.Timestamp(txt)
    except (ValueError, TypeError):
        raise ValueError(f"No entendí la fecha de corte «{txt}». Escríbela así: 2024-05-31.")
    if pd.isna(t):
        raise ValueError(f"No entendí la fecha de corte «{txt}». Escríbela así: 2024-05-31.")
    return t.tz_localize(None) if t.tzinfo is not None else t


def aplicar_fecha_corte(df: pd.DataFrame, col_fecha, fecha_corte) -> pd.DataFrame:
    """Deja solo las filas con fecha hasta el corte (el día completo si el corte no trae hora)."""
    corte = leer_fecha_corte(fecha_corte)
    if corte is None or not col_fecha or col_fecha not in df.columns:
        return df
    lim = corte + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) if corte == corte.normalize() else corte
    dt = leer_fechas(df[col_fecha], col_fecha, exigir_validas=False)
    sub = df[(dt <= lim).to_numpy()]
    if len(sub) == 0:
        raise ValueError(f"No hay datos hasta la fecha de corte ({fecha_txt(corte)}). Elige una fecha de corte más reciente.")
    return sub


def aviso_datos_antiguos(df: pd.DataFrame, columnas, fecha_corte=None, hoy=None, umbral_dias=None):
    """Frase de aviso si el último dato está lejos de hoy (las fechas se cuentan desde el último dato, no desde hoy).
    Devuelve None si no hace falta."""
    hoy = pd.Timestamp(hoy) if hoy is not None else pd.Timestamp.now().normalize()
    corte = leer_fecha_corte(fecha_corte)
    ult = None
    if corte is None:
        for c in columnas:
            if c and c in df.columns:
                try:
                    m = leer_fechas(df[c], c, exigir_validas=False).max()
                except ValueError:
                    continue
                if pd.notna(m) and (ult is None or m > ult):
                    ult = m
    else:
        ult = corte
    if ult is None:
        return None
    dias = (hoy - ult.normalize()).days
    if dias <= (UMBRAL_DATOS_ANTIGUOS_DIAS if umbral_dias is None else umbral_dias):
        return None
    origen = "La fecha de corte es" if corte is not None else "El último dato de la tabla es del"
    return (f"{origen} {fecha_txt(ult)} (hace {dias} días): las fechas y plazos se cuentan desde ahí, no desde hoy. "
            f"Si ya pasó tiempo, lo que dice sobre «los próximos días» puede estar en el pasado.")