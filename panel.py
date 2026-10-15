"""
Pestaña "Panel": un vistazo rápido a los datos. Etapa 1: hasta 4
indicadores grandes (con variación y mini tendencia) + un gráfico principal.

Reutiliza lo que ya existe: los Indicadores de la pestaña Indicadores, el
recorte por fechas de la Línea de Tiempo y los colores del tema activo. La
parte de cálculo (funciones sueltas de arriba) no depende de Qt, así se
puede probar sola.
"""
import copy
import html
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer, QPointF, QObject, QEvent
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QComboBox, QMessageBox, QCheckBox, QTabWidget, QScrollArea, QApplication,
)

from .retorno import TITULAR_NIVEL
from .config import COLOR_ACCENT, COLOR_ACCENT_3
from .alarmas import (
    calcular_valor_actual, evaluar_regla, evaluar_columna, estado_en_palabras, lineas_filas,
)
from .indicadores import Indicador
from .linea_tiempo import anomalias_con_fecha
from .graficos import ChartPanel, DonutChartWidget, agrupar_por_categoria, recortar_con_otros, _FUNCIONES_AGG
from .filtro_filas import aplicar_filtro, describir_filtro, columnas_del_filtro

MAX_INDICADORES = 4
PUNTOS_TENDENCIA = 12
_PISTAS_FECHA = ("fecha", "date", "dia", "día", "time", "hora")
_PISTAS_VALOR = ("total", "monto", "importe", "final", "venta", "ingreso", "valor")

# Estilo de "subpestaña" (texto sin caja, con línea inferior índigo cuando está
# activo): el mismo de Preguntas / Entrenar un modelo en Predicción. También
# existe como QSS_SUBPESTANA en main_window.py; si se cambia uno, cambiar el otro.
QSS_SUBPESTANA = (
    "QPushButton{background: transparent; border: none; border-bottom: 2px solid transparent; "
    "border-radius: 0; padding: 6px 14px; font-weight: 600; color: #a1a1aa;} "
    "QPushButton:checked{color: #6366f1; border-bottom: 2px solid #6366f1;} "
    "QPushButton:hover{background: transparent; color: #6366f1;} "
    "QPushButton:pressed{background: transparent;} "
    "QPushButton:disabled{color: rgba(127,127,127,0.6);}"
)


# ----------------------------------------------------------------------------
# Cálculo (sin Qt)
# ----------------------------------------------------------------------------
def _es_columna_id(nombre):
    n = str(nombre).lower()
    return n == "id" or n.startswith("id_") or n.endswith("_id")


def _es_bandera(serie):
    """Columna que solo trae 0 y 1 (activo, anulada, es_venta_libre...):
    no sirve para sumar ni promediar como si fuera un monto."""
    vals = set(pd.unique(serie.dropna()))
    return bool(vals) and vals <= {0, 1, 0.0, 1.0, True, False}


def columnas_numericas_utiles(df, solo=None):
    """Columnas numéricas que sirven para sumar/promediar (sin ID ni
    banderas 0/1). `solo` limita a las columnas propias de la tabla: los
    datos que vienen de otra tabla se repiten en cada fila y, sumados, se
    inflarían."""
    if df is None:
        return []
    cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if not _es_columna_id(c) and not _es_bandera(df[c])]
    if solo is not None:
        cols = [c for c in cols if c in set(solo)]
    return cols


def elegir_columna_valor(df, solo=None):
    """La columna numérica más probable de ser 'el monto': una con nombre
    tipo total/monto/final, y si no hay, la primera que no sea un ID."""
    util = columnas_numericas_utiles(df, solo)
    if not util:
        return None
    for pista in _PISTAS_VALOR:
        for c in util:
            if pista in str(c).lower():
                return c
    return util[0]


# ----------------------------------------------------------------------------
# Combinar tablas relacionadas (sin Qt)
# ----------------------------------------------------------------------------
def _norm(texto):
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", t)


def _raiz_de_clave(col):
    c = str(col)
    if c.lower().endswith("_id") and len(c) > 3:
        return c[:-3]
    if c.lower().startswith("id_") and len(c) > 3:
        return c[3:]
    return None


def inferir_padres(tablas, min_cobertura=0.9):
    """{tabla_hija: [(columna_clave, tabla_padre, columna_en_padre)]}.

    Una columna 'venta_id' o 'id_venta' apunta a la tabla 'Ventas' solo si
    (1) el nombre coincide, (2) la columna del padre no repite valores y
    (3) casi todos los valores de la hija existen en el padre. Así una
    relación solo se acepta si los datos la confirman."""
    padres = {n: [] for n in tablas}
    norm = {n: _norm(n) for n in tablas}
    for hija, df_h in tablas.items():
        for col in df_h.columns:
            raiz = _raiz_de_clave(col)
            if not raiz:
                continue
            r = _norm(raiz)
            for padre, df_p in tablas.items():
                if padre == hija or norm[padre] not in (r, r + "s", r + "es"):
                    continue
                pk = next((k for k in (col, "id", f"id_{raiz}", f"{raiz}_id") if k in df_p.columns), None)
                if pk is None:
                    continue
                claves = df_p[pk]
                if claves.isna().any() or not claves.is_unique:
                    continue
                valores = df_h[col].dropna()
                if valores.empty:
                    continue
                try:
                    cobertura = float(valores.isin(claves).mean())
                except Exception:
                    continue
                if cobertura >= min_cobertura:
                    padres[hija].append((col, padre, pk))
                    break
    return padres


def alcanzables(base, padres, max_saltos=2):
    """Tablas a las que se llega desde `base` siguiendo claves hacia los
    'padres' (una venta -> su cliente, un detalle -> su venta -> su caja)."""
    orden, vistos, frontera = [], {base}, [base]
    for _ in range(max_saltos):
        nueva = []
        for t in frontera:
            for _fk, p, _pk in padres.get(t, []):
                if p not in vistos:
                    vistos.add(p)
                    orden.append(p)
                    nueva.append(p)
        frontera = nueva
    return orden


def _enriquecido(nombre, df, tablas, padres, permitidas, saltos, visitados):
    out, usadas = df, []
    if saltos <= 0:
        return out, usadas
    for fk, padre, pk in padres.get(nombre, []):
        if padre in visitados or (permitidas is not None and padre not in permitidas):
            continue
        sub, sub_usadas = _enriquecido(padre, tablas[padre], tablas, padres, permitidas,
                                       saltos - 1, visitados | {padre})
        traer = {}
        for c in sub.columns:
            propia = "." not in str(c)
            if c == pk or (propia and _es_columna_id(c)) or sub[c].isna().all():
                continue
            nuevo = f"{padre}.{c}" if propia else c
            if nuevo in out.columns or nuevo in traer.values():
                continue
            traer[c] = nuevo
        if not traer:
            continue
        sub_sel = sub[[pk] + list(traer)].rename(columns={pk: "__pk__", **traer})
        try:
            # m:1 -- cada fila de la tabla apunta a UNA fila del padre, así
            # el número de filas nunca cambia y los totales no se inflan.
            m = out.merge(sub_sel, how="left", left_on=fk, right_on="__pk__", validate="m:1")
        except Exception:
            continue
        if len(m) != len(out):
            continue
        m = m.drop(columns="__pk__")
        m.index = out.index
        out = m
        for u in [padre] + sub_usadas:
            if u not in usadas:
                usadas.append(u)
    return out, usadas


def combinar_con_padres(nombre, df, tablas, padres, permitidas=None, max_saltos=2):
    """(df con columnas 'Tabla.columna' traídas de las tablas conectadas,
    lista de tablas realmente usadas)."""
    return _enriquecido(nombre, df, tablas, padres, permitidas, max_saltos, {nombre})


def elegir_columna_fecha(df, preferida=None, fechas_de=None):
    """Devuelve (columna, serie_de_fechas) o (None, None). `preferida` es la
    columna que ya usa la Línea de Tiempo. `fechas_de(df, col)` permite
    reutilizar el interpretador (con caché) de la ventana principal."""
    if df is None or df.empty:
        return None, None

    def interpretar(col):
        if fechas_de is not None:
            try:
                f = fechas_de(df, col)
                return pd.Series(pd.to_datetime(np.asarray(f)), index=df.index)
            except Exception:
                pass
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return pd.Series(df[col].values, index=df.index)
        muestra = df[col].dropna().astype(str).head(20)
        iso = len(muestra) > 0 and muestra.str.match(r"^\d{4}-\d{2}-\d{2}").all()
        return pd.Series(pd.to_datetime(df[col], dayfirst=not iso, errors="coerce"), index=df.index)

    candidatas = []
    if preferida and preferida in df.columns:
        candidatas.append(preferida)
    for c in df.columns:
        if c in candidatas:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]) or any(p in str(c).lower() for p in _PISTAS_FECHA):
            candidatas.append(c)
    for col in candidatas:
        try:
            fechas = interpretar(col)
        except Exception:
            continue
        if fechas.notna().mean() >= 0.8:
            return col, fechas
    return None, None


_PERIODOS = [
    ("1min", "minuto", pd.Timedelta(minutes=1)),
    ("5min", "5 minutos", pd.Timedelta(minutes=5)),
    ("15min", "15 minutos", pd.Timedelta(minutes=15)),
    ("30min", "30 minutos", pd.Timedelta(minutes=30)),
    ("1h", "hora", pd.Timedelta(hours=1)),
    ("3h", "3 horas", pd.Timedelta(hours=3)),
    ("6h", "6 horas", pd.Timedelta(hours=6)),
    ("1D", "día", pd.Timedelta(days=1)),
    ("1W", "semana", pd.Timedelta(weeks=1)),
    ("MS", "mes", pd.Timedelta(days=30)),
]
_MAX_PUNTOS = 62


def elegir_periodo(inicio, fin):
    """El período más fino con el que el gráfico no pasa de ~62 puntos:
    minutos si los datos abarcan una hora, meses si abarcan años."""
    span = fin - inicio
    for regla, nombre, ancho in _PERIODOS:
        if span / ancho <= _MAX_PUNTOS:
            return regla, nombre
    return _PERIODOS[-1][0], _PERIODOS[-1][1]


def serie_por_periodo(valores, fechas, contar=False, funcion="sum"):
    """Suma de `valores` (o cantidad de filas si contar=True; o la función
    indicada: sum, mean, max, min) por período. Devuelve (serie, nombre)."""
    s = pd.Series(pd.to_numeric(valores, errors="coerce").values, index=pd.DatetimeIndex(fechas.values))
    s = s[~s.index.isna()]
    if s.empty:
        return s, "día"
    regla, nombre = elegir_periodo(s.index.min(), s.index.max())
    grupos = s.resample(regla)
    return (grupos.size() if contar else getattr(grupos, funcion)()), nombre


def _partes_de_tendencia(df, fechas, n=PUNTOS_TENDENCIA):
    """Parte `df` en n tramos: iguales en tiempo si hay fechas, o iguales en
    cantidad de filas (según el orden en que vienen) si no las hay."""
    if fechas is not None:
        t = fechas.values.astype("datetime64[ns]").astype("int64")
        ok = fechas.notna().values
        if ok.sum() < n:
            return []
        lo, hi = t[ok].min(), t[ok].max()
        if hi == lo:
            return []
        bordes = np.linspace(lo, hi, n + 1)
        pos = np.clip(np.digitize(t, bordes[1:-1]), 0, n - 1)
        return [df.iloc[np.where(ok & (pos == k))[0]] for k in range(n)]
    if len(df) < n * 2:
        return []
    return [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), n)]


def _sin_ventana(ind):
    """Copia del indicador sin la ventana 'últimos N días'. Para dibujar la
    evolución hay que mirar todo el período: con la ventana puesta, cada tramo
    se recortaría por su cuenta y la línea no diría nada."""
    if not getattr(ind, "ventana", None):
        return ind
    copia = copy.copy(ind)
    copia.ventana = None
    return copia


def tendencia_indicador(ind, df, fechas):
    """Valor del indicador en cada tramo, para la mini línea. [] si no hay
    suficientes datos para dibujar algo con sentido."""
    ind = _sin_ventana(ind)
    valores = []
    for parte in _partes_de_tendencia(df, fechas):
        v = ind.calcular(parte) if len(parte) else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        valores.append(v)
    validos = [v for v in valores if np.isfinite(v)]
    return valores if len(validos) >= 3 else []


def variacion_indicador(ind, df, fechas):
    """Cambio porcentual de la 2ª mitad del período contra la 1ª. Solo tiene
    sentido con fechas; sin ellas devuelve None."""
    ind = _sin_ventana(ind)
    if fechas is None or df is None or df.empty:
        return None
    ok = fechas.notna()
    if ok.sum() < 2:
        return None
    t0, t1 = fechas[ok].min(), fechas[ok].max()
    if t0 == t1:
        return None
    medio = t0 + (t1 - t0) / 2
    a = ind.calcular(df[np.asarray(ok & (fechas <= medio))])
    b = ind.calcular(df[np.asarray(ok & (fechas > medio))])
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(a) and np.isfinite(b)) or a == 0:
        return None
    return (b - a) / abs(a) * 100


_GRAVEDAD_POR_SEVERIDAD = {"crítico": 3, "grave": 3, "moderado": 2, "leve": 1}
_NOMBRE_ESCALA = {"hora": "hora", "dia": "día", "semana": "semana", "mes": "mes",
                  "trimestre": "trimestre", "anio": "año"}


def _gravedad(anomalia):
    """3 = alta, 2 = media, 1 = baja. Solo las anomalías de tipo
    'quiebre_patron' traen severidad; el resto queda en 1 (informativa)."""
    sev = str(anomalia.get("severidad_impacto") or "").lower()
    return _GRAVEDAD_POR_SEVERIDAD.get(sev, 1)


def columnas_ocultas_por_defecto(anomalias):
    """Mientras el usuario no elija, se ocultan los hallazgos que son solo
    sobre columnas ID (un ID 'atípico' casi nunca dice algo útil)."""
    cols = {c for a in anomalias or [] for c in (a.get("columnas") or [])}
    return {c for c in cols if _es_columna_id(c)}


def filtrar_anomalias(anomalias, ocultas=None):
    if ocultas is None:
        ocultas = columnas_ocultas_por_defecto(anomalias)
    resultado = []
    for a in anomalias or []:
        cols = a.get("columnas") or []
        if cols and all(c in ocultas for c in cols):
            continue
        resultado.append(a)
    return resultado


def hallazgos_principales(anomalias, maximo=3, largo=115):
    """Los pocos hallazgos que valen la pena ver de un vistazo: sin repetir
    (mismo tipo sobre las mismas columnas), los más graves primero, y con la
    descripción que ya escribió el detector, acortada."""
    mejores = {}
    for a in anomalias or []:
        texto = str(a.get("descripcion") or "").strip()
        if not texto:
            continue
        clave = (a.get("tipo"), tuple(a.get("columnas") or ()))
        if clave not in mejores or _gravedad(a) > _gravedad(mejores[clave]):
            mejores[clave] = a
    ordenados = sorted(mejores.values(), key=_gravedad, reverse=True)[:maximo]
    resultado = []
    for a in ordenados:
        t = str(a["descripcion"]).strip()
        corto = t if len(t) <= largo else t[: largo - 1].rstrip() + "…"
        resultado.append({"texto": corto, "completo": t, "gravedad": _gravedad(a)})
    return resultado


def marcadores_hallazgos(anomalias, df, col_fecha, dayfirst=True):
    """[(fecha, gravedad)] de las anomalías que caen dentro de las filas
    visibles, para marcarlas sobre la línea de tiempo."""
    if not anomalias or df is None or col_fecha is None:
        return []
    try:
        con_fecha = anomalias_con_fecha(anomalias, df, col_fecha, dayfirst)
    except Exception:
        return []
    return [(m["fecha"], _gravedad(m["anomalia"])) for m in con_fecha]


def indicadores_por_defecto(df, propias=None):
    """4 indicadores razonables cuando el usuario todavía no creó los suyos.
    A propósito no toca la lista de la pestaña Indicadores."""
    if df is None or df.empty:
        return []
    col = elegir_columna_valor(df, propias)
    resultado = []
    if col:
        resultado.append(Indicador(f"Total {col}", "suma", col, formato_tipo="entero"))
        resultado.append(Indicador(f"Promedio {col}", "promedio", col, formato_tipo="decimales"))
    primera = (propias or list(df.columns))[0]
    resultado.append(Indicador("Registros", "conteo", primera, formato_tipo="entero"))
    otras = [c for c in columnas_numericas_utiles(df, propias) if c != col]
    if otras:
        resultado.append(Indicador(f"Total {otras[0]}", "suma", otras[0], formato_tipo="entero"))
    return resultado[:MAX_INDICADORES]


# --- Gráficos propios (los que el usuario crea en la pestaña Gráficos) ------
MAX_GRAFICOS_PROPIOS = 4
GRAFICOS_SOPORTADOS = ("Barras", "Líneas", "Histograma", "Dona / Torta")
CLAVE_AUTO = "__auto__"   # el gráfico automático de Hadar, dentro de la lista de Personalizar


def clave_grafico(cfg):
    """Nombre con el que se identifica un gráfico propio: el que le puso el
    usuario o, si no tiene, 'Tipo — X / Y'."""
    if cfg.get("nombre"):
        return cfg["nombre"]
    clave = f"{cfg.get('tipo')} — {cfg.get('x')} / {cfg.get('y')}"
    solo = describir_filtro(cfg.get("filtro"))
    return f"{clave} (solo donde {solo})" if solo else clave


def datos_grafico_propio(cfg, df):
    """Prepara los datos de un gráfico propio (sin Qt, para poder probarlo).
    Devuelve un dict con 'modo' (fecha / categorias / histograma / dona) y
    los datos a dibujar, o {'error': texto} si no se puede dibujar."""
    tipo, x = cfg.get("tipo"), cfg.get("x")
    y = cfg.get("y") or "Conteo"
    if tipo not in GRAFICOS_SOPORTADOS:
        return {"error": f"El tipo '{tipo}' todavía no se puede mostrar en el Panel."}
    pedidas = [x, None if y == "Conteo" else y, *sorted(columnas_del_filtro(cfg.get("filtro")))]
    faltan = [c for c in pedidas if c and c not in df.columns]
    if faltan:
        return {"error": "Este gráfico usa columnas que no están en lo que ves ahora: "
                         + ", ".join(map(str, faltan))}
    df = aplicar_filtro(df, cfg.get("filtro"))
    if df.empty:
        return {"error": "No hay filas que cumplan el filtro de este gráfico."}
    funcion = _FUNCIONES_AGG.get(cfg.get("calcular"), "sum")
    top = int(cfg.get("top") or 0)

    if tipo == "Histograma":
        datos = pd.to_numeric(df[x], errors="coerce").dropna()
        if datos.empty:
            return {"error": "No hay datos numéricos para dibujar este gráfico."}
        cuentas, bordes = np.histogram(datos, bins=20)
        return {"modo": "histograma", "x": ((bordes[:-1] + bordes[1:]) / 2).tolist(),
                "y": cuentas.tolist(), "ancho": float(bordes[1] - bordes[0]) * 0.9}

    if tipo in ("Barras", "Líneas") and ChartPanel._es_columna_fecha(df, x):
        contar = (y == "Conteo")
        fechas = ChartPanel._parsear_fechas(df[x])
        serie, periodo = serie_por_periodo(df[x] if contar else df[y], fechas,
                                           contar=contar, funcion=funcion)
        if len(serie) >= 2:
            xs = serie.index.values.astype("datetime64[s]").astype("int64").astype(float)
            return {"modo": "fecha", "tipo": tipo, "x": xs.tolist(),
                    "y": serie.fillna(0).astype(float).tolist(), "detalle": f"por {periodo}"}

    orden = "valor" if tipo in ("Barras", "Dona / Torta") else "clave"
    etiquetas, valores = agrupar_por_categoria(df, x, y, funcion, orden)
    if not etiquetas:
        return {"error": "No hay datos para dibujar este gráfico."}
    if tipo == "Dona / Torta":
        e, v, _ = recortar_con_otros(etiquetas, valores, 8, funcion, incluir_otros=True)
        return {"modo": "dona", "etiquetas": e, "valores": v}
    if tipo == "Barras":
        n = top if 0 < top <= 15 else 15   # más de 15 barras no se leen en una tarjeta
        e, v, nota = recortar_con_otros(etiquetas, valores, n, funcion)
        return {"modo": "categorias", "tipo": tipo, "etiquetas": e, "valores": v,
                "detalle": f"top {n} de {nota['total']}" if nota else ""}
    return {"modo": "categorias", "tipo": tipo, "etiquetas": etiquetas[:60],
            "valores": valores[:60], "detalle": ""}


# ----------------------------------------------------------------------------
# Piezas de interfaz
# ----------------------------------------------------------------------------
def _es_tema_claro(colors):
    return QColor(colors["bg"]).lightness() > 140


def colores_estado(colors):
    """Verde-azulado para 'sube', rojo para 'baja', gris para 'estable'.
    En temas claros se usan tonos más oscuros para que se lean."""
    if _es_tema_claro(colors):
        return {"sube": "#0F766E", "baja": "#DC2626", "igual": colors["muted"]}
    return {"sube": "#2DD4BF", "baja": "#F87171", "igual": colors["muted"]}


def estilo_tarjeta(colors):
    """Una sola definición de 'tarjeta' para todo el Panel. Los QLabel se
    dejan transparentes: la hoja de estilos global de la app les pone el
    color de fondo de la ventana y, sin esto, se ven como cajas dentro de la
    tarjeta."""
    return (
        f"QFrame#panelTarjeta {{ background-color: {colors['card']};"
        f" border: 1px solid {colors['border']}; border-radius: 12px; }}"
        "QFrame#panelTarjeta QLabel { background: transparent; border: none; }"
    )


class MiniTendencia(QWidget):
    """Línea chica con un suave relleno debajo y un punto al final. Sin ejes
    ni números: solo la forma de la tendencia."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self._valores = []
        self._color = QColor(COLOR_ACCENT)

    def poner(self, valores, color=None):
        self._valores = [v for v in valores if np.isfinite(v)]
        if color:
            self._color = QColor(color)
        self.update()

    def paintEvent(self, _event):
        v = self._valores
        if len(v) < 2:
            return
        lo, hi = min(v), max(v)
        rango = (hi - lo) or 1.0
        w, h, mx, my = self.width(), self.height(), 4, 5
        puntos = [
            QPointF(mx + i * (w - 2 * mx) / (len(v) - 1), h - my - (y - lo) / rango * (h - 2 * my))
            for i, y in enumerate(v)
        ]
        linea = QPainterPath(puntos[0])
        for p in puntos[1:]:
            linea.lineTo(p)
        relleno = QPainterPath(linea)
        relleno.lineTo(puntos[-1].x(), h)
        relleno.lineTo(puntos[0].x(), h)
        relleno.closeSubpath()
        suave = QColor(self._color)
        suave.setAlpha(38)
        pintor = QPainter(self)
        pintor.setRenderHint(QPainter.Antialiasing)
        pintor.fillPath(relleno, suave)
        pintor.setPen(QPen(self._color, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        pintor.drawPath(linea)
        pintor.setPen(Qt.NoPen)
        pintor.setBrush(self._color)
        pintor.drawEllipse(puntos[-1], 3, 3)
        pintor.end()


class TarjetaIndicador(QFrame):
    def __init__(self, ind, valor, variacion, tendencia, colors, parent=None):
        super().__init__(parent)
        self.setObjectName("panelTarjeta")
        self.setStyleSheet(estilo_tarjeta(colors))
        est = colores_estado(colors)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(4)

        titulo = QLabel(ind.nombre)
        titulo.setStyleSheet(f"color: {colors['muted']}; font-size: 12px;")
        titulo.setWordWrap(True)
        lay.addWidget(titulo)

        numero = QLabel(ind.formatear(valor))
        numero.setStyleSheet(f"color: {colors['text']}; font-size: 30px; font-weight: 600;")
        lay.addWidget(numero)

        if variacion is None:
            texto, color = " ", est["igual"]
        elif abs(variacion) < 0.5:
            texto, color = "■ estable vs 1ª mitad", est["igual"]
        elif variacion > 0:
            texto, color = f"▲ {variacion:.1f} % vs 1ª mitad", est["sube"]
        else:
            texto, color = f"▼ {abs(variacion):.1f} % vs 1ª mitad", est["baja"]
        cambio = QLabel(texto)
        cambio.setStyleSheet(f"color: {color}; font-size: 12px;")
        lay.addWidget(cambio)

        lay.addSpacing(4)
        self.tendencia = MiniTendencia()
        self.tendencia.poner(tendencia, color)
        lay.addWidget(self.tendencia)


def trazar_area(plot, x, y, colors):
    """Línea con relleno suave y base en cero (con área, la base tiene que
    ser cero para no exagerar las diferencias). Si un punto se despega
    claramente del resto, se marca como 'pico'. Es el estilo de los gráficos
    del Panel: lo usan el gráfico automático y los gráficos propios."""
    y = np.asarray(y, dtype=float)
    acento = QColor(COLOR_ACCENT)
    suave = QColor(acento)
    suave.setAlpha(45)
    plot.plot(x, y, pen=pg.mkPen(acento, width=2.5), fillLevel=0, brush=suave)
    techo = float(np.nanmax(y)) if len(y) else 1.0
    piso = min(0.0, float(np.nanmin(y))) if len(y) else 0.0
    plot.setYRange(piso, techo * 1.15 if techo > 0 else 1.0, padding=0)
    if len(y) >= 5:
        mediana = float(np.nanmedian(y))
        i = int(np.nanargmax(y))
        if mediana > 0 and y[i] > 1.6 * mediana:
            est = colores_estado(colors)
            punto = pg.ScatterPlotItem([x[i]], [y[i]], size=10, brush=pg.mkBrush(est["baja"]), pen=pg.mkPen(None))
            plot.addItem(punto)
            txt = pg.TextItem("pico", color=est["baja"], anchor=(0.5, 1.6))
            txt.setPos(x[i], y[i])
            plot.addItem(txt)


def _rango_barras(plot, valores):
    techo = max(valores) if len(valores) else 1.0
    piso = min(0.0, min(valores)) if len(valores) else 0.0
    plot.setYRange(piso, techo * 1.15 if techo > 0 else 1.0, padding=0)


def _etiquetas_eje(plot, etiquetas, maximo=8):
    """Pocas etiquetas y cortas en el eje X, para que no se pisen."""
    paso = max(1, -(-len(etiquetas) // maximo))
    ticks = [(i, e if len(e) <= 14 else e[:13] + "…")
             for i, e in enumerate(etiquetas) if i % paso == 0]
    plot.getAxis("bottom").setTicks([ticks])


class TarjetaGrafico(QFrame):
    """Un gráfico que el usuario creó en la pestaña Gráficos, dibujado con la
    misma tarjeta del Panel (colores, borde, tipografía y estilo de línea)."""

    def __init__(self, cfg, datos, colors, parent=None):
        super().__init__(parent)
        self.setObjectName("panelTarjeta")
        self.setStyleSheet(estilo_tarjeta(colors))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(6)

        texto = clave_grafico(cfg)
        if cfg.get("nombre") and cfg.get("filtro"):
            texto += f" · solo donde {describir_filtro(cfg['filtro'])}"
        if datos.get("detalle"):
            texto += f" · {datos['detalle']}"
        titulo = QLabel(texto)
        titulo.setWordWrap(True)
        titulo.setStyleSheet(f"color: {colors['text']}; font-size: 13px;")
        lay.addWidget(titulo)

        if "error" in datos:
            aviso = QLabel(datos["error"])
            aviso.setWordWrap(True)
            aviso.setStyleSheet(f"color: {colors['muted']}; font-size: 12px;")
            lay.addWidget(aviso)
            return

        if datos["modo"] == "dona":
            dona = DonutChartWidget()
            dona.setMinimumHeight(190)
            dona.set_data(datos["etiquetas"], datos["valores"], colors["text"])
            lay.addWidget(dona)
            return

        con_fechas = datos["modo"] == "fecha"
        eje = pg.DateAxisItem(orientation="bottom") if con_fechas else pg.AxisItem(orientation="bottom")
        pw = pg.PlotWidget(axisItems={"bottom": eje})
        pw.setMinimumHeight(190)
        pw.setFrameShape(QFrame.NoFrame)
        pw.setMenuEnabled(False)
        pw.showGrid(x=False, y=True, alpha=0.12)
        pw.setBackground(colors["card"])
        for nombre_eje in ("left", "bottom"):
            pw.getAxis(nombre_eje).setPen(pg.mkPen(colors["border"]))
            pw.getAxis(nombre_eje).setTextPen(pg.mkPen(colors["muted"]))
        lay.addWidget(pw)

        if datos["modo"] == "histograma":
            pw.addItem(pg.BarGraphItem(x=datos["x"], height=datos["y"], width=datos["ancho"], brush=COLOR_ACCENT))
            _rango_barras(pw, datos["y"])
        elif con_fechas and datos["tipo"] == "Líneas":
            trazar_area(pw, datos["x"], datos["y"], colors)
        elif con_fechas:
            xs = datos["x"]
            ancho = (float(np.min(np.diff(xs))) if len(xs) > 1 else 86400.0) * 0.8
            pw.addItem(pg.BarGraphItem(x=xs, height=datos["y"], width=ancho, brush=COLOR_ACCENT))
            _rango_barras(pw, datos["y"])
        else:
            n = len(datos["valores"])
            xs = list(range(n))
            if datos["tipo"] == "Barras":
                pw.addItem(pg.BarGraphItem(x=xs, height=datos["valores"], width=0.7, brush=COLOR_ACCENT))
                _rango_barras(pw, datos["valores"])
            else:
                trazar_area(pw, xs, datos["valores"], colors)
            _etiquetas_eje(pw, datos["etiquetas"])


def _lista_con_casillas(items, marcados, alto=120):
    """items: [(texto, clave)]; marcados: set de claves."""
    lista = QListWidget()
    lista.setMaximumHeight(alto)
    for texto, clave in items:
        it = QListWidgetItem(texto)
        it.setData(Qt.UserRole, clave)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Checked if clave in marcados else Qt.Unchecked)
        lista.addItem(it)
    return lista


def _claves_marcadas(lista, marcado=True):
    est = Qt.Checked if marcado else Qt.Unchecked
    return [lista.item(i).data(Qt.UserRole) for i in range(lista.count())
            if lista.item(i).checkState() == est]


class DialogoPersonalizarPanel(QDialog):
    """Personalizar el Panel, en pestañas: qué se ve, qué hallazgos entran y
    con qué tablas se combina."""

    def __init__(self, candidatos, seleccionados, columnas_valor, col_valor,
                 mostrar_hallazgos, mostrar_tiempo, cols_hallazgos, ocultas,
                 combinables, combinar_sel, parent=None, mostrar_alarmas=True,
                 nombres_propios=None, graficos_opciones=None, graficos_marcados=None,
                 auto_marcado=True):
        super().__init__(parent)
        self.setWindowTitle("Personalizar panel")
        self.setMinimumWidth(420)
        raiz = QVBoxLayout(self)
        pestanas = QTabWidget()
        raiz.addWidget(pestanas)

        # -- 1. Qué se ve
        p1 = QWidget()
        l1 = QVBoxLayout(p1)
        l1.addWidget(QLabel(f"Indicadores a mostrar (hasta {MAX_INDICADORES}):"))
        propios = set(nombres_propios or [])
        self.lista = _lista_con_casillas(
            [(i.nombre + ("  · tuyo" if i.nombre in propios else "  · automático"), i.nombre)
             for i in candidatos], set(seleccionados), 130)
        l1.addWidget(self.lista)
        l1.addWidget(QLabel("Columna del gráfico principal:"))
        self.combo = QComboBox()
        self.combo.addItems([str(c) for c in columnas_valor])
        if col_valor is not None and str(col_valor) in [str(c) for c in columnas_valor]:
            self.combo.setCurrentText(str(col_valor))
        l1.addWidget(self.combo)
        l1.addWidget(QLabel("Secciones:"))
        self.chk_alarmas = QCheckBox("Alarmas")
        self.chk_alarmas.setChecked(mostrar_alarmas)
        l1.addWidget(self.chk_alarmas)
        self.chk_hallazgos = QCheckBox("Lo que Hadar notó (hallazgos)")
        self.chk_hallazgos.setChecked(mostrar_hallazgos)
        l1.addWidget(self.chk_hallazgos)
        self.chk_tiempo = QCheckBox("Línea de tiempo")
        self.chk_tiempo.setChecked(mostrar_tiempo)
        l1.addWidget(self.chk_tiempo)
        l1.addStretch(1)
        pestanas.addTab(p1, "Qué se ve")

        # -- 1b. Gráficos (el automático de Hadar + los que creaste en Gráficos)
        p1b = QWidget()
        l1b = QVBoxLayout(p1b)
        l1b.addWidget(QLabel("Gráficos a mostrar en el panel:"))
        items_g = [("Gráfico automático de Hadar", CLAVE_AUTO)]
        items_g += [(k + "  · tuyo", k) for k in (graficos_opciones or [])]
        marcados_g = set(graficos_marcados or []) | ({CLAVE_AUTO} if auto_marcado else set())
        self.lista_graficos = _lista_con_casillas(items_g, marcados_g, 170)
        l1b.addWidget(self.lista_graficos)
        l1b.addWidget(QLabel(
            f"Los «tuyos» son los de la pestaña Gráficos (hasta {MAX_GRAFICOS_PROPIOS}).\n"
            "Se pueden mostrar: Barras, Líneas, Histograma y Dona / Torta."))
        l1b.addStretch(1)
        pestanas.addTab(p1b, "Gráficos")

        # -- 2. Hallazgos
        p2 = QWidget()
        l2 = QVBoxLayout(p2)
        self.lista_cols = None
        if cols_hallazgos:
            l2.addWidget(QLabel("Columnas cuyos hallazgos aparecen en el panel\n"
                                "(desmarca las que no te sirven, por ejemplo los ID):"))
            items = [(f"{c}  ({n})", c) for c, n in sorted(cols_hallazgos.items())]
            self.lista_cols = _lista_con_casillas(items, set(cols_hallazgos) - set(ocultas), 220)
            l2.addWidget(self.lista_cols)
        else:
            l2.addWidget(QLabel("Todavía no hay hallazgos. Genera la Narrativa y vuelve aquí\n"
                                "para elegir qué columnas se muestran."))
        l2.addStretch(1)
        pestanas.addTab(p2, "Hallazgos")

        # -- 3. Datos combinados (solo con varias tablas conectadas)
        self.lista_tablas = None
        if combinables:
            p3 = QWidget()
            l3 = QVBoxLayout(p3)
            l3.addWidget(QLabel("Traer columnas de estas tablas conectadas\n"
                                "(sirven para tener fechas, nombres o categorías):"))
            marcadas = set(combinables) if combinar_sel is None else set(combinar_sel)
            self.lista_tablas = _lista_con_casillas([(t, t) for t in combinables], marcadas, 160)
            l3.addWidget(self.lista_tablas)
            l3.addWidget(QLabel("Cada fila sigue siendo una fila: los totales no se inflan."))
            l3.addStretch(1)
            pestanas.addTab(p3, "Datos combinados")

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self._aceptar)
        botones.rejected.connect(self.reject)
        raiz.addWidget(botones)

    def _aceptar(self):
        if len(self.nombres_elegidos()) > MAX_INDICADORES:
            QMessageBox.information(
                self, "Demasiados indicadores",
                f"Elige {MAX_INDICADORES} o menos para que el panel se lea de un vistazo.",
            )
            return
        if len(self.graficos_elegidos()) > MAX_GRAFICOS_PROPIOS:
            QMessageBox.information(
                self, "Demasiados gráficos",
                f"Elige {MAX_GRAFICOS_PROPIOS} gráficos tuyos o menos para que el panel se lea de un vistazo.",
            )
            return
        self.accept()

    def nombres_elegidos(self):
        return _claves_marcadas(self.lista)

    def graficos_elegidos(self):
        return [k for k in _claves_marcadas(self.lista_graficos) if k != CLAVE_AUTO]

    def auto_elegido(self):
        return CLAVE_AUTO in _claves_marcadas(self.lista_graficos)

    def columna_elegida(self):
        return self.combo.currentText() or None

    def columnas_ocultas(self):
        """None si no hubo hallazgos que elegir (se mantiene la regla por defecto)."""
        return None if self.lista_cols is None else _claves_marcadas(self.lista_cols, False)

    def tablas_elegidas(self):
        return None if self.lista_tablas is None else _claves_marcadas(self.lista_tablas)


def _con_alfa(color, alfa):
    """El mismo color con transparencia, para que la barra se note en
    cualquier tema (claro u oscuro) sin inventar un color nuevo."""
    q = QColor(color)
    return f"rgba({q.red()}, {q.green()}, {q.blue()}, {alfa})"


class _RuedaAlScroll(QObject):
    """Los gráficos de pyqtgraph se quedan con la ruedita del ratón (para
    hacer zoom) y el Panel no se desliza cuando el puntero está encima de
    uno. Este filtro le pasa la ruedita al Panel; con Ctrl apretado deja que
    el gráfico haga su zoom de siempre."""

    def __init__(self, area, parent=None):
        super().__init__(parent)
        self._area = area

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and not (event.modifiers() & Qt.ControlModifier):
            QApplication.sendEvent(self._area.viewport(), event)
            return True
        return False


class PanelControl(QWidget):
    """La pestaña Panel. `host` es la ventana principal: se leen de ella
    df, filtered_df, indicadores, colors y rango_tiempo."""

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.colors = host.colors
        self._sucio = True
        if not hasattr(host, "panel_config"):
            host.panel_config = {"indicadores": None, "col_valor": None}
        self._cache_padres = (None, {})
        self._propias = []
        self._usadas = []
        self._aplican_hallazgos = True

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.actualizar)

        # La cabecera queda fija arriba; todo lo demás va dentro de un área
        # con barra para deslizarse hacia abajo cuando no cabe en pantalla.
        externo = QVBoxLayout(self)
        externo.setContentsMargins(18, 16, 18, 0)
        externo.setSpacing(12)

        cab = QHBoxLayout()
        self.lbl_titulo = QLabel("Panel de control")
        cab.addWidget(self.lbl_titulo)
        self.lbl_sub = QLabel("")
        cab.addWidget(self.lbl_sub)
        cab.addStretch()
        self.lbl_base = QLabel("Ver desde:")
        cab.addWidget(self.lbl_base)
        self.combo_base = QComboBox()
        self.combo_base.setMinimumWidth(190)
        self.combo_base.activated.connect(self._cambiar_base)
        cab.addWidget(self.combo_base)
        self.lbl_base.setVisible(False)
        self.combo_base.setVisible(False)
        self.btn_personalizar = QPushButton("Personalizar")
        self.btn_personalizar.clicked.connect(self._personalizar)
        self.btn_personalizar.setCursor(Qt.PointingHandCursor)
        self.btn_personalizar.setStyleSheet(QSS_SUBPESTANA)
        cab.addWidget(self.btn_personalizar)
        externo.addLayout(cab)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("panelScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.verticalScrollBar().setSingleStep(32)
        self.scroll.viewport().setObjectName("panelViewport")
        self._contenido = QWidget()
        self._contenido.setObjectName("panelContenido")
        self.scroll.setWidget(self._contenido)
        externo.addWidget(self.scroll, 1)
        self._rueda = _RuedaAlScroll(self.scroll, self)

        raiz = QVBoxLayout(self._contenido)
        raiz.setContentsMargins(0, 2, 6, 16)
        raiz.setSpacing(14)

        self.lbl_vacio = QLabel("Carga datos para ver tu panel.")
        self.lbl_vacio.setObjectName("muted")
        self.lbl_vacio.setAlignment(Qt.AlignCenter)
        raiz.addWidget(self.lbl_vacio)

        self.fila_tarjetas = QGridLayout()
        self.fila_tarjetas.setSpacing(12)
        raiz.addLayout(self.fila_tarjetas)
        self._tarjetas = []

        self.card_alarmas = QFrame()
        self.card_alarmas.setObjectName("panelTarjeta")
        la = QVBoxLayout(self.card_alarmas)
        la.setContentsMargins(16, 12, 16, 12)
        la.setSpacing(8)
        cab_al = QHBoxLayout()
        self.lbl_alarmas = QLabel("Alarmas")
        cab_al.addWidget(self.lbl_alarmas)
        self.lbl_alarmas_estado = QLabel("")
        cab_al.addWidget(self.lbl_alarmas_estado)
        cab_al.addStretch()
        self.btn_alarmas = QPushButton("Ver alarmas →")
        self.btn_alarmas.setFlat(True)
        self.btn_alarmas.setCursor(Qt.PointingHandCursor)
        self.btn_alarmas.clicked.connect(self._ir_a_alarmas)
        cab_al.addWidget(self.btn_alarmas)
        la.addLayout(cab_al)
        self.grid_alarmas = QGridLayout()
        self.grid_alarmas.setSpacing(10)
        la.addLayout(self.grid_alarmas)
        self._chips_alarmas = []
        self.card_alarmas.setVisible(False)
        raiz.addWidget(self.card_alarmas)

        # «Lo que se viene»: respuestas de Predicción > Preguntas que el usuario
        # llevó al Panel (ver tablero.py). Solo aparece si hay alguna.
        self.card_pronosticos = QFrame()
        self.card_pronosticos.setObjectName("panelTarjeta")
        lp = QVBoxLayout(self.card_pronosticos)
        lp.setContentsMargins(16, 12, 16, 12)
        lp.setSpacing(8)
        cab_pr = QHBoxLayout()
        self.lbl_pronosticos = QLabel("Lo que se viene")
        cab_pr.addWidget(self.lbl_pronosticos)
        self.lbl_pronosticos_n = QLabel("")
        cab_pr.addWidget(self.lbl_pronosticos_n)
        cab_pr.addStretch()
        lp.addLayout(cab_pr)
        self.grid_pronosticos = QGridLayout()
        self.grid_pronosticos.setSpacing(10)
        lp.addLayout(self.grid_pronosticos)
        self._chips_pronosticos = []
        self.card_pronosticos.setVisible(False)
        raiz.addWidget(self.card_pronosticos)

        # «Qué tan bien acertó»: lo predicho contra lo real (circuito de retorno,
        # ver retorno.py). Solo aparece si el usuario ya hizo alguna comparación.
        self.card_retorno = QFrame()
        self.card_retorno.setObjectName("panelTarjeta")
        lr = QVBoxLayout(self.card_retorno)
        lr.setContentsMargins(16, 12, 16, 12)
        lr.setSpacing(8)
        cab_re = QHBoxLayout()
        self.lbl_retorno = QLabel("Qué tan bien acertó")
        cab_re.addWidget(self.lbl_retorno)
        cab_re.addStretch()
        lr.addLayout(cab_re)
        self.grid_retorno = QGridLayout()
        self.grid_retorno.setSpacing(10)
        lr.addLayout(self.grid_retorno)
        self._chips_retorno = []
        self.card_retorno.setVisible(False)
        raiz.addWidget(self.card_retorno)

        self.card_grafico = QFrame()
        self.card_grafico.setObjectName("panelTarjeta")
        lg = QVBoxLayout(self.card_grafico)
        lg.setContentsMargins(16, 14, 16, 12)
        self.lbl_grafico = QLabel("")
        lg.addWidget(self.lbl_grafico)
        self.plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem(orientation="bottom")})
        self.plot.setMinimumHeight(260)
        self.plot.setFrameShape(QFrame.NoFrame)
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=False, y=True, alpha=0.12)
        lg.addWidget(self.plot)

        fila = QHBoxLayout()
        fila.setSpacing(12)
        fila.addWidget(self.card_grafico, 2)
        fila.addWidget(self._crear_card_hallazgos(), 1)
        raiz.addLayout(fila, 1)
        self.grid_propios = QGridLayout()
        self.grid_propios.setSpacing(12)
        raiz.addLayout(self.grid_propios)
        self._tarjetas_graficos = []
        raiz.addWidget(self._crear_card_tiempo())

        for pw in (self.plot, self.plot_t):
            pw.viewport().installEventFilter(self._rueda)

        self._estilo_plot()
        self.card_grafico.setVisible(False)

    # -- ciclo de vida -----------------------------------------------------
    def notificar_cambio(self):
        """La ventana principal avisa que cambiaron datos, filtros o
        indicadores. Solo se redibuja si el Panel está a la vista; si no,
        queda pendiente hasta que el usuario entre a la pestaña."""
        self._sucio = True
        if self.isVisible():
            self._timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if self._sucio:
            self.actualizar()

    def aplicar_tema(self, colors):
        self.colors = colors
        self._estilo_plot()
        if self.isVisible():
            self.actualizar()
        else:
            self._sucio = True

    def _crear_card_hallazgos(self):
        self.card_hallazgos = QFrame()
        self.card_hallazgos.setObjectName("panelTarjeta")
        lay = QVBoxLayout(self.card_hallazgos)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(8)
        self.lbl_hallazgos = QLabel("Lo que Hadar notó")
        lay.addWidget(self.lbl_hallazgos)
        self.caja_hallazgos = QVBoxLayout()
        self.caja_hallazgos.setSpacing(8)
        lay.addLayout(self.caja_hallazgos)
        lay.addStretch(1)
        self.btn_informe = QPushButton("")
        self.btn_informe.setFlat(True)
        self.btn_informe.setCursor(Qt.PointingHandCursor)
        self.btn_informe.clicked.connect(self._ir_a_narrativa)
        lay.addWidget(self.btn_informe, alignment=Qt.AlignLeft)
        return self.card_hallazgos

    def _crear_card_tiempo(self):
        self.card_tiempo = QFrame()
        self.card_tiempo.setObjectName("panelTarjeta")
        lay = QVBoxLayout(self.card_tiempo)
        lay.setContentsMargins(16, 12, 16, 8)
        lay.setSpacing(2)
        self.lbl_tiempo = QLabel("")
        lay.addWidget(self.lbl_tiempo)
        self.plot_t = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem(orientation="bottom")})
        self.plot_t.setFixedHeight(96)
        self.plot_t.setFrameShape(QFrame.NoFrame)
        self.plot_t.setMenuEnabled(False)
        self.plot_t.hideAxis("left")
        self.plot_t.setMouseEnabled(x=False, y=False)
        lay.addWidget(self.plot_t)
        self.card_tiempo.setVisible(False)
        return self.card_tiempo

    def _ir_a_narrativa(self):
        tv = getattr(self.host, "tabview", None)
        if tv is None:
            return
        for i in range(tv.count()):
            if tv.tabText(i).strip().lower().startswith("narrativa"):
                tv.setCurrentIndex(i)
                return

    def _ir_a_alarmas(self):
        tv = getattr(self.host, "tabview", None)
        if tv is None:
            return
        for i in range(tv.count()):
            if tv.tabText(i).strip().lower().startswith("alarma"):
                tv.setCurrentIndex(i)
                return

    def _estilo_plot(self):
        c = self.colors
        self.plot.setBackground(c["card"])
        for eje in ("left", "bottom"):
            self.plot.getAxis(eje).setPen(pg.mkPen(c["border"]))
            self.plot.getAxis(eje).setTextPen(pg.mkPen(c["muted"]))
        self.plot_t.setBackground(c["card"])
        self.plot_t.getAxis("bottom").setPen(pg.mkPen(c["border"]))
        self.plot_t.getAxis("bottom").setTextPen(pg.mkPen(c["muted"]))
        self._estilo_panel()

    def _estilo_panel(self):
        c = self.colors
        self.lbl_titulo.setStyleSheet(f"color: {c['text']}; font-size: 20px; font-weight: 600;")
        self.lbl_sub.setStyleSheet(f"color: {c['muted']}; font-size: 13px;")
        self.lbl_base.setStyleSheet(f"color: {c['muted']}; font-size: 13px;")
        self.lbl_vacio.setStyleSheet(f"color: {c['muted']}; font-size: 14px;")
        self.lbl_grafico.setStyleSheet(f"color: {c['text']}; font-size: 13px;")
        self.card_grafico.setStyleSheet(estilo_tarjeta(c))
        self.card_hallazgos.setStyleSheet(estilo_tarjeta(c))
        self.card_tiempo.setStyleSheet(estilo_tarjeta(c))
        self.card_alarmas.setStyleSheet(estilo_tarjeta(c))
        self.card_pronosticos.setStyleSheet(estilo_tarjeta(c))
        self.card_retorno.setStyleSheet(estilo_tarjeta(c))
        self.lbl_retorno.setStyleSheet(f"color: {c['text']}; font-size: 13px;")
        self.lbl_pronosticos.setStyleSheet(f"color: {c['text']}; font-size: 13px;")
        self.lbl_alarmas.setStyleSheet(f"color: {c['text']}; font-size: 13px;")
        self.btn_alarmas.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {COLOR_ACCENT};"
            " padding: 2px 0; }"
            f"QPushButton:hover {{ color: {c['text']}; }}"
        )
        self.lbl_hallazgos.setStyleSheet(f"color: {c['text']}; font-size: 13px;")
        self.lbl_tiempo.setStyleSheet(f"color: {c['text']}; font-size: 13px;")
        self.btn_informe.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {COLOR_ACCENT};"
            " padding: 2px 0; text-align: left; }"
            f"QPushButton:hover {{ color: {c['text']}; }}"
        )
        self.scroll.setStyleSheet(
            "QScrollArea#panelScroll { background: transparent; border: none; }"
            "QWidget#panelViewport, QWidget#panelContenido { background: transparent; }"
            "QScrollBar:vertical { background: transparent; width: 10px; margin: 2px 0; }"
            f"QScrollBar::handle:vertical {{ background: {_con_alfa(c['muted'], 110)};"
            " border-radius: 4px; min-height: 36px; }"
            f"QScrollBar::handle:vertical:hover {{ background: {_con_alfa(c['muted'], 200)}; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
        )
        self.btn_personalizar.setCursor(Qt.PointingHandCursor)
        self.btn_personalizar.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {c['text']};"
            f" border: 1px solid {c['border']}; border-radius: 8px; padding: 6px 14px; }}"
            f"QPushButton:hover {{ border-color: {COLOR_ACCENT}; color: {COLOR_ACCENT}; }}"
        )

    # -- datos ---------------------------------------------------------------
    def _padres(self):
        tablas = getattr(self.host, "tablas", None) or {}
        firma = tuple((n, len(d), tuple(map(str, d.columns))) for n, d in tablas.items())
        if self._cache_padres[0] != firma:
            self._cache_padres = (firma, inferir_padres(tablas) if len(tablas) > 1 else {})
        return self._cache_padres[1]

    def _base_y_df(self):
        """(nombre de la tabla base, su df). Por defecto, la tabla activa con
        sus filtros; si el usuario eligió otra en 'Ver desde', esa completa."""
        h = self.host
        tablas = getattr(h, "tablas", None) or {}
        activa = getattr(h, "nombre_tabla_activa", None)
        base = h.panel_config.get("tabla_base")
        if base is not None and (base not in tablas or base == activa):
            base = None
        if base is None:
            df = h.filtered_df if getattr(h, "filtered_df", None) is not None else getattr(h, "df", None)
            return activa, df
        return base, tablas[base]

    def _df(self):
        """El df que se dibuja: la tabla base + columnas 'Tabla.columna' de
        las tablas conectadas que el usuario permitió."""
        h = self.host
        tablas = getattr(h, "tablas", None) or {}
        nombre, df = self._base_y_df()
        self._base = nombre
        self._aplican_hallazgos = nombre == getattr(h, "nombre_tabla_activa", None)
        self._propias = list(df.columns) if df is not None else []
        self._usadas = []
        if df is None or df.empty or len(tablas) < 2 or nombre not in tablas:
            return df
        combinado, usadas = combinar_con_padres(
            nombre, df, tablas, self._padres(), h.panel_config.get("tablas_combinar"))
        self._usadas = usadas
        return combinado

    def _actualizar_selector_base(self):
        tablas = list(getattr(self.host, "tablas", None) or {})
        varias = len(tablas) > 1
        self.lbl_base.setVisible(varias)
        self.combo_base.setVisible(varias)
        if not varias:
            return
        activa = getattr(self.host, "nombre_tabla_activa", None)
        base = self.host.panel_config.get("tabla_base")
        self.combo_base.blockSignals(True)
        self.combo_base.clear()
        self.combo_base.addItem(f"{activa} (activa)", None)
        for t in tablas:
            self.combo_base.addItem(t, t)
        idx = self.combo_base.findData(base) if base in tablas else 0
        self.combo_base.setCurrentIndex(max(idx, 0))
        self.combo_base.blockSignals(False)

    def _cambiar_base(self, indice):
        cfg = self.host.panel_config
        cfg["tabla_base"] = self.combo_base.itemData(indice)
        cfg["col_valor"] = None
        cfg["indicadores"] = None
        cfg["tablas_combinar"] = None
        self.actualizar()

    def _propios_validos(self, df):
        """Indicadores de la pestaña Indicadores que se pueden calcular con
        las columnas de lo que se está viendo."""
        validos = []
        for ind in list(getattr(self.host, "indicadores", []) or []):
            try:
                deps = set(ind.columnas_necesarias())
            except Exception:
                deps = set()
            if deps <= set(df.columns):
                validos.append(ind)
        return validos

    def _candidatos(self, df):
        propios = self._propios_validos(df)
        automaticos = [i for i in indicadores_por_defecto(df, self._propias)
                       if i.nombre not in {p.nombre for p in propios}]
        return propios + automaticos

    def _indicadores_visibles(self, df):
        cfg = self.host.panel_config
        candidatos = self._candidatos(df)
        if cfg.get("indicadores"):
            por_nombre = {i.nombre: i for i in candidatos}
            elegidos = [por_nombre[n] for n in cfg["indicadores"] if n in por_nombre]
            if elegidos:
                return elegidos[:MAX_INDICADORES]
        propios = self._propios_validos(df)
        return (propios or indicadores_por_defecto(df, self._propias))[:MAX_INDICADORES]

    def _anomalias_visibles(self):
        """Anomalías detectadas por Narrativa, sin las columnas que el
        usuario decidió ocultar (por defecto, los ID). Solo aplican cuando
        se mira la tabla que estaba activa al generarlas."""
        if not self._aplican_hallazgos:
            return []
        return filtrar_anomalias(getattr(self.host, "_ultimas_anomalias", []),
                                 self.host.panel_config.get("columnas_ocultas"))

    def _dayfirst(self):
        return getattr(getattr(self.host, "panel_linea_tiempo", None), "_dayfirst", True)

    def _columna_fecha(self, df):
        rango = getattr(self.host, "rango_tiempo", None)
        preferida = rango[2] if rango else None
        dayfirst = self._dayfirst()
        fechas_de = None
        if hasattr(self.host, "_fechas_de_columna"):
            fechas_de = lambda d, c: self.host._fechas_de_columna(d, c, dayfirst)
        return elegir_columna_fecha(df, preferida, fechas_de)

    # -- dibujo --------------------------------------------------------------
    def actualizar(self):
        self._sucio = False
        df = self._df()
        self._actualizar_selector_base()
        for t in self._tarjetas:
            t.setParent(None)
            t.deleteLater()
        self._tarjetas.clear()
        self._limpiar_graficos_propios()

        if df is None or df.empty:
            self.lbl_vacio.setVisible(True)
            self.card_grafico.setVisible(False)
            self.card_hallazgos.setVisible(False)
            self.card_tiempo.setVisible(False)
            self.card_alarmas.setVisible(False)
            self.card_pronosticos.setVisible(False)
            self.card_retorno.setVisible(False)
            self.lbl_sub.setText("")
            return
        self.lbl_vacio.setVisible(False)
        cfg = self.host.panel_config
        tabla = self._base
        u = self._usadas
        extra = ""
        if u:
            extra = f" · con datos de {', '.join(u)}" if len(u) <= 2 else f" · con datos de {len(u)} tablas"
        self.lbl_sub.setText(("  " + tabla + " · " if tabla else "  ") + f"{len(df):,} filas{extra}")
        self.lbl_sub.setToolTip("Tablas combinadas: " + ", ".join(u) if u else "")

        col_fecha, fechas = self._columna_fecha(df)
        for i, ind in enumerate(self._indicadores_visibles(df)):
            tarjeta = TarjetaIndicador(
                ind, ind.calcular(df),
                variacion_indicador(ind, df, fechas),
                tendencia_indicador(ind, df, fechas),
                self.colors,
            )
            self.fila_tarjetas.addWidget(tarjeta, 0, i)
            self.fila_tarjetas.setColumnStretch(i, 1)
            self._tarjetas.append(tarjeta)

        self._dibujar_alarmas(cfg.get("mostrar_alarmas", True))
        self._dibujar_pronosticos()
        self._dibujar_retorno()
        self._dibujar_grafico(df, col_fecha, fechas)
        self._dibujar_graficos_propios(df)
        self._dibujar_hallazgos(cfg.get("mostrar_hallazgos", True))
        self._dibujar_tiempo(df, col_fecha, fechas, cfg.get("mostrar_tiempo", True))

    def _dibujar_alarmas(self, mostrar, maximo=6, columnas=3):
        """Recuadro con el estado de las alarmas configuradas (las que
        suenan primero). Se evalúan igual que en la pestaña Alarma: sobre
        los datos de la tabla activa con los filtros puestos, sin el cruce
        con otras tablas que usa el resto del Panel."""
        for chip in self._chips_alarmas:
            chip.setParent(None)
            chip.deleteLater()
        self._chips_alarmas.clear()
        reglas = list(getattr(self.host, "reglas_alarma", []) or [])
        if not mostrar or not reglas:
            self.card_alarmas.setVisible(False)
            return
        self.card_alarmas.setVisible(True)

        h = self.host
        df_alarmas = h.filtered_df if getattr(h, "filtered_df", None) is not None else h.df
        estados = []
        for regla in reglas:
            valor = calcular_valor_actual(regla, df_alarmas, h.indicadores)
            resumen = evaluar_columna(regla, df_alarmas) if regla["tipo"] == "Columna" else None
            estados.append((regla, evaluar_regla(regla, valor), estado_en_palabras(regla, valor, resumen),
                            valor is None, resumen))
        activas = sum(1 for _, en, _, _, _ in estados if en)
        c = self.colors
        est = colores_estado(c)
        if activas:
            self.lbl_alarmas_estado.setText(f"  {activas} activa" if activas == 1 else f"  {activas} activas")
            self.lbl_alarmas_estado.setStyleSheet(f"color: {est['baja']}; font-size: 12px; font-weight: 600;")
        else:
            self.lbl_alarmas_estado.setText("  todo en orden")
            self.lbl_alarmas_estado.setStyleSheet(f"color: {est['sube']}; font-size: 12px;")

        # Las que suenan primero; el resto en el orden en que se crearon.
        estados.sort(key=lambda e: 0 if e[1] else 1)
        visibles = estados[:maximo]
        for i, (regla, en_alarma, frase, sin_datos, resumen) in enumerate(visibles):
            color = est["baja"] if en_alarma else (est["igual"] if sin_datos else est["sube"])
            borde = est["baja"] if en_alarma else c["border"]
            chip = QFrame()
            chip.setObjectName("panelChip")
            chip.setStyleSheet(
                f"QFrame#panelChip {{ background: transparent; border: 1px solid {borde}; border-radius: 8px; }}"
                "QFrame#panelChip QLabel { background: transparent; border: none; }"
            )
            lay = QVBoxLayout(chip)
            lay.setContentsMargins(12, 8, 12, 8)
            lay.setSpacing(2)
            titulo = QLabel(f"<span style='color:{color}'>●</span>&nbsp; " + html.escape(regla["nombre"]))
            titulo.setTextFormat(Qt.RichText)
            titulo.setWordWrap(True)
            titulo.setStyleSheet(f"color: {c['text']}; font-size: 12px; font-weight: 600;")
            lay.addWidget(titulo)
            detalle = QLabel(frase)
            detalle.setWordWrap(True)
            detalle.setStyleSheet(f"color: {c['muted']}; font-size: 12px;")
            lay.addWidget(detalle)
            if en_alarma and regla.get("mensaje"):
                aviso = QLabel(regla["mensaje"])
                aviso.setWordWrap(True)
                aviso.setStyleSheet(f"color: {est['baja']}; font-size: 12px;")
                lay.addWidget(aviso)
            filas = lineas_filas(regla, resumen) if en_alarma else []
            if filas:
                chip.setToolTip("Filas que la activaron:\n" + "\n".join(filas))
            self.grid_alarmas.addWidget(chip, i // columnas, i % columnas)
            self._chips_alarmas.append(chip)
        for col in range(columnas):
            self.grid_alarmas.setColumnStretch(col, 1)
        resto = len(estados) - len(visibles)
        self.btn_alarmas.setText(f"Ver alarmas (+{resto} más) →" if resto > 0 else "Ver alarmas →")

    def _dibujar_pronosticos(self, maximo=6, columnas=2):
        """«Lo que se viene»: las respuestas de Preguntas que el usuario llevó al
        Panel, las más urgentes primero. Es el texto tal como se calculó (con su
        rango y su confianza): el Panel no vuelve a calcular nada."""
        for chip in self._chips_pronosticos:
            chip.setParent(None)
            chip.deleteLater()
        self._chips_pronosticos.clear()
        tablero = getattr(self.host, "tablero", None)
        lista = tablero.pronosticos() if tablero is not None else []
        if not lista:
            self.card_pronosticos.setVisible(False)
            return
        self.card_pronosticos.setVisible(True)
        c = self.colors
        est = colores_estado(c)
        color_nivel = {"alerta": est["baja"], "atencion": "#D97706" if _es_tema_claro(c) else "#FBBF24"}
        varias_tablas = len({p.tabla for p in lista}) > 1
        visibles = lista[:maximo]
        resto = len(lista) - len(visibles)
        self.lbl_pronosticos_n.setText(f"  {len(lista)}" + (f" · se ven las {maximo} más urgentes" if resto > 0 else ""))
        self.lbl_pronosticos_n.setStyleSheet(f"color: {c['muted']}; font-size: 12px;")
        for i, p in enumerate(visibles):
            color = color_nivel.get(p.nivel, COLOR_ACCENT)
            borde = color if p.nivel == "alerta" else c["border"]
            chip = QFrame()
            chip.setObjectName("panelChip")
            chip.setStyleSheet(
                f"QFrame#panelChip {{ background: transparent; border: 1px solid {borde}; border-radius: 8px; }}"
                "QFrame#panelChip QLabel { background: transparent; border: none; }"
            )
            lay = QVBoxLayout(chip)
            lay.setContentsMargins(12, 8, 12, 8)
            lay.setSpacing(3)
            fila = QHBoxLayout()
            nombre = html.escape(p.titulo or p.palabra)
            if varias_tablas and p.tabla:
                nombre += f" <span style='color:{c['muted']}; font-weight:400'>· {html.escape(p.tabla)}</span>"
            titulo = QLabel(f"<span style='color:{color}'>●</span>&nbsp; " + nombre)
            titulo.setTextFormat(Qt.RichText)
            titulo.setWordWrap(True)
            titulo.setStyleSheet(f"color: {c['text']}; font-size: 12px; font-weight: 600;")
            fila.addWidget(titulo, 1)
            quitar = QPushButton("Quitar")
            quitar.setFlat(True)
            quitar.setCursor(Qt.PointingHandCursor)
            quitar.setToolTip("Quita esta respuesta del Panel (no borra nada más).")
            quitar.setStyleSheet(f"QPushButton {{ color: {c['muted']}; font-size: 11px; border: none; padding: 0 2px; }}"
                                 f"QPushButton:hover {{ color: {c['text']}; }}")
            quitar.clicked.connect(lambda _=False, k=p.clave: QTimer.singleShot(0, lambda: self._quitar_pronostico(k)))
            fila.addWidget(quitar, 0, Qt.AlignTop)
            lay.addLayout(fila)
            titular = QLabel(p.titular)
            titular.setWordWrap(True)
            titular.setStyleSheet(f"color: {c['text']}; font-size: 13px;")
            lay.addWidget(titular)
            if p.aviso_critico:
                aviso = QLabel(p.aviso_critico)
                aviso.setWordWrap(True)
                aviso.setStyleSheet(f"color: {est['baja']}; font-size: 12px;")
                lay.addWidget(aviso)
            pie = QLabel(f"Confianza {p.confianza}" + (f" · calculado el {p.calculado_en}" if p.calculado_en else ""))
            pie.setStyleSheet(f"color: {c['muted']}; font-size: 11px;")
            if p.por_que_confianza:
                pie.setToolTip(p.por_que_confianza)
            lay.addWidget(pie)
            self.grid_pronosticos.addWidget(chip, i // columnas, i % columnas)
            self._chips_pronosticos.append(chip)
        for col in range(columnas):
            self.grid_pronosticos.setColumnStretch(col, 1)

    def _dibujar_retorno(self, maximo=4, columnas=2):
        """«Qué tan bien acertó»: la última comparación de lo predicho con lo real,
        por tabla, las peores primero. Es el texto que dio retorno.comparar: el Panel
        no vuelve a calcular nada."""
        for chip in self._chips_retorno:
            chip.setParent(None)
            chip.deleteLater()
        self._chips_retorno.clear()
        tablero = getattr(self.host, "tablero", None)
        lista = tablero.retornos() if tablero is not None else []
        if not lista:
            self.card_retorno.setVisible(False)
            return
        self.card_retorno.setVisible(True)
        c = self.colors
        est = colores_estado(c)
        ambar = "#D97706" if _es_tema_claro(c) else "#FBBF24"
        color_nivel = {"bien": est["sube"], "atencion": ambar, "alerta": est["baja"], "info": c["muted"]}
        varias_tablas = len({d.get("tabla") for d in lista}) > 1
        for i, d in enumerate(lista[:maximo]):
            nivel = d.get("nivel", "info")
            color = color_nivel.get(nivel, c["muted"])
            borde = color if nivel == "alerta" else c["border"]
            chip = QFrame()
            chip.setObjectName("panelChip")
            chip.setStyleSheet(
                f"QFrame#panelChip {{ background: transparent; border: 1px solid {borde}; border-radius: 8px; }}"
                "QFrame#panelChip QLabel { background: transparent; border: none; }"
            )
            lay = QVBoxLayout(chip)
            lay.setContentsMargins(12, 8, 12, 8)
            lay.setSpacing(3)
            fila = QHBoxLayout()
            nombre = html.escape(str(d.get("objetivo", "")))
            if varias_tablas and d.get("tabla"):
                nombre += f" <span style='color:{c['muted']}; font-weight:400'>· {html.escape(str(d['tabla']))}</span>"
            titulo = QLabel(f"<span style='color:{color}'>●</span>&nbsp; " + nombre)
            titulo.setTextFormat(Qt.RichText)
            titulo.setWordWrap(True)
            titulo.setStyleSheet(f"color: {c['text']}; font-size: 12px; font-weight: 600;")
            fila.addWidget(titulo, 1)
            quitar = QPushButton("Quitar")
            quitar.setFlat(True)
            quitar.setCursor(Qt.PointingHandCursor)
            quitar.setToolTip("Quita esta comparación del Panel (no borra nada más).")
            quitar.setStyleSheet(f"QPushButton {{ color: {c['muted']}; font-size: 11px; border: none; padding: 0 2px; }}"
                                 f"QPushButton:hover {{ color: {c['text']}; }}")
            quitar.clicked.connect(
                lambda _=False, k=d.get("tabla"): QTimer.singleShot(0, lambda: self._quitar_retorno(k)))
            fila.addWidget(quitar, 0, Qt.AlignTop)
            lay.addLayout(fila)
            titular = QLabel(TITULAR_NIVEL.get(nivel, ""))
            titular.setWordWrap(True)
            titular.setStyleSheet(f"color: {c['text']}; font-size: 13px;")
            lay.addWidget(titular)
            lineas = d.get("lineas", [])
            for texto in lineas[1:3]:                  # la cifra principal y cómo se compara con lo esperado
                lbl = QLabel(str(texto))
                lbl.setWordWrap(True)
                lbl.setStyleSheet(f"color: {c['text']}; font-size: 12px;")
                lay.addWidget(lbl)
            pie = QLabel(f"{d.get('n_comparadas', 0)} filas comparadas · el {d.get('comparado_en', '')}"
                         f" · predicho el {d.get('predicho_en', '')}")
            pie.setWordWrap(True)
            pie.setStyleSheet(f"color: {c['muted']}; font-size: 11px;")
            pie.setToolTip("\n".join(str(x) for x in lineas))
            lay.addWidget(pie)
            self.grid_retorno.addWidget(chip, i // columnas, i % columnas)
            self._chips_retorno.append(chip)
        for col in range(columnas):
            self.grid_retorno.setColumnStretch(col, 1)

    def _quitar_retorno(self, tabla):
        tablero = getattr(self.host, "tablero", None)
        if tablero is not None and tabla is not None:
            tablero.quitar_retorno(tabla)
        self._dibujar_retorno()

    def _quitar_pronostico(self, clave):
        tablero = getattr(self.host, "tablero", None)
        if tablero is not None:
            tablero.quitar_pronostico(clave)
        self._dibujar_pronosticos()
        linea = getattr(self.host, "panel_linea_tiempo", None)      # sus franjas de «lo que se viene»
        if linea is not None:
            linea.refrescar_futuro()

    def _limpiar_graficos_propios(self):
        for t in self._tarjetas_graficos:
            t.setParent(None)
            t.deleteLater()
        self._tarjetas_graficos.clear()

    def _dibujar_graficos_propios(self, df):
        """Los gráficos que el usuario eligió desde la pestaña Gráficos, en
        tarjetas del mismo estilo, de a dos por fila bajo el gráfico principal."""
        lista = (self.host.panel_config.get("graficos") or [])[:MAX_GRAFICOS_PROPIOS]
        for i, g in enumerate(lista):
            tarjeta = TarjetaGrafico(g, datos_grafico_propio(g, df), self.colors)
            self.grid_propios.addWidget(tarjeta, i // 2, i % 2)
            self.grid_propios.setColumnStretch(i % 2, 1)
            for pw in tarjeta.findChildren(pg.PlotWidget):
                pw.viewport().installEventFilter(self._rueda)
            self._tarjetas_graficos.append(tarjeta)

    def _opciones_graficos(self):
        """{nombre: config} de los gráficos propios que se pueden elegir: los
        ya enviados al Panel y los que hay ahora en la pestaña Gráficos (si
        hay uno con el mismo nombre, manda el de la pestaña Gráficos)."""
        opciones = {}
        for g in self.host.panel_config.get("graficos") or []:
            opciones[clave_grafico(g)] = g
        for p in getattr(self.host, "chart_panels", None) or []:
            try:
                g = p.config_a_dict()
            except Exception:
                continue
            if g.get("tipo") in GRAFICOS_SOPORTADOS and g.get("x"):
                opciones[clave_grafico(g)] = g
        return opciones

    # -- "Al Panel": lo que llega desde las pestañas Indicadores y Gráficos --
    def nombres_en_panel(self):
        """Nombres de los indicadores que el Panel muestra ahora mismo."""
        try:
            df = self._df()
            if df is None or df.empty:
                return set()
            return {i.nombre for i in self._indicadores_visibles(df)}
        except Exception:
            return set()

    def enviar_indicador(self, nombre):
        """'ok', 'ya' (estaba), 'lleno' (ya hay 4) o 'invalido' (usa columnas
        que esta vista no tiene)."""
        df = self._df()
        if df is None or df.empty or nombre not in {i.nombre for i in self._candidatos(df)}:
            return "invalido"
        actuales = [i.nombre for i in self._indicadores_visibles(df)]
        if nombre in actuales:
            return "ya"
        if len(actuales) >= MAX_INDICADORES:
            return "lleno"
        self.host.panel_config["indicadores"] = actuales + [nombre]
        self.notificar_cambio()
        return "ok"

    def enviar_grafico(self, cfg_grafico):
        """'ok', 'actualizado' (ya estaba: se reemplaza con los cambios),
        'tipo' (no se puede mostrar en el Panel) o 'lleno'."""
        if cfg_grafico.get("tipo") not in GRAFICOS_SOPORTADOS:
            return "tipo"
        cfg = self.host.panel_config
        lista = list(cfg.get("graficos") or [])
        clave = clave_grafico(cfg_grafico)
        for i, g in enumerate(lista):
            if clave_grafico(g) == clave:
                lista[i] = cfg_grafico
                cfg["graficos"] = lista
                self.notificar_cambio()
                return "actualizado"
        if len(lista) >= MAX_GRAFICOS_PROPIOS:
            return "lleno"
        cfg["graficos"] = lista + [cfg_grafico]
        self.notificar_cambio()
        return "ok"

    def _dibujar_grafico(self, df, col_fecha, fechas):
        cfg_g = self.host.panel_config
        if not cfg_g.get("grafico_auto", True) and cfg_g.get("graficos"):
            self.plot.clear()   # el usuario eligió ver solo sus gráficos
            self.card_grafico.setVisible(False)
            return
        col = self.host.panel_config.get("col_valor")
        if col not in df.columns or col not in self._propias:
            col = elegir_columna_valor(df, self._propias)
        self.plot.clear()
        if col is None:
            self.card_grafico.setVisible(False)
            return
        self.card_grafico.setVisible(True)
        if fechas is not None:
            serie, periodo = serie_por_periodo(df[col], fechas)
            if len(serie) >= 2:
                x = serie.index.values.astype("datetime64[s]").astype("int64").astype(float)
                self._usar_eje(True)
                self._trazar(x, serie.values.astype(float))
                self.lbl_grafico.setText(f"Suma de {col} por {periodo}")
                return
        # Sin fechas: tramos iguales en el orden en que vienen las filas.
        partes = _partes_de_tendencia(df, None, 30)
        if not partes:
            self.card_grafico.setVisible(False)
            return
        y = [float(pd.to_numeric(p[col], errors="coerce").sum()) for p in partes]
        self._usar_eje(False)
        self._trazar(list(range(1, len(y) + 1)), y)
        self.lbl_grafico.setText(f"Suma de {col} por tramos de filas (tus datos no traen una columna de fecha)")

    def _dibujar_hallazgos(self, mostrar):
        self.card_hallazgos.setVisible(mostrar)
        if not mostrar:
            return
        while self.caja_hallazgos.count():
            item = self.caja_hallazgos.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        c = self.colors
        est = colores_estado(c)
        color_por_gravedad = {3: est["baja"], 2: COLOR_ACCENT_3, 1: COLOR_ACCENT}
        hallazgos = hallazgos_principales(self._anomalias_visibles())
        if not hallazgos:
            if not self._aplican_hallazgos:
                aviso = "Los hallazgos son de la tabla activa. Cambia a ella para verlos aquí."
            elif getattr(self.host, "_ultimas_anomalias", []):
                aviso = "No quedan hallazgos por mostrar con las columnas elegidas."
            else:
                aviso = "Todavía no hay hallazgos. Aparecen cuando se genera la Narrativa."
            vacio = QLabel(aviso)
            vacio.setWordWrap(True)
            vacio.setStyleSheet(f"color: {c['muted']}; font-size: 12px;")
            self.caja_hallazgos.addWidget(vacio)
            self.btn_informe.setText("Ir a Narrativa →")
            return
        for i, h in enumerate(hallazgos):
            if i:
                linea = QFrame()
                linea.setFixedHeight(1)
                linea.setStyleSheet(f"background-color: {c['border']}; border: none;")
                self.caja_hallazgos.addWidget(linea)
            fila = QLabel(
                f"<span style='color:{color_por_gravedad[h['gravedad']]}'>●</span>&nbsp; "
                + html.escape(h["texto"])
            )
            fila.setTextFormat(Qt.RichText)
            fila.setWordWrap(True)
            fila.setToolTip(h["completo"])
            fila.setStyleSheet(f"color: {c['text']}; font-size: 12px;")
            self.caja_hallazgos.addWidget(fila)
        self.btn_informe.setText("Ver informe completo →")

    def _dibujar_tiempo(self, df, col_fecha, fechas, mostrar):
        self.plot_t.clear()
        if not mostrar or fechas is None:
            self.card_tiempo.setVisible(False)
            return
        validas = fechas.dropna()
        try:
            serie, escala = serie_por_periodo(pd.Series(1.0, index=validas.index), validas, contar=True)
        except Exception:
            serie, escala = pd.Series(dtype=float), "período"
        if len(serie) < 2:
            self.card_tiempo.setVisible(False)
            return
        self.card_tiempo.setVisible(True)
        conteos = serie.values.astype(float)
        seg = serie.index.values.astype("datetime64[s]").astype("int64").astype(float)
        ancho = float(np.median(np.diff(seg)))
        techo = float(np.max(conteos)) or 1.0
        barras = QColor(COLOR_ACCENT)
        barras.setAlpha(150)
        self.plot_t.addItem(pg.BarGraphItem(
            x0=seg, width=ancho * 0.85, height=conteos, brush=barras, pen=pg.mkPen(None)))
        marcas = marcadores_hallazgos(self._anomalias_visibles(), df, col_fecha, self._dayfirst())
        if marcas:
            est = colores_estado(self.colors)
            color_por_gravedad = {3: est["baja"], 2: COLOR_ACCENT_3, 1: COLOR_ACCENT}
            xs = [pd.Timestamp(f).timestamp() for f, _ in marcas]
            self.plot_t.addItem(pg.ScatterPlotItem(
                xs, [techo * 1.28] * len(xs), size=9,
                brush=[pg.mkBrush(color_por_gravedad[g]) for _, g in marcas], pen=pg.mkPen(None)))
        self.plot_t.setYRange(0, techo * 1.45, padding=0)
        self.plot_t.setXRange(seg[0], seg[-1] + ancho, padding=0.01)
        extra = f" · {len(marcas)} hallazgo(s) marcados" if marcas else ""
        self.lbl_tiempo.setText(f"Línea de tiempo · registros por {escala}{extra}")

    def _trazar(self, x, y):
        """Línea con relleno suave y base en cero (con área, la base tiene
        que ser cero para no exagerar las diferencias). Si un punto se
        despega claramente del resto, se marca como 'pico'."""
        trazar_area(self.plot, x, y, self.colors)

    def _usar_eje(self, con_fechas):
        """Cambia el eje X entre fechas y números solo cuando hace falta."""
        if getattr(self, "_eje_con_fechas", True) == con_fechas and hasattr(self, "_eje_listo"):
            return
        self._eje_listo = True
        self._eje_con_fechas = con_fechas
        eje = pg.DateAxisItem(orientation="bottom") if con_fechas else pg.AxisItem(orientation="bottom")
        self.plot.setAxisItems({"bottom": eje})
        self._estilo_plot()

    def _personalizar(self):
        df = self._df()
        if df is None or df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return
        cfg = self.host.panel_config
        visibles = [i.nombre for i in self._indicadores_visibles(df)]
        todas = getattr(self.host, "_ultimas_anomalias", []) if self._aplican_hallazgos else []
        cols_hallazgos = Counter(c for a in todas for c in (a.get("columnas") or []))
        ocultas = cfg.get("columnas_ocultas")
        if ocultas is None:
            ocultas = columnas_ocultas_por_defecto(todas)
        combinables = alcanzables(self._base, self._padres()) if self._base else []
        opciones = self._opciones_graficos()
        dlg = DialogoPersonalizarPanel(
            self._candidatos(df), visibles, columnas_numericas_utiles(df, self._propias),
            cfg.get("col_valor") or elegir_columna_valor(df, self._propias),
            cfg.get("mostrar_hallazgos", True), cfg.get("mostrar_tiempo", True),
            dict(cols_hallazgos), ocultas, combinables, cfg.get("tablas_combinar"), parent=self,
            mostrar_alarmas=cfg.get("mostrar_alarmas", True),
            nombres_propios={i.nombre for i in self._propios_validos(df)},
            graficos_opciones=list(opciones),
            graficos_marcados=[clave_grafico(g) for g in cfg.get("graficos") or []],
            auto_marcado=cfg.get("grafico_auto", True),
        )
        if dlg.exec() == QDialog.Accepted:
            elegidos_g = [opciones[k] for k in dlg.graficos_elegidos() if k in opciones]
            cfg["graficos"] = elegidos_g
            cfg["grafico_auto"] = dlg.auto_elegido() or not elegidos_g   # nunca dejar el panel sin gráfico
            cfg["indicadores"] = dlg.nombres_elegidos()
            cfg["col_valor"] = dlg.columna_elegida()
            cfg["mostrar_alarmas"] = dlg.chk_alarmas.isChecked()
            cfg["mostrar_hallazgos"] = dlg.chk_hallazgos.isChecked()
            cfg["mostrar_tiempo"] = dlg.chk_tiempo.isChecked()
            nuevas = dlg.columnas_ocultas()
            if nuevas is not None:
                cfg["columnas_ocultas"] = nuevas
            elegidas = dlg.tablas_elegidas()
            if elegidas is not None:
                cfg["tablas_combinar"] = None if set(elegidas) == set(combinables) else elegidas
            self.actualizar()