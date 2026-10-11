"""
pronosticos_pdf.py — Exporta a PDF la respuesta de una pregunta de la subpestaña «Preguntas».

Dos modos (salen del resultado ya calculado, no se vuelve a calcular nada):
  - "todo":        la pregunta y la configuración usada + la respuesta completa.
  - "resultados":  solo la respuesta: titular con rango, confianza, tres cifras, explicación,
                   gráfico, tabla, qué asume Hadar y avisos.

Reutiliza las piezas de prediccion_pdf.py (misma tipografía, mismos colores de impresión y la
misma paginación con número de página), así los dos informes se ven como de la misma familia.
Tampoco agrega librerías: los gráficos se dibujan con QPainter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainterPath, QPen, QPolygonF, QTextDocument

try:
    from . import prediccion_pdf as _pp
    from .pronostico_base import fecha_txt, num_txt, pct_txt
    from .pronosticos import PALABRAS_CLAVE
except ImportError:  # pruebas sueltas fuera del paquete
    import prediccion_pdf as _pp
    from pronostico_base import fecha_txt, num_txt, pct_txt
    from pronosticos import PALABRAS_CLAVE

ACENTO, AMBAR, ROJO, GRIS, TINTA, REGLA = _pp.ACENTO, _pp.AMBAR, _pp.ROJO, _pp.GRIS, _pp.TINTA, _pp.REGLA
ANCHO_FIG = _pp.ANCHO_FIG
COLOR_NIVEL = {"info": ACENTO, "atencion": AMBAR, "alerta": ROJO}
ROTULO_NIVEL = {"info": "", "atencion": "Atención · ", "alerta": "Alerta · "}
MAX_FILAS_TABLA = 200
COLUMNAS_TABLA_ANCHA = 8       # desde aquí la hoja pasa a horizontal y la tabla usa letra más chica

_AGREGACION = {"auto": "Automática", "suma": "Suma", "promedio": "Promedio", "ultimo": "Último valor",
               "max": "Máximo", "min": "Mínimo", "cuenta": "Cantidad de registros"}
_FRECUENCIA = {"auto": "Automático (según tus fechas)", "H": "Por hora", "D": "Por día", "W": "Por semana", "M": "Por mes"}
_MODO_EVENTO = {"repetido": "Repetido (tras el evento sigue, por ejemplo un equipo que se repara)",
                "unico": "Una sola vez (por ejemplo un cliente que se va)"}
_METODO_EVENTO = {"estadistico": "Estadístico (solo usa cuándo ocurren los eventos)",
                  "modelo": "Con modelo (además usa las columnas que explican)"}
_DIRECCION = {"auto": "Automática", "sube": "Que suba hasta el límite", "baja": "Que baje hasta el límite"}


@dataclass
class ContextoPregunta:
    """Foto de lo que se calculó (la pantalla puede cambiar después; el PDF describe lo que de verdad se calculó)."""
    config: dict
    n_filas: int = 0
    n_columnas: int = 0
    calculado_en: str = ""


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
def _segundos(x) -> np.ndarray:
    return pd.to_datetime(np.asarray(x)).values.astype("datetime64[ns]").astype("int64") / 1e9


def _linea(lienzo, xs, ys, color, ancho, X, Y, punteada=False):
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 2:
        return
    xs, ys = xs[ok], ys[ok]
    ruta = QPainterPath(QPointF(X(xs[0]), Y(ys[0])))
    for a, b in zip(xs[1:], ys[1:]):
        ruta.lineTo(QPointF(X(a), Y(b)))
    pen = QPen(QColor(color), ancho)
    if punteada:
        pen.setStyle(Qt.DashLine)
    lienzo.p.setPen(pen)
    lienzo.p.setBrush(Qt.NoBrush)
    lienzo.p.drawPath(ruta)


def _fig_serie(g) -> QImage:
    """Lo observado (gris), la proyección (violeta) con su banda de rango, la línea de tendencia y el límite."""
    W, H = ANCHO_FIG, 310
    izq, der, arr, aba = 74, 16, 26, 40
    c = _pp._Lienzo(W, H)
    p = c.p
    xh = _segundos(g["x_hist"])
    yh = np.asarray(g["y_hist"], dtype=float)
    xs, ys = [xh], [yh]
    fut = g.get("x_fut") is not None
    if fut:
        xf = _segundos(g["x_fut"])
        yf = np.asarray(g["y_fut"], dtype=float)
        yb, ya = np.asarray(g["y_bajo"], dtype=float), np.asarray(g["y_alto"], dtype=float)
        xs.append(xf)
        ys += [yf, yb, ya]
    if g.get("linea") is not None:
        lx, ly = g["linea"]
        lx, ly = _segundos(lx), np.asarray(ly, dtype=float)
        xs.append(lx)
        ys.append(ly)
    umbral = g.get("umbral")
    if umbral is not None:
        ys.append(np.array([float(umbral)]))
    todo_x = np.concatenate([a[np.isfinite(a)] for a in xs])
    todo_y = np.concatenate([a[np.isfinite(a)] for a in ys])
    if todo_x.size == 0 or todo_y.size == 0:
        raise ValueError("sin datos para dibujar")
    x0, x1 = float(todo_x.min()), float(todo_x.max())
    y0, y1 = float(todo_y.min()), float(todo_y.max())
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    pad = (y1 - y0) * 0.06
    y0p, y1p = y0 - pad, y1 + pad
    ancho, alto = W - izq - der, H - arr - aba

    def X(v):
        return izq + (v - x0) / (x1 - x0) * ancho

    def Y(v):
        return arr + alto - (v - y0p) / (y1p - y0p) * alto

    # banda del rango y líneas
    if fut:
        ok = np.isfinite(xf) & np.isfinite(yb) & np.isfinite(ya)
        if ok.sum() >= 2:
            poli = QPolygonF([QPointF(X(a), Y(b)) for a, b in zip(xf[ok], ya[ok])]
                             + [QPointF(X(a), Y(b)) for a, b in zip(xf[ok][::-1], yb[ok][::-1])])
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(79, 70, 229, 55))
            p.drawPolygon(poli)
    if umbral is not None:
        p.setPen(QPen(QColor(AMBAR), 1.0, Qt.DashLine))
        p.drawLine(QPointF(izq, Y(float(umbral))), QPointF(izq + ancho, Y(float(umbral))))
        p.setFont(_pp._fuente(11))
        p.setPen(QColor(AMBAR))
        p.drawText(QPointF(izq + 6, Y(float(umbral)) - 5), f"límite {num_txt(umbral)}")
    _linea(c, xh, yh, "#8b8f98", 1.3, X, Y)
    if g.get("linea") is not None:
        _linea(c, lx, ly, ACENTO, 1.2, X, Y, punteada=True)
    if fut:
        _linea(c, xf, yf, ACENTO, 1.9, X, Y)

    # ejes que cubren solo el rango de los datos
    p.setPen(QPen(QColor(GRIS), 0.9))
    p.drawLine(QPointF(izq - 6, Y(y0)), QPointF(izq - 6, Y(y1)))
    p.drawLine(QPointF(X(x0), arr + alto + 6), QPointF(X(x1), arr + alto + 6))
    p.setFont(_pp._fuente(11))
    fm = p.fontMetrics()
    con_hora = g.get("freq") == "H"
    for v in np.linspace(y0, y1, 5):
        p.setPen(QPen(QColor(GRIS), 0.9))
        p.drawLine(QPointF(izq - 10, Y(v)), QPointF(izq - 6, Y(v)))
        p.setPen(QColor(TINTA))
        t = num_txt(v)
        p.drawText(QPointF(izq - 14 - fm.horizontalAdvance(t), Y(v) + 4), t)
    n_ticks = 4 if con_hora else 5
    for i, v in enumerate(np.linspace(x0, x1, n_ticks)):
        p.setPen(QPen(QColor(GRIS), 0.9))
        p.drawLine(QPointF(X(v), arr + alto + 6), QPointF(X(v), arr + alto + 10))
        p.setPen(QColor(TINTA))
        t = fecha_txt(pd.Timestamp(v, unit="s"), con_hora=con_hora)
        tw = fm.horizontalAdvance(t)
        px = min(max(X(v) - tw / 2, 0), W - tw)
        p.drawText(QPointF(px, arr + alto + 26), t)

    # leyenda directa, arriba a la izquierda
    p.setFont(_pp._fuente(11))
    lx0, ly0 = izq, 12
    p.setPen(QPen(QColor("#8b8f98"), 1.6))
    p.drawLine(QPointF(lx0, ly0 - 3), QPointF(lx0 + 18, ly0 - 3))
    p.setPen(QColor(GRIS))
    p.drawText(QPointF(lx0 + 24, ly0), "observado")
    if fut:
        lx1 = lx0 + 24 + p.fontMetrics().horizontalAdvance("observado") + 22
        p.setPen(QPen(QColor(ACENTO), 2))
        p.drawLine(QPointF(lx1, ly0 - 3), QPointF(lx1 + 18, ly0 - 3))
        p.setPen(QColor(GRIS))
        p.drawText(QPointF(lx1 + 24, ly0), "proyección y rango probable")
    et = g.get("etiqueta_y")
    if et:
        p.setPen(QColor(GRIS))
        p.setFont(_pp._fuente(11))
        txt = str(et)
        p.drawText(QPointF(W - der - p.fontMetrics().horizontalAdvance(txt), ly0), txt)
    return c.fin()


def _fig_barras(g) -> QImage:
    """Una barra por posición del ciclo (día de la semana, mes, hora…)."""
    etiquetas = [str(x) for x in g["x"]]
    y = np.asarray(g["y"], dtype=float)
    relativo = bool(g.get("relativo"))
    fmt = (lambda v: pct_txt(v, 0, True)) if relativo else num_txt
    n = len(y)
    W, H = ANCHO_FIG, 300
    izq, der, arr, aba = 74, 16, 22, 46
    c = _pp._Lienzo(W, H)
    p = c.p
    lo, hi = min(float(np.nanmin(y)), 0.0), max(float(np.nanmax(y)), 0.0)
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo_p, hi_p = lo - pad, hi + pad
    ancho, alto = W - izq - der, H - arr - aba

    def Y(v):
        return arr + alto - (v - lo_p) / (hi_p - lo_p) * alto

    slot = ancho / max(n, 1)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(79, 70, 229, 200))
    for i, v in enumerate(y):
        if not np.isfinite(v):
            continue
        xc = izq + slot * (i + 0.5)
        top, bot = sorted((Y(v), Y(0.0)))
        p.drawRect(QRectF(xc - slot * 0.3, top, slot * 0.6, max(bot - top, 0.8)))
    p.setPen(QPen(QColor(GRIS), 0.9))
    p.drawLine(QPointF(izq, Y(0.0)), QPointF(izq + ancho, Y(0.0)))
    p.drawLine(QPointF(izq - 6, Y(lo)), QPointF(izq - 6, Y(hi)))
    p.setFont(_pp._fuente(11))
    fm = p.fontMetrics()
    for v in sorted({lo, 0.0, hi}):
        p.setPen(QPen(QColor(GRIS), 0.9))
        p.drawLine(QPointF(izq - 10, Y(v)), QPointF(izq - 6, Y(v)))
        p.setPen(QColor(TINTA))
        t = "0" if v == 0 else fmt(float(v))
        p.drawText(QPointF(izq - 14 - fm.horizontalAdvance(t), Y(v) + 4), t)
    # etiquetas de cada barra: si hay muchas, se muestran una de cada k
    salto = 1
    while salto < n and max(fm.horizontalAdvance(t) for t in etiquetas) + 6 > slot * salto:
        salto += 1
    p.setPen(QColor(TINTA))
    for i, t in enumerate(etiquetas):
        if i % salto:
            continue
        xc = izq + slot * (i + 0.5)
        t = fm.elidedText(t, Qt.ElideRight, int(slot * salto - 4))
        p.drawText(QPointF(xc - fm.horizontalAdvance(t) / 2, arr + alto + 20), t)
    et = g.get("etiqueta_y")
    if et:
        p.setPen(QColor(GRIS))
        p.drawText(QPointF(izq, 12), str(et))
    return c.fin()


def _figura(g):
    """Devuelve la imagen del gráfico, o None si no se puede dibujar (el resto del informe sale igual)."""
    if not g:
        return None
    try:
        tipo = g.get("tipo")
        if tipo == "serie":
            return _fig_serie(g)
        if tipo == "barras":
            return _fig_barras(g)
    except Exception:
        return None
    return None


# ----------------------------------------------------------------------
# Textos
# ----------------------------------------------------------------------
def _valor_campo(campo, tipo, valor, ayuda="") -> str:
    if campo == "entidad":
        return str(valor) if valor else "(todos juntos)"
    if valor is None or valor == "" or valor == []:
        return "(ninguna)" if tipo in ("fecha", "numerica", "columna", "multi_columna", "valor_de_columna") else "—"
    if campo == "agregacion":
        return _AGREGACION.get(valor, str(valor))
    if campo == "frecuencia":
        return _FRECUENCIA.get(valor, str(valor))
    if campo == "modo_evento":
        return _MODO_EVENTO.get(valor, str(valor))
    if campo == "metodo_evento":
        return _METODO_EVENTO.get(valor, str(valor))
    if campo == "direccion":
        return _DIRECCION.get(valor, str(valor))
    if isinstance(valor, (list, tuple)):
        return ", ".join(str(v) for v in valor)
    if tipo == "entero" and valor == 0:
        m = re.search(r"0 = ([^.]+)\.", ayuda or "")
        return f"0 ({m.group(1)})" if m else "0"
    if isinstance(valor, float):
        return f"{valor:g}"
    return str(valor)


def _celda(v, con_hora=True) -> str:
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        t = pd.Timestamp(v)
        return fecha_txt(t, con_hora=bool(con_hora and (t.hour or t.minute)))
    if isinstance(v, (bool, np.bool_)):
        return "Sí" if v else "No"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return num_txt(v)
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "—"
    return str(v)


def _es_numero(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _tabla_pdf(encabezados, filas, alinear, pt=10.0) -> str:
    """Tabla fina (solo líneas horizontales). `pt` baja la letra cuando hay muchas columnas."""
    p = _pp._p
    pad = 3 if pt >= 9 else 2
    out = [f'<table width="100%" cellspacing="0" cellpadding="{p(pad)}"><thead><tr>']
    for i, h in enumerate(encabezados):
        out.append(f'<th align="{alinear[i]}" style="color:{GRIS}; font-weight:600; font-size:{pt - 1:g}pt; '
                   f'border-bottom: {p(1)}px solid {GRIS};">{_pp._esc(h)}</th>')
    out.append("</tr></thead>")
    for fila in filas:
        out.append("<tr>")
        for i, celda in enumerate(fila):
            out.append(f'<td align="{alinear[i]}" style="font-size:{pt:g}pt; white-space:nowrap; '
                       f'border-bottom: {p(1)}px solid #eceef1;">{_pp._esc(celda)}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _es_ancha(r) -> bool:
    return r.tabla is not None and len(r.tabla) > 0 and r.tabla.shape[1] >= COLUMNAS_TABLA_ANCHA


# ----------------------------------------------------------------------
# Secciones
# ----------------------------------------------------------------------
def _html_cabecera(r, spec, ctx, modo, ahora) -> str:
    sub = "Informe completo: configuración y respuesta" if modo == "todo" else "Respuesta de Hadar"
    partes = [f'<p style="margin:0; font-size:20pt; color:{TINTA};">{_pp._esc(r.titulo)}</p>',
              f'<p style="margin-top:{_pp._p(2)}px; margin-bottom:{_pp._p(2)}px; font-size:11.5pt; color:{TINTA};">'
              f'{_pp._esc(spec.get("pregunta", ""))}</p>',
              f'<p style="margin-top:0; margin-bottom:{_pp._p(2)}px; font-size:10.5pt; color:{GRIS};">'
              f'{_pp._esc(sub)} · Hadar · {_pp._esc(ahora)}</p>']
    if modo == "resultados":
        linea = [f"Tabla: {ctx.config.get('tabla') or '—'}", f"{ctx.n_filas:,} filas"]
        partes.append(f'<p style="margin-top:0; font-size:9pt; color:{GRIS};">{_pp._esc(" · ".join(linea))}</p>')
    return "".join(partes)


def _html_configuracion(r, spec, ctx) -> str:
    cfg = ctx.config
    out = [_pp._titulo_seccion("1   Configuración")]
    out.append(_pp._subtitulo("Qué se preguntó"))
    filas = [("Pregunta", f"{spec.get('nombre', r.titulo)}: {spec.get('pregunta', '')}"),
             ("Tabla", cfg.get("tabla") or "—"),
             ("Tamaño de la tabla", f"{ctx.n_filas:,} filas · {ctx.n_columnas} columnas")]
    if ctx.calculado_en:
        filas.append(("Calculado el", ctx.calculado_en))
    out.append(_pp._tabla_clave_valor(filas))

    out.append(_pp._subtitulo("Con qué datos y opciones"))
    filas = []
    for campo in spec.get("campos", []):
        nombre = campo["campo"]
        filas.append((campo["etiqueta"], _valor_campo(nombre, campo["tipo"], cfg.get(nombre), campo.get("ayuda", ""))))
    if spec.get("mecanismo") == "serie" and cfg.get("nivel_intervalo"):
        filas.append(("Amplitud del rango", f"{float(cfg['nivel_intervalo']) * 100:.0f}% de probabilidad"))
    if filas:
        out.append(_pp._tabla_clave_valor(filas))
    else:
        out.append(_pp._parrafo("Esta pregunta no tiene opciones adicionales."))
    return "".join(out)


def _html_respuesta(r, ctx, modo, b: "_pp._Constructor") -> str:
    out = [_pp._titulo_seccion("2   Respuesta" if modo == "todo" else "Respuesta", alto_min=330)]
    color = COLOR_NIVEL.get(r.nivel, ACENTO)
    rotulo = ROTULO_NIVEL.get(r.nivel, "")

    # titular y confianza (el titular y las tres cifras no se separan entre hojas)
    _pp._TITULOS.append(((rotulo + r.titular).strip(), 280))
    out.append(f'<p style="margin-top:{_pp._p(8)}px; font-size:13pt; color:{TINTA};">'
               f'<span style="color:{color}; font-weight:600;">{_pp._esc(rotulo)}</span>{_pp._esc(r.titular)}</p>')
    if r.por_que_confianza or r.confianza:
        out.append(_pp._nota(f"Confianza {r.confianza}" + (f": {r.por_que_confianza}" if r.por_que_confianza else "")))

    # tres cifras
    tarjetas = list(r.tarjetas[:3])
    if tarjetas:
        ancho = f"{100 // len(tarjetas)}%"
        celdas = []
        for valor, titulo, explicacion in tarjetas:
            celdas.append(
                f'<td width="{ancho}" valign="top"><span style="font-size:22pt; color:{TINTA};">{_pp._esc(valor)}</span><br>'
                f'<span style="font-weight:600;">{_pp._esc(titulo)}</span><br>'
                f'<span style="font-size:9pt; color:{GRIS};">{_pp._esc(explicacion)}</span></td>')
        out.append(f'<table width="100%" cellspacing="0" cellpadding="{_pp._p(4)}" style="margin-top:{_pp._p(6)}px;"><tr>'
                   + "".join(celdas) + "</tr></table>")

    # explicación
    for frase in r.detalle:
        out.append(_pp._parrafo(frase))

    # gráfico
    img = _figura(r.grafico)
    if img is not None:
        tipo = (r.grafico or {}).get("tipo")
        if tipo == "serie":
            nota = ("Gris: lo observado. Violeta: lo que Hadar proyecta; la banda es el rango probable"
                    if (r.grafico or {}).get("x_fut") is not None else "Lo observado en el tiempo.")
        else:
            nota = (r.grafico or {}).get("etiqueta_y") or ""
        out.append('<p style="margin-top:%dpx; margin-bottom:0;"></p>' % _pp._p(4))
        out.append(b.figura(img, ANCHO_FIG))
        if tipo == "serie" and nota:
            out.append(_pp._nota(nota + "."))

    # tabla
    if r.tabla is not None and len(r.tabla):
        df = r.tabla.head(MAX_FILAS_TABLA)
        out.append(_pp._subtitulo(r.tabla_titulo or "Detalle", 150))
        encabezados = [str(c) for c in df.columns]
        ancha = _es_ancha(r)
        g = r.grafico or {}
        # En estacionalidad el efecto se dibuja en % (+35%): en la tabla también, para que coincidan.
        como_pct = {"Efecto"} if (g.get("tipo") == "barras" and g.get("relativo")) else set()

        def texto(i, j):
            v = df.iloc[i, j]
            if str(df.columns[j]) in como_pct and _es_numero(v):
                return pct_txt(float(v), 0, True)
            return _celda(v, con_hora=not ancha)

        filas = [tuple(texto(i, j) for j in range(df.shape[1])) for i in range(len(df))]
        alinear = ["right" if (len(df) and _es_numero(df.iloc[0, j])) else "left" for j in range(df.shape[1])]
        out.append(_tabla_pdf(encabezados, filas, alinear, pt=8.5 if ancha else 10.0))
        if len(r.tabla) > MAX_FILAS_TABLA:
            out.append(_pp._nota(f"Se muestran las primeras {MAX_FILAS_TABLA} filas de {len(r.tabla):,}."))

    # supuestos y avisos
    if r.supuestos:
        out.append(_pp._subtitulo("Qué asume Hadar", 100))
        out.append(_pp._nota(" ".join(r.supuestos)))
    for aviso in r.advertencias:
        out.append(f'<p style="margin-top:{_pp._p(2)}px; margin-bottom:{_pp._p(4)}px; font-size:9.5pt;">'
                   f'<span style="color:{AMBAR}; font-weight:600;">Ojo:</span> {_pp._esc(aviso)}</p>')
    return "".join(out)


# ----------------------------------------------------------------------
# Punto de entrada
# ----------------------------------------------------------------------
def exportar_pdf_pregunta(ruta: str, resultado, contexto: ContextoPregunta, modo: str = "todo") -> str:
    """Crea el PDF en `ruta`. `modo` = "todo" o "resultados"."""
    if modo not in ("todo", "resultados"):
        raise ValueError("modo debe ser «todo» o «resultados».")
    spec = PALABRAS_CLAVE.get(resultado.palabra) or PALABRAS_CLAVE.get(contexto.config.get("palabra"), {}) or {}
    ahora = datetime.now().strftime("%d/%m/%Y %H:%M")
    _pp._TITULOS.clear()
    doc = QTextDocument()
    base = QFont()
    base.setFamilies(_pp.FAMILIAS)
    base.setPointSizeF(10)
    doc.setDefaultFont(base)
    b = _pp._Constructor(doc)
    partes = [_html_cabecera(resultado, spec, contexto, modo, ahora)]
    if modo == "todo":
        partes.append(_html_configuracion(resultado, spec, contexto))
    partes.append(_html_respuesta(resultado, contexto, modo, b))
    doc.setHtml(f'<html><body style="color:{TINTA};">' + "".join(partes) + "</body></html>")
    _pp._escribir(doc, ruta, resultado.titulo, f"Hadar · {resultado.titulo}", horizontal=_es_ancha(resultado))
    return ruta
