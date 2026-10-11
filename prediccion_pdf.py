"""
prediccion_pdf.py — Exporta a PDF lo que muestra la pestaña Predicción.

Dos modos (los dos salen del mismo resultado ya calculado, no se reentrena):
  - "todo":        configuración (qué se predijo, cómo se probó, opciones, tablas
                   relacionadas, columnas usadas e ignoradas) + resultados.
  - "resultados":  solo los resultados del modelo: veredicto, cifras clave, todos
                   los indicadores, gráficos y casos para revisar.

No agrega ninguna librería: usa lo que PySide6 ya trae (QTextDocument +
QPdfWriter) y dibuja los gráficos con QPainter sobre imágenes. Por eso el .exe
no pesa más. El texto queda como texto en el PDF (se puede buscar y copiar).

Estilo: el mismo "poca tinta" de la pestaña (Tufte), pero en papel blanco y
con verde/ámbar/rojo un poco más oscuros para que se lean al imprimir.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PySide6.QtCore import QMarginsF, QPointF, QRectF, QSizeF, Qt, QUrl
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QPageLayout, QPageSize,
                           QPainter, QPdfWriter, QPen, QTextCursor, QTextDocument, QTextFormat, QTextTable)

try:
    from .prediccion import _fmt
except ImportError:  # pruebas sueltas fuera del paquete
    from prediccion import _fmt

# --- colores de impresión ----------------------------------------------
ACENTO = "#4f46e5"
VERDE = "#059669"
AMBAR = "#b45309"
ROJO = "#dc2626"
GRIS = "#6b7280"
TINTA = "#1f2937"
REGLA = "#c9ccd1"
COLOR_NIVEL = {"muy_bueno": VERDE, "bueno": VERDE, "regular": AMBAR, "no_aporta": ROJO, "sospechoso": AMBAR}
FAMILIAS = ["Palatino Linotype", "Book Antiqua", "Palatino", "Georgia", "DejaVu Serif", "Serif"]

# --- página ---------------------------------------------------------------
MARGEN_MM = 18
RESOLUCION = 192                # puntos por pulgada de la página
F = RESOLUCION / 96.0           # de «pixeles de pantalla» a pixeles de página
ESCALA_IMG = 3                  # las figuras se dibujan 3 veces más grandes para que salgan nítidas
ANCHO_FIG = 640                 # ancho de las figuras (en pixeles de pantalla; la hoja útil mide ~657)
ALTO_PIE = 22                   # espacio reservado abajo para el pie de página
FILAS_REVISION = 15
MAX_IGNORADAS = 60
MAX_COLUMNAS_TEXTO = 400

_VALIDACION = {
    "aleatoria": "Aleatoria (las filas se mezclan al azar)",
    "temporal": "Por fecha (siempre se prueba con filas más nuevas que las de entrenamiento)",
    "grupos": "Por grupos (un mismo grupo no queda a la vez en entrenamiento y en prueba)",
}


@dataclass
class ContextoPDF:
    """Foto de la configuración en el momento de entrenar (la pantalla puede
    cambiar después, el PDF debe describir lo que de verdad se entrenó)."""
    config: dict
    columnas_usadas: list = field(default_factory=list)
    columnas_ignoradas: list = field(default_factory=list)   # [(nombre, motivo)]
    marcadas_a_mano: list = field(default_factory=list)
    columnas_de_otras_tablas: list = field(default_factory=list)
    relaciones_activas: list = field(default_factory=list)   # ["Tabla.col ↔ Otra.col"]
    relaciones_info: str = ""


def _p(n) -> int:
    """Pixeles de pantalla -> pixeles de página."""
    return int(round(n * F))


def _esc(x) -> str:
    return html.escape(str(x))


# ----------------------------------------------------------------------
# Dibujo de figuras (QPainter sobre QImage)
# ----------------------------------------------------------------------
def _fuente(px, negrita=False) -> QFont:
    f = QFont()
    f.setFamilies(FAMILIAS)
    f.setPixelSize(int(px))
    f.setBold(negrita)
    return f


class _Lienzo:
    def __init__(self, ancho, alto):
        self.ancho, self.alto = ancho, alto
        self.img = QImage(int(ancho * ESCALA_IMG), int(alto * ESCALA_IMG), QImage.Format_ARGB32_Premultiplied)
        self.img.fill(QColor(255, 255, 255))
        self.p = QPainter(self.img)
        self.p.setRenderHint(QPainter.Antialiasing, True)
        self.p.setRenderHint(QPainter.TextAntialiasing, True)
        self.p.scale(ESCALA_IMG, ESCALA_IMG)

    def fin(self) -> QImage:
        self.p.end()
        return self.img


def _geometria_mancuerna(ancho, a, b, maximo, wa, wb):
    m = 10
    x0, x1 = m, max(ancho - m, m + 1)

    def X(v):
        return x0 + (x1 - x0) * min(max(v / maximo, 0.0), 1.0) if maximo > 0 else x0

    xa, xb = X(a), X(b)
    ta = min(max(xa - wa / 2, x0), max(x1 - wa, x0))
    tb = min(max(xb - wb / 2, x0), max(x1 - wb, x0))
    choque = not (ta + wa + 8 <= tb or tb + wb + 8 <= ta)
    return {"x0": x0, "x1": x1, "xa": xa, "xb": xb, "ta": ta, "tb": tb, "b_arriba": choque}


def _fig_mancuerna(a, b, maximo, texto_a, texto_b, brecha="") -> QImage:
    """Dos puntos sobre un eje: el modelo y la «respuesta fácil»."""
    W, H = ANCHO_FIG, 72
    c = _Lienzo(W, H)
    p = c.p
    p.setFont(_fuente(12))
    fm = p.fontMetrics()
    g = _geometria_mancuerna(W, a, b, maximo, fm.horizontalAdvance(texto_a), fm.horizontalAdvance(texto_b))
    y = 34
    p.setPen(QPen(QColor(GRIS), 0.8))
    p.drawLine(QPointF(g["x0"], y), QPointF(g["x1"], y))
    p.setPen(QPen(QColor(ACENTO), 2))
    p.drawLine(QPointF(g["xa"], y), QPointF(g["xb"], y))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(ACENTO))
    p.drawEllipse(QPointF(g["xa"], y), 5, 5)
    p.setBrush(QColor(GRIS))
    p.drawEllipse(QPointF(g["xb"], y), 5, 5)
    p.setPen(QColor(TINTA))
    p.drawText(QPointF(g["ta"], y + 26), texto_a)
    p.drawText(QPointF(g["tb"], (y - 12) if g["b_arriba"] else (y + 26)), texto_b)
    if brecha and not g["b_arriba"] and abs(g["xb"] - g["xa"]) > fm.horizontalAdvance(brecha) + 16:
        p.setPen(QColor(ACENTO))
        p.drawText(QPointF((g["xa"] + g["xb"]) / 2 - fm.horizontalAdvance(brecha) / 2, y - 10), brecha)
    return c.fin()


def _fig_puntos(filas, ancho_etiqueta=210) -> QImage:
    """Gráfico de puntos: etiqueta · línea punteada con un punto · valor.
    filas: [(etiqueta, porcentaje 0-100, color, texto_valor)]"""
    alto_fila = 22
    W, H = ANCHO_FIG, alto_fila * len(filas) + 8
    c = _Lienzo(W, H)
    p = c.p
    p.setFont(_fuente(12))
    fm = p.fontMetrics()
    x0, x1 = ancho_etiqueta + 8, W - 70
    for i, (etiqueta, pct, color, valor) in enumerate(filas):
        y = 4 + alto_fila * i + alto_fila / 2
        p.setPen(QColor(TINTA))
        p.drawText(QPointF(0, y + 4), fm.elidedText(str(etiqueta), Qt.ElideRight, ancho_etiqueta))
        p.setPen(QPen(QColor(REGLA), 1, Qt.DotLine))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        pct = max(0.0, min(float(pct), 100.0))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(x0 + (x1 - x0) * pct / 100.0, y), 4.2, 4.2)
        p.setPen(QColor(TINTA))
        p.drawText(QRectF(W - 62, y - 10, 62, 20), Qt.AlignRight | Qt.AlignVCenter, valor)
    return c.fin()


def _fig_predicho_real(reales, preds) -> QImage:
    """Cada punto es una fila de la parte de prueba; la diagonal es el acierto perfecto."""
    W, H = ANCHO_FIG, 372
    izq, der, arr, aba = 74, 14, 24, 44
    c = _Lienzo(W, H)
    p = c.p
    reales = np.asarray(reales, dtype=float)
    preds = np.asarray(preds, dtype=float)
    lo = float(min(reales.min(), preds.min()))
    hi = float(max(reales.max(), preds.max()))
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.03
    lo2, hi2 = lo - pad, hi + pad
    ancho, alto = W - izq - der, H - arr - aba

    def X(v):
        return izq + (v - lo2) / (hi2 - lo2) * ancho

    def Y(v):
        return arr + alto - (v - lo2) / (hi2 - lo2) * alto

    # puntos y diagonal
    p.setPen(QPen(QColor(GRIS), 0.9))
    p.drawLine(QPointF(X(lo), Y(lo)), QPointF(X(hi), Y(hi)))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(79, 70, 229, 120))
    for xr, yp in zip(reales, preds):
        p.drawEllipse(QPointF(X(xr), Y(yp)), 2.0, 2.0)
    # ejes que cubren solo el rango de los datos
    p.setPen(QPen(QColor(GRIS), 0.9))
    p.drawLine(QPointF(izq - 6, Y(lo)), QPointF(izq - 6, Y(hi)))
    p.drawLine(QPointF(X(lo), arr + alto + 6), QPointF(X(hi), arr + alto + 6))
    p.setFont(_fuente(11))
    fm = p.fontMetrics()
    for v in np.linspace(lo, hi, 5):
        p.setPen(QPen(QColor(GRIS), 0.9))
        p.drawLine(QPointF(izq - 10, Y(v)), QPointF(izq - 6, Y(v)))
        p.drawLine(QPointF(X(v), arr + alto + 6), QPointF(X(v), arr + alto + 10))
        p.setPen(QColor(TINTA))
        txt = _fmt(float(v))
        p.drawText(QPointF(izq - 14 - fm.horizontalAdvance(txt), Y(v) + 4), txt)
        p.drawText(QPointF(X(v) - fm.horizontalAdvance(txt) / 2, arr + alto + 24), txt)
    p.setPen(QColor(GRIS))
    p.setFont(_fuente(12))
    et = "valor real"
    p.drawText(QPointF(izq + ancho / 2 - p.fontMetrics().horizontalAdvance(et) / 2, H - 6), et)
    p.save()
    p.translate(13, arr + alto / 2)
    p.rotate(-90)
    et = "valor predicho"
    p.drawText(QPointF(-p.fontMetrics().horizontalAdvance(et) / 2, 0), et)
    p.restore()
    p.setPen(QColor(GRIS))
    p.setFont(_fuente(11))
    txt = "acierto perfecto"
    p.drawText(QPointF(X(hi) - p.fontMetrics().horizontalAdvance(txt) - 4, Y(hi) - 8), txt)
    return c.fin()


# ----------------------------------------------------------------------
# Piezas de HTML
# ----------------------------------------------------------------------
class _Constructor:
    def __init__(self, doc: QTextDocument):
        self.doc = doc
        self._n = 0

    def figura(self, img: QImage, ancho_logico: float) -> str:
        self._n += 1
        nombre = f"fig{self._n}"
        self.doc.addResource(QTextDocument.ImageResource, QUrl(nombre), img)
        alto_logico = ancho_logico * img.height() / img.width()
        # Qt ya escala el tamaño de las imágenes según la resolución de la hoja:
        # aquí van los tamaños «de pantalla», sin multiplicar por F.
        return f'<img src="{nombre}" width="{int(round(ancho_logico))}" height="{int(round(alto_logico))}">'


# Títulos de subsección que se escribieron: (texto, alto mínimo que debe quedar libre en la hoja,
# en pixeles de pantalla). Sirve para no dejar un título solo al final de una página.
_TITULOS: list = []


def _titulo_seccion(texto, pagina_nueva=False, alto_min=None) -> str:
    """`alto_min` (px de pantalla): si al final de la hoja queda menos espacio, el título pasa a la hoja siguiente."""
    salto = " page-break-before: always;" if pagina_nueva else ""
    if alto_min:
        _TITULOS.append((texto.upper(), alto_min))
    return (f'<table width="100%" cellspacing="0" cellpadding="0" style="margin-top:{_p(16)}px;{salto}"><tr>'
            f'<td style="border-top: {_p(1)}px solid {REGLA}; padding-top:{_p(6)}px;">'
            f'<span style="font-size:10.5pt; color:{GRIS}; font-weight:600; letter-spacing:{_p(1)}px;">'
            f'{_esc(texto.upper())}</span></td></tr></table>')


def _subtitulo(texto, alto_min=110) -> str:
    _TITULOS.append((texto, alto_min))
    return (f'<p style="margin-top:{_p(10)}px; margin-bottom:{_p(2)}px; font-size:10pt; font-weight:600; '
            f'color:{TINTA};">{_esc(texto)}</p>')


def _nota(texto, color=GRIS) -> str:
    return (f'<p style="margin-top:{_p(2)}px; margin-bottom:{_p(4)}px; font-size:9pt; color:{color};">'
            f'{_esc(texto)}</p>')


def _parrafo(texto) -> str:
    return f'<p style="margin-top:{_p(4)}px; margin-bottom:{_p(4)}px;">{_esc(texto)}</p>'


def _tabla_clave_valor(filas) -> str:
    out = [f'<table width="100%" cellspacing="0" cellpadding="{_p(3)}">']
    for k, v in filas:
        out.append(
            f'<tr><td width="36%" style="color:{GRIS}; border-bottom: {_p(1)}px solid #eceef1;">{_esc(k)}</td>'
            f'<td style="border-bottom: {_p(1)}px solid #eceef1;">{_esc(v)}</td></tr>')
    out.append("</table>")
    return "".join(out)


def _tabla(encabezados, filas, anchos=None, alinear=None, colores=None) -> str:
    """Tabla fina: solo líneas horizontales. `alinear` y `colores` son listas por columna."""
    n = len(encabezados)
    alinear = alinear or ["left"] * n
    out = [f'<table width="100%" cellspacing="0" cellpadding="{_p(3)}"><thead><tr>']
    for i, h in enumerate(encabezados):
        w = f' width="{anchos[i]}"' if anchos and anchos[i] else ""
        out.append(f'<th{w} align="{alinear[i]}" style="color:{GRIS}; font-weight:600; font-size:9pt; '
                   f'border-bottom: {_p(1)}px solid {GRIS};">{_esc(h)}</th>')
    out.append("</tr></thead>")
    for fila in filas:
        out.append("<tr>")
        for i, celda in enumerate(fila):
            color = (colores[i] if colores and colores[i] else None)
            estilo = f"border-bottom: {_p(1)}px solid #eceef1;" + (f" color:{color};" if color else "")
            out.append(f'<td align="{alinear[i]}" style="{estilo}">{_esc(celda)}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


# ----------------------------------------------------------------------
# Contenido
# ----------------------------------------------------------------------
def _pct(x, dec=1) -> str:
    return "—" if x is None else f"{x * 100:.{dec}f}%"


def _filas_metricas(r):
    """[(indicador, modelo, variación entre pruebas, respuesta fácil, modelo lineal)]"""
    m, b, sd = r.metricas, r.baseline, r.metricas_sd
    varias = r.n_pliegues > 1
    filas = []

    def mas_menos(clave, formato):
        if not varias or clave not in sd:
            return "—"
        return "± " + formato(sd[clave])

    if r.tipo == "regresion":
        filas.append(("Error típico (RMSE)", _fmt(m["rmse"]), mas_menos("rmse", _fmt),
                      _fmt(b.get("rmse")) if b.get("rmse") else "—",
                      _fmt(b["lineal_rmse"]) if b.get("lineal_rmse") else "—"))
        filas.append(("Error promedio absoluto (MAE)", _fmt(m["mae"]), mas_menos("mae", _fmt), "—", "—"))
        filas.append(("Variación explicada (R²)", f"{m['r2']:.3f}", mas_menos("r2", lambda v: f"{v:.3f}"), "—", "—"))
        return filas
    filas.append(("Aciertos", _pct(m["accuracy"]), mas_menos("accuracy", lambda v: f"{v * 100:.1f} pts"),
                  _pct(b.get("accuracy")), "—"))
    filas.append(("Aciertos parejos entre clases", _pct(m["bal_acc"]),
                  mas_menos("bal_acc", lambda v: f"{v * 100:.1f} pts"), _pct(b.get("bal_acc")),
                  _pct(b.get("lineal_bal_acc"))))
    filas.append(("Equilibrio entre clases (F1)", _pct(m["f1"]), mas_menos("f1", lambda v: f"{v * 100:.1f} pts"), "—", "—"))
    if m.get("auc") is not None:
        filas.append(("Separa las clases (AUC)", f"{m['auc']:.3f}", mas_menos("auc", lambda v: f"{v:.3f}"), "0.500 (azar)", "—"))
    if m.get("logloss") is not None:
        filas.append(("Pérdida logarítmica (menor es mejor)", f"{m['logloss']:.3f}",
                      mas_menos("logloss", lambda v: f"{v:.3f}"), "—", "—"))
    if m.get("recall_min") is not None:
        filas.append((f"Casos «{r.clase_minoritaria}» detectados", _pct(m["recall_min"]),
                      mas_menos("recall_min", lambda v: f"{v * 100:.1f} pts"), "—", "—"))
    if m.get("precision_min") is not None:
        filas.append((f"Alarmas «{r.clase_minoritaria}» correctas", _pct(m["precision_min"]),
                      mas_menos("precision_min", lambda v: f"{v * 100:.1f} pts"), "—", "—"))
    if m.get("ap") is not None:
        filas.append(("Precisión promedio de la clase rara", f"{m['ap']:.3f}",
                      mas_menos("ap", lambda v: f"{v:.3f}"), f"{b.get('ap', 0):.3f}" if b.get("ap") is not None else "—", "—"))
    return filas


def _texto_tipo(r, config) -> str:
    base = "Estimar un número (regresión)" if r.tipo == "regresion" else "Clasificar en categorías"
    return base + (" · elegido automáticamente" if config.get("tipo", "auto") == "auto" else "")


def _html_cabecera(r, ctx, modo, ahora) -> str:
    cfg = ctx.config
    titulo = f"Predicción de «{r.objetivo}»"
    sub = ("Informe completo: configuración y resultados" if modo == "todo"
           else "Resultados del modelo")
    partes = [f'<p style="margin:0; font-size:20pt; color:{TINTA};">{_esc(titulo)}</p>',
              f'<p style="margin-top:{_p(2)}px; margin-bottom:{_p(2)}px; font-size:10.5pt; color:{GRIS};">'
              f'{_esc(sub)} · Hadar · {_esc(ahora)}</p>']
    if modo == "resultados":
        linea = [f"Tabla de entrenamiento: {cfg.get('tabla_entrenamiento') or '—'}", _texto_tipo(r, cfg),
                 f"{r.n_entrenamiento:,} filas de entrenamiento · {r.n_validacion:,} de prueba"]
        partes.append(f'<p style="margin-top:0; font-size:9pt; color:{GRIS};">{_esc(" · ".join(linea))}</p>')
    return "".join(partes)


def _html_configuracion(r, ctx) -> str:
    cfg = ctx.config
    out = [_titulo_seccion("1   Configuración")]

    # --- qué se predijo
    out.append(_subtitulo("Qué se predijo"))
    filas = [
        ("Tabla de entrenamiento", cfg.get("tabla_entrenamiento") or "—"),
        ("Tabla de prueba (a la que se predice)", cfg.get("tabla_prueba") or "(ninguna)"),
        ("Columna a predecir", r.objetivo),
        ("Tipo de problema", _texto_tipo(r, cfg)),
    ]
    if r.tipo == "regresion":
        filas.append(("Escala logarítmica", "Sí" if r.log_objetivo else "No"))
    if r.tipo == "clasificacion" and r.clases:
        etiquetas = ", ".join(str(c) for c in r.clases[:12]) + ("…" if len(r.clases) > 12 else "")
        filas.append((f"Categorías ({len(r.clases)})", etiquetas))
    filas.append(("Qué se entrega al predecir",
                  "Probabilidades" if cfg.get("salida") == "probabilidad" else "Clase / valor"))
    filas.append(("Columna id del archivo de entrega", cfg.get("columna_id_salida") or "(automática)"))
    out.append(_tabla_clave_valor(filas))

    # --- cómo se probó
    out.append(_subtitulo("Cómo se probó"))
    modo = cfg.get("validacion", "aleatoria")
    filas = [("Forma de probar", _VALIDACION.get(modo, modo))]
    if modo == "temporal" and cfg.get("columna_fecha"):
        filas.append(("Columna de fecha", cfg["columna_fecha"]))
    if modo == "grupos" and cfg.get("columna_grupo"):
        filas.append(("Columna de grupos", cfg["columna_grupo"]))
    filas.append(("Cuántas pruebas se hicieron", str(r.n_pliegues)))
    filas.append(("Filas de entrenamiento por prueba (aprox.)", f"{r.n_entrenamiento:,}"))
    filas.append(("Filas de prueba (en total)", f"{r.n_validacion:,}"))
    if r.tipo == "clasificacion":
        pref = {"auto": "Automático", "si": "Siempre", "no": "Nunca"}.get(cfg.get("balancear", "auto"), "")
        filas.append(("Más peso a la clase poco frecuente",
                      ("Sí" if r.balanceado else "No") + (f" (ajuste: {pref.lower()})" if pref else "")))
    out.append(_tabla_clave_valor(filas))

    # --- opciones del modelo
    out.append(_subtitulo("Opciones del modelo"))
    prof = cfg.get("profundidad_max") or 0
    filas = [
        ("Modelo", "Árboles con boosting (HistGradientBoosting, scikit-learn)"),
        ("Iteraciones máximas", str(cfg.get("max_iter", "—"))),
        ("Iteraciones usadas en el modelo final", str(r.n_iteraciones)),
        ("Tasa de aprendizaje", str(cfg.get("tasa_aprendizaje", "—"))),
        ("Profundidad máxima de los árboles", str(prof) if prof else "automática"),
        ("Máximo de filas para entrenar", f"{int(cfg.get('max_filas', 0)):,}"),
        ("Semilla", str(cfg.get("semilla", 0))),
    ]
    out.append(_tabla_clave_valor(filas))

    # --- tablas relacionadas
    out.append(_subtitulo("Tablas relacionadas"))
    if cfg.get("usar_relaciones"):
        fr = cfg.get("fecha_relaciones")
        fecha = "(automática)" if not fr else ("(ninguna: todo el historial)" if fr == "__ninguna__" else fr)
        filas = [("Incluidas en el modelo", "Sí"), ("Saltos máximos entre tablas", str(cfg.get("profundidad_relaciones", 2))),
                 ("Fecha para resumir tablas con muchas filas", fecha)]
        out.append(_tabla_clave_valor(filas))
        if ctx.relaciones_activas:
            out.append(_nota("Relaciones usadas:"))
            out.append("<ul style=\"margin-top:0;\">" + "".join(f"<li>{_esc(t)}</li>" for t in ctx.relaciones_activas) + "</ul>")
        if ctx.relaciones_info:
            out.append(_nota(ctx.relaciones_info.replace("\n", " ")))
    else:
        out.append(_parrafo("No se usaron: el modelo trabajó solo con la tabla de entrenamiento."))

    # --- columnas
    usadas = list(ctx.columnas_usadas)
    out.append(_subtitulo(f"Columnas en que se fijó el modelo ({len(usadas)})"))
    otras = set(ctx.columnas_de_otras_tablas)
    propias = [c for c in usadas if c not in otras]
    ajenas = [c for c in usadas if c in otras]
    if propias:
        out.append(_parrafo(_lista_corta(propias)))
    if ajenas:
        out.append(_nota(f"Traídas de otras tablas ({len(ajenas)}):"))
        out.append(_parrafo(_lista_corta(ajenas)))
    if ctx.marcadas_a_mano:
        out.append(_nota("Marcadas a mano por ti: " + _lista_corta(ctx.marcadas_a_mano, 40)))
    ign = list(ctx.columnas_ignoradas)
    out.append(_subtitulo(f"Columnas que no se usaron ({len(ign)})"))
    if ign:
        filas = [(n, m or "—") for n, m in ign[:MAX_IGNORADAS]]
        out.append(_tabla(["Columna", "Motivo"], filas, anchos=["34%", "66%"]))
        if len(ign) > MAX_IGNORADAS:
            out.append(_nota(f"… y {len(ign) - MAX_IGNORADAS} más."))
    else:
        out.append(_parrafo("Todas las columnas se usaron."))
    return "".join(out)


def _lista_corta(nombres, maximo=MAX_COLUMNAS_TEXTO) -> str:
    nombres = [str(n) for n in nombres]
    if len(nombres) <= maximo:
        return ", ".join(nombres)
    return ", ".join(nombres[:maximo]) + f" … y {len(nombres) - maximo} más"


def _html_resultados(r, ctx, modo, rev, b: _Constructor) -> str:
    out = [_titulo_seccion("2   Resultados" if modo == "todo" else "Resultados", pagina_nueva=(modo == "todo"))]
    nivel, titulo, frase = r.veredicto()
    color = COLOR_NIVEL.get(nivel, GRIS)

    # 1) veredicto
    out.append(f'<p style="margin-top:{_p(8)}px; font-size:13pt; color:{TINTA};">'
               f'<span style="color:{color}; font-weight:600;">{_esc(titulo)}.</span> {_esc(frase)}</p>')
    for aviso in r.advertencias:
        out.append(_nota("Nota: " + aviso, AMBAR))

    # 2) tres cifras
    celdas = []
    for valor, tit, expl in r.tarjetas():
        celdas.append(
            f'<td width="33%" valign="top"><span style="font-size:24pt; color:{TINTA};">{_esc(valor)}</span><br>'
            f'<span style="font-weight:600;">{_esc(tit)}</span><br>'
            f'<span style="font-size:9pt; color:{GRIS};">{_esc(expl)}</span></td>')
    out.append(f'<table width="100%" cellspacing="0" cellpadding="{_p(4)}" style="margin-top:{_p(6)}px;"><tr>'
               + "".join(celdas) + "</tr></table>")

    # 3) todos los indicadores
    out.append(_subtitulo("Todos los indicadores", 150))
    out.append(_tabla(["Indicador", "Modelo", "Entre pruebas", "Respuesta fácil", "Modelo lineal"],
                      _filas_metricas(r), anchos=["34%", "16%", "16%", "18%", "16%"],
                      alinear=["left", "right", "right", "right", "right"]))
    out.append(_nota("«Respuesta fácil» es responder siempre el promedio (números) o la clase más frecuente "
                     "(categorías). «Entre pruebas» muestra cuánto cambia la cifra de una prueba a otra."))

    # 4) modelo frente a la respuesta fácil
    m, base = r.metricas, r.baseline
    out.append(_subtitulo("El modelo frente a la respuesta fácil", 140))
    if r.tipo == "regresion":
        rb = max(base.get("rmse", 0.0), 1e-12)
        brecha = f"{(1 - m['rmse'] / rb) * 100:.0f}% menos error" if m["rmse"] < rb else "sin mejora"
        out.append(_nota("Más a la izquierda = menos error."))
        out.append(b.figura(_fig_mancuerna(m["rmse"], rb, max(m["rmse"], rb), f"con el modelo  {_fmt(m['rmse'])}",
                                           f"respondiendo el promedio  {_fmt(rb)}", brecha), ANCHO_FIG))
    elif r.desbalanceado():
        bal, fb = m["bal_acc"] * 100, base["bal_acc"] * 100
        out.append(_nota("Con clases muy desparejas, acertar en general no mide nada: se compara cuánto acierta en "
                         "cada clase por igual. Más a la derecha = mejor."))
        out.append(b.figura(_fig_mancuerna(bal, fb, 100, f"modelo  {bal:.0f}%", f"siempre la clase más frecuente  {fb:.0f}%",
                                           f"+{bal - fb:.0f} puntos" if bal > fb else "sin mejora"), ANCHO_FIG))
    else:
        acc, fac = m["accuracy"] * 100, base["accuracy"] * 100
        out.append(_nota("Más a la derecha = más aciertos."))
        out.append(b.figura(_fig_mancuerna(acc, fac, 100, f"modelo  {acc:.0f}%", f"siempre la clase más frecuente  {fac:.0f}%",
                                           f"+{acc - fac:.0f} puntos" if acc > fac else "sin mejora"), ANCHO_FIG))
    ref = r.texto_lineal()
    if ref:
        out.append(_nota(ref))

    # 5) resultado en cada prueba
    if r.n_pliegues > 1 and len(r.mejoras_pliegue) > 1:
        out.append(_subtitulo("Resultado en cada prueba", 90 + 22 * min(r.n_pliegues, 10)))
        out.append(_nota("Cuánto del error de la respuesta fácil evita el modelo en cada prueba (0% = nada, 100% = todo). "
                         "Si los puntos están muy separados, el resultado es aproximado."))
        filas = [(f"Prueba {i + 1}", v * 100, ACENTO, f"{v * 100:.0f}%") for i, v in enumerate(r.mejoras_pliegue)]
        out.append(b.figura(_fig_puntos(filas, ancho_etiqueta=120), ANCHO_FIG))

    # 6) columnas
    if r.importancias:
        out.append(_subtitulo("En qué columnas se fijó más", 90 + 22 * min(len(r.importancias), 12)))
        out.append(_nota("Cuánto ayudó cada columna a adivinar. Todas suman 100%."))
        filas = [(str(c), pct, ACENTO, "<1%" if pct < 1 else f"{pct:.0f}%") for c, pct in r.importancias[:12]]
        out.append(b.figura(_fig_puntos(filas), ANCHO_FIG))
        if len(r.importancias) > 12:
            out.append(_nota(f"Se muestran las 12 primeras de {len(r.importancias)}."))

    # 7) predicho frente a real
    if r.tipo == "regresion" and r.validacion is not None:
        try:
            reales, preds = r.validacion
            out.append(_subtitulo("Lo predicho frente a lo real", 460))
            out.append(_nota("Cada punto es una fila de la parte de prueba. Mientras más cerca de la línea, mejor acertó."))
            out.append(b.figura(_fig_predicho_real(reales, preds), ANCHO_FIG))
        except Exception:
            pass   # el gráfico es un extra: si falla, el resto del informe sale igual

    # 8) casos para revisar
    out.append(_html_revision(r, rev))
    out.append(_nota(r.texto_validacion()))
    return "".join(out)


def _html_revision(r, rev) -> str:
    if not r.revision or not rev:
        return ""
    cols = list(rev.items())
    if len(cols) < 4:
        return ""
    n = min(len(cols[0][1]), FILAS_REVISION)
    if n == 0:
        return ""

    def texto(v):
        if isinstance(v, str):
            return v
        try:
            return _fmt(float(v))
        except Exception:
            return str(v)

    regresion = r.revision["tipo"] == "regresion"
    out = [_subtitulo("Casos para revisar primero", 150)]
    if regresion:
        out.append(_nota("Las filas donde el modelo más se equivocó (en la parte de prueba). Verde: el valor real fue mayor "
                         "de lo esperado; rojo: menor. Pueden ser errores de registro, casos especiales u oportunidades."))
        encabezados = [cols[0][0], "Real", "Predicho", "Diferencia"]
    else:
        out.append(_nota("Filas donde el modelo se equivocó estando muy seguro. Conviene mirar si el dato real está bien "
                         "registrado o si es un caso especial."))
        encabezados = [cols[0][0], "Real", "Predijo", "Seguridad del modelo"]
    filas, colores_fila = [], []
    for i in range(n):
        if regresion:
            d = float(cols[3][1][i])
            filas.append((str(cols[0][1][i]), texto(cols[1][1][i]), texto(cols[2][1][i]),
                          ("+" if d > 0 else "−") + _fmt(abs(d))))
        else:
            filas.append((str(cols[0][1][i]), texto(cols[1][1][i]), texto(cols[2][1][i]),
                          f"{float(cols[3][1][i]) * 100:.0f}%"))
    # el color de la última columna depende de la fila: se arma a mano
    out.append(_tabla_revision(encabezados, filas, regresion))
    return "".join(out)


def _tabla_revision(encabezados, filas, regresion) -> str:
    out = [f'<table width="100%" cellspacing="0" cellpadding="{_p(3)}"><thead><tr>']
    for i, h in enumerate(encabezados):
        out.append(f'<th align="{"left" if i == 0 else "right"}" style="color:{GRIS}; font-weight:600; font-size:9pt; '
                   f'border-bottom: {_p(1)}px solid {GRIS};">{_esc(h)}</th>')
    out.append("</tr></thead>")
    for fila in filas:
        out.append("<tr>")
        for i, celda in enumerate(fila):
            color = ""
            if i == 3:
                color = f" color:{VERDE if celda.startswith('+') else ROJO};" if regresion else f" color:{AMBAR};"
            out.append(f'<td align="{"left" if i == 0 else "right"}" style="border-bottom: {_p(1)}px solid #eceef1;{color}">'
                       f'{_esc(celda)}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


# ----------------------------------------------------------------------
# Escribir el PDF (con paginación propia para poner número de página)
# ----------------------------------------------------------------------
def _evitar_titulos_huerfanos(doc: QTextDocument, cuerpo: float):
    """Si un título quedaría con menos espacio libre del necesario al final de la hoja,
    lo pasa a la hoja siguiente (así no queda el título arriba y su gráfico abajo).
    Sirve para subtítulos (párrafos) y para títulos de sección (tablas de una celda)."""
    def limpio(t):                       # el HTML junta los espacios repetidos: se compara igual
        return " ".join(str(t).split())

    pendientes = {limpio(t): a for t, a in _TITULOS}
    if not pendientes:
        return
    capa = doc.documentLayout()
    candidatos = []
    bloque = doc.begin()
    while bloque.isValid():
        texto = limpio(bloque.text())
        if texto in pendientes and QTextCursor(bloque).currentTable() is None:
            candidatos.append((bloque.position(), "bloque", bloque, pendientes[texto]))
        bloque = bloque.next()
    for marco in doc.rootFrame().childFrames():
        if isinstance(marco, QTextTable) and marco.rows() == 1 and marco.columns() == 1:
            texto = limpio(marco.cellAt(0, 0).firstCursorPosition().block().text())
            if texto in pendientes:
                candidatos.append((marco.firstPosition(), "marco", marco, pendientes[texto]))
    candidatos.sort(key=lambda c: c[0])
    for _pos, tipo, obj, alto in candidatos:
        doc.pageCount()                                       # fuerza a recalcular el diseño
        arriba = (capa.blockBoundingRect(obj) if tipo == "bloque" else capa.frameBoundingRect(obj)).top() % cuerpo
        if arriba > 1 and cuerpo - arriba < alto * F:
            if tipo == "bloque":
                fmt = obj.blockFormat()
                fmt.setPageBreakPolicy(QTextFormat.PageBreak_AlwaysBefore)
                QTextCursor(obj).setBlockFormat(fmt)
            else:
                fmt = obj.format()
                fmt.setPageBreakPolicy(QTextFormat.PageBreak_AlwaysBefore)
                obj.setFormat(fmt)


def _escribir(doc: QTextDocument, ruta: str, titulo: str, pie_izq: str, horizontal: bool = False):
    w = QPdfWriter(ruta)
    w.setPageSize(QPageSize(QPageSize.A4))
    if horizontal:
        w.setPageOrientation(QPageLayout.Landscape)
    w.setPageMargins(QMarginsF(MARGEN_MM, MARGEN_MM, MARGEN_MM, MARGEN_MM), QPageLayout.Millimeter)
    w.setResolution(RESOLUCION)
    w.setTitle(titulo)
    w.setCreator("Hadar")
    ancho, alto = float(w.width()), float(w.height())
    cuerpo = alto - _p(ALTO_PIE)
    # La página se mide en pixeles de la propia hoja: así el resultado no depende de
    # la escala de pantalla de cada computador.
    doc.documentLayout().setPaintDevice(w)
    doc.setDocumentMargin(0)
    doc.setPageSize(QSizeF(ancho, cuerpo))
    _evitar_titulos_huerfanos(doc, cuerpo)
    paginas = max(doc.pageCount(), 1)

    p = QPainter()
    if not p.begin(w):
        raise RuntimeError("No se pudo escribir el archivo. Si lo tienes abierto en otro programa, ciérralo "
                           "e inténtalo de nuevo.")
    try:
        for i in range(paginas):
            if i:
                w.newPage()
            p.save()
            p.translate(0, -i * cuerpo)
            doc.drawContents(p, QRectF(0, i * cuerpo, ancho, cuerpo))
            p.restore()
            p.save()
            p.setPen(QColor(GRIS))
            f = QFont()
            f.setFamilies(FAMILIAS)
            f.setPointSizeF(8)
            p.setFont(f)
            caja = QRectF(0, cuerpo + _p(4), ancho, _p(ALTO_PIE - 6))
            p.drawText(caja, Qt.AlignLeft | Qt.AlignVCenter, pie_izq)
            p.drawText(caja, Qt.AlignRight | Qt.AlignVCenter, f"Página {i + 1} de {paginas}")
            p.restore()
    finally:
        p.end()


def exportar_pdf(ruta: str, resultado, contexto: ContextoPDF, modo: str = "todo", revision=None) -> str:
    """Crea el PDF en `ruta`. `modo` = "todo" o "resultados". `revision` es la
    lista de casos para revisar que ya muestra la pantalla (dict de listas)."""
    if modo not in ("todo", "resultados"):
        raise ValueError("modo debe ser «todo» o «resultados».")
    ahora = datetime.now().strftime("%d/%m/%Y %H:%M")
    _TITULOS.clear()
    doc = QTextDocument()
    base = QFont()
    base.setFamilies(FAMILIAS)
    base.setPointSizeF(10)
    doc.setDefaultFont(base)
    b = _Constructor(doc)
    partes = [_html_cabecera(resultado, contexto, modo, ahora)]
    if modo == "todo":
        partes.append(_html_configuracion(resultado, contexto))
    partes.append(_html_resultados(resultado, contexto, modo, revision, b))
    doc.setHtml(f'<html><body style="color:{TINTA};">' + "".join(partes) + "</body></html>")
    titulo = f"Predicción de {resultado.objetivo}"
    _escribir(doc, ruta, titulo, f"Hadar · {titulo}")
    return ruta