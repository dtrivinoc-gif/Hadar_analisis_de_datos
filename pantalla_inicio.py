"""
Pantalla de inicio de Hadar: lo primero que se ve al abrir la app, antes
de la ventana grande de siempre -- igual que Word o Excel preguntan si
quieres un documento nuevo o seguir uno reciente.

Dos zonas, una al lado de la otra:
  - A la izquierda, la marca y las dos formas de empezar: "Nuevo Proyecto"
    (la acción principal, en el color de acento) y "Análisis Casual" (la
    secundaria, solo con borde).
  - A la derecha, tus proyectos recientes agrupados por cuándo los usaste
    (Hoy, Esta semana, Este mes, Antes), con sus datos en columnas alineadas.

Pensada para usarse con teclado y con mouse:
  Ctrl+N          nuevo proyecto            Ctrl+Mayús+N   análisis casual
  Ctrl+O          buscar otro proyecto      Ctrl+F         buscar en la lista
  ↑ ↓ y Enter     recorrer y abrir un proyecto reciente
  Supr            borrar el proyecto seleccionado (pide confirmación)
  Arrastrar un archivo .hadarproy sobre la ventana lo abre.
  El icono de papelera al final de cada fila también borra el proyecto: pregunta
  antes y manda el archivo a la papelera de reciclaje (se puede recuperar).

Uso típico, desde main.py (no cambió):

    app = QApplication(sys.argv)
    pantalla = PantallaInicio()
    if pantalla.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)   # el usuario cerró la pantalla de inicio sin elegir nada

    ventana = HadarApp()
    if pantalla.modo == "continuar" and pantalla.ruta_proyecto:
        ventana.abrir_proyecto_desde_ruta(pantalla.ruta_proyecto)
    ventana.show()
    sys.exit(app.exec())

Esta pantalla no sabe nada de pandas, tablas ni de cómo se abre un
proyecto -- solo pregunta y devuelve una decisión (`self.modo` y
`self.ruta_proyecto`). Quien la usa (main.py) es quien de verdad abre el
proyecto con HadarApp.abrir_proyecto_desde_ruta().

self.modo puede ser:
    "nuevo"     -- proyecto nuevo, pensado para guardarse y usarse seguido
    "casual"    -- análisis rápido y desechable, nunca se guarda ni tiene ML
    "continuar" -- retomar un .hadarproy existente (self.ruta_proyecto)
"""
import os
from datetime import datetime

from PySide6.QtCore import QEvent, QFile, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QWidget,
    QFileDialog, QListWidget, QListWidgetItem, QLineEdit, QMessageBox, QStyle, QStyledItemDelegate,
    QToolTip,
)

from .config import THEMES, COLOR_ACCENT, LOGO_PNG_PATH
from .proyecto import listar_proyectos_recientes, EXTENSION, ruta_carpeta_proyectos

# Palatino en Windows y Mac, Cambria si no está (mismo criterio que la
# pestaña Indicadores). Para la letra chica se usa la fuente normal de la app.
_SERIF = ["Palatino Linotype", "Book Antiqua", "Palatino", "Cambria", "Times New Roman"]
_SERIF_CSS = "'Palatino Linotype', 'Book Antiqua', Palatino, Cambria, 'Times New Roman', serif"

_ROL_INFO = Qt.ItemDataRole.UserRole + 1    # datos del proyecto (dict)
_ROL_GRUPO = Qt.ItemDataRole.UserRole + 2   # texto del encabezado de grupo (solo en los encabezados)

_MARGEN = 28            # margen izquierdo/derecho de la lista
_ALTO_FILA = 66
_ALTO_GRUPO = 42
_MIN_PARA_FILTRAR = 7   # con menos proyectos que esto no hace falta un buscador
_LADO_PAPELERA = 30     # cuadrito clicable del icono de borrar, al final de cada fila
_COLOR_PELIGRO = "#DC2626"


# ----------------------------------------------------------------------------
# Textos y utilidades
# ----------------------------------------------------------------------------
def _fecha_amigable(timestamp):
    """'hace 2 días', 'hoy', etc. -- sin librerías extra."""
    ahora = datetime.now()
    fecha = datetime.fromtimestamp(timestamp)
    dias = (ahora.date() - fecha.date()).days
    if dias <= 0:
        return f"hoy, {fecha.strftime('%H:%M')}"
    if dias == 1:
        return "ayer"
    if dias < 7:
        return f"hace {dias} días"
    if dias < 31:
        semanas = dias // 7
        return f"hace {semanas} semana{'s' if semanas > 1 else ''}"
    return fecha.strftime("%d-%m-%Y")


def _formato_tamano(num_bytes):
    """'300 KB', '7,4 MB', etc. -- sin librerías extra."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    kb = num_bytes / 1024
    if kb < 1024:
        return f"{kb:.0f} KB"
    return f"{kb / 1024:.1f}".replace(".", ",") + " MB"


def _plural(n, singular, plural):
    return f"{n} {singular if n == 1 else plural}"


def _grupo_de(timestamp):
    """En qué grupo de la lista cae un proyecto según cuándo se usó."""
    dias = (datetime.now().date() - datetime.fromtimestamp(timestamp).date()).days
    if dias <= 0:
        return "Hoy"
    if dias < 7:
        return "Esta semana"
    if dias < 31:
        return "Este mes"
    return "Antes"


def _serif(px, cursiva=False):
    f = QFont()
    f.setFamilies(_SERIF)
    f.setPixelSize(px)
    f.setItalic(cursiva)
    return f


def _sans(base, px, negrita=False):
    f = QFont(base)
    f.setPixelSize(px)
    f.setBold(negrita)
    return f


def _rgba(color, alfa):
    q = QColor(color)
    return f"rgba({q.red()},{q.green()},{q.blue()},{alfa})"


def _tinta_sobre(fondo):
    """Texto claro u oscuro, el que se lea mejor sobre ese color de fondo."""
    return "#111827" if QColor(fondo).lightness() > 150 else "#FFFFFF"


# ----------------------------------------------------------------------------
# Lista de proyectos: se dibuja a mano para tener columnas alineadas, estados
# de teclado y de mouse claros, y encabezados de grupo -- sin una caja por fila.
# ----------------------------------------------------------------------------
class _DelegadoProyectos(QStyledItemDelegate):
    _ANCHO_FECHA, _ANCHO_PESO, _ANCHO_TABLAS = 128, 74, 84

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.c = colors

    def sizeHint(self, option, index):
        alto = _ALTO_GRUPO if index.data(_ROL_GRUPO) else _ALTO_FILA
        return QSize(option.rect.width(), alto)

    @staticmethod
    def rect_papelera(r):
        """Dónde queda el icono de borrar dentro de una fila `r`. Lo usa tanto
        el dibujo como la lista (para saber si un clic cayó encima)."""
        return QRect(r.right() - _MARGEN - _LADO_PAPELERA + 6,
                     r.top() + (r.height() - _LADO_PAPELERA) // 2,
                     _LADO_PAPELERA, _LADO_PAPELERA)

    def _dibujar_papelera(self, p, rect, resaltada, fila_activa):
        """Papelera dibujada con trazos (no es un emoji ni una imagen): tapa con
        asa, cuerpo y dos rayitas. Apagada en reposo, más clara cuando el mouse
        está sobre la fila, y roja cuando está justo sobre el icono."""
        if resaltada:
            fondo = QColor(_COLOR_PELIGRO)
            fondo.setAlpha(34)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(fondo)
            p.drawRoundedRect(QRectF(rect), 6, 6)
        color = QColor(_COLOR_PELIGRO) if resaltada else QColor(self.c["muted"])
        if not resaltada and not fila_activa:
            color.setAlpha(120)
        lapiz = QPen(color, 1.5)
        lapiz.setCapStyle(Qt.PenCapStyle.RoundCap)
        lapiz.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(lapiz)
        p.setBrush(Qt.BrushStyle.NoBrush)

        centro = QRectF(rect).center()
        cx, cy = centro.x(), centro.y()
        p.drawLine(QPointF(cx - 6, cy - 4.5), QPointF(cx + 6, cy - 4.5))        # tapa
        asa = QPainterPath(QPointF(cx - 2, cy - 4.5))
        asa.lineTo(cx - 2, cy - 6.5)
        asa.lineTo(cx + 2, cy - 6.5)
        asa.lineTo(cx + 2, cy - 4.5)
        p.drawPath(asa)
        cuerpo = QPainterPath(QPointF(cx - 4.5, cy - 2.5))
        cuerpo.lineTo(cx - 3.8, cy + 6.5)
        cuerpo.lineTo(cx + 3.8, cy + 6.5)
        cuerpo.lineTo(cx + 4.5, cy - 2.5)
        p.drawPath(cuerpo)
        p.drawLine(QPointF(cx - 1.5, cy - 0.5), QPointF(cx - 1.5, cy + 4))      # rayitas
        p.drawLine(QPointF(cx + 1.5, cy - 0.5), QPointF(cx + 1.5, cy + 4))

    def paint(self, p, option, index):
        c = self.c
        r = option.rect
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        grupo = index.data(_ROL_GRUPO)
        if grupo:
            p.setFont(_sans(option.font, 12))
            p.setPen(QColor(c["muted"]))
            p.drawText(QRect(r.left() + _MARGEN, r.top(), r.width() - 2 * _MARGEN, r.height() - 8),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, grupo)
            p.restore()
            return

        info = index.data(_ROL_INFO)
        lista_con_foco = bool(option.widget and option.widget.hasFocus())
        seleccionada = bool(option.state & QStyle.StateFlag.State_Selected) and lista_con_foco
        encima = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if seleccionada or encima:
            tinte = QColor(c["text"])
            tinte.setAlpha(24 if seleccionada else 12)
            p.fillRect(r, tinte)
        if seleccionada:
            p.fillRect(QRect(r.left(), r.top(), 3, r.height()), QColor(COLOR_ACCENT))

        # Papelera al final de la fila; las columnas de datos van a su izquierda.
        papelera = self.rect_papelera(r)
        sobre_papelera = getattr(option.widget, "fila_papelera_hover", -1) == index.row()
        self._dibujar_papelera(p, papelera, sobre_papelera, seleccionada or encima)

        # Columnas de datos, alineadas a la derecha (de derecha a izquierda).
        p.setFont(_sans(option.font, 13))
        p.setPen(QColor(c["muted"]))
        textos = (
            _fecha_amigable(info["modificado"]),
            _formato_tamano(info.get("peso_bytes", 0)),
            _plural(info["n_tablas"], "tabla", "tablas"),
        )
        x = papelera.left() - 8
        for ancho, texto in zip((self._ANCHO_FECHA, self._ANCHO_PESO, self._ANCHO_TABLAS), textos):
            p.drawText(QRect(x - ancho, r.top(), ancho, r.height()),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, texto)
            x -= ancho
        borde_texto = x - 18

        # Nombre y, debajo, la tabla activa.
        izq = r.left() + _MARGEN
        ancho_texto = max(60, borde_texto - izq)
        f_nombre = _serif(17)
        fm = QFontMetrics(f_nombre)
        con_ml = bool(info.get("ml_activado"))
        nombre = fm.elidedText(info["nombre_archivo"], Qt.TextElideMode.ElideRight,
                               ancho_texto - (34 if con_ml else 0))
        sub = f"tabla activa: {info['tabla_activa']}" if info.get("tabla_activa") else ""
        y_nombre = r.top() + (13 if sub else (r.height() - fm.height()) // 2)
        p.setFont(f_nombre)
        p.setPen(QColor(c["text"]))
        p.drawText(izq, y_nombre + fm.ascent(), nombre)
        if sub:
            f_sub = _serif(12, cursiva=True)
            fm_sub = QFontMetrics(f_sub)
            p.setFont(f_sub)
            p.setPen(QColor(c["muted"]))
            p.drawText(izq, y_nombre + fm.height() + 2 + fm_sub.ascent(),
                       fm_sub.elidedText(sub, Qt.TextElideMode.ElideRight, ancho_texto))

        # Marca "ML" solo cuando el aprendizaje adaptativo está activado.
        if con_ml:
            etiqueta = QRect(izq + fm.horizontalAdvance(nombre) + 10,
                             y_nombre + (fm.height() - 16) // 2, 24, 16)
            p.setPen(QPen(QColor(COLOR_ACCENT), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(etiqueta, 3, 3)
            p.setFont(_sans(option.font, 10, negrita=True))
            p.drawText(etiqueta, Qt.AlignmentFlag.AlignCenter, "ML")

        # Línea fina entre filas.
        linea = QColor(c["border"])
        linea.setAlpha(110)
        p.setPen(linea)
        p.drawLine(r.left() + _MARGEN, r.bottom(), r.right() - _MARGEN, r.bottom())
        p.restore()


class _ListaProyectos(QListWidget):
    """La lista de siempre, más: sabe si el mouse está sobre la papelera de una
    fila, y en ese caso el clic NO abre el proyecto sino que emite
    `borrar_pedido(item)`. La tecla Supr hace lo mismo con la fila seleccionada.
    Esta lista nunca borra nada por sí sola: solo avisa; quien decide (y
    pregunta) es PantallaInicio."""

    borrar_pedido = Signal(object)   # el QListWidgetItem del proyecto

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.fila_papelera_hover = -1   # fila con el mouse sobre su papelera (la lee el delegado)
        self._presionado = None         # item donde se apretó el botón sobre la papelera

    def _item_con_papelera_en(self, pos):
        item = self.itemAt(pos)
        if item is None or not item.data(_ROL_INFO):
            return None
        if _DelegadoProyectos.rect_papelera(self.visualItemRect(item)).contains(pos):
            return item
        return None

    def mouseMoveEvent(self, event):
        item = self._item_con_papelera_en(event.position().toPoint())
        fila = self.row(item) if item is not None else -1
        if fila != self.fila_papelera_hover:
            self.fila_papelera_hover = fila
            self.viewport().setCursor(
                Qt.CursorShape.PointingHandCursor if item is not None else Qt.CursorShape.ArrowCursor)
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self.fila_papelera_hover != -1:
            self.fila_papelera_hover = -1
            self.viewport().unsetCursor()
            self.viewport().update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self._item_con_papelera_en(event.position().toPoint())
            if item is not None:
                self._presionado = item
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._presionado is not None:
            item, self._presionado = self._presionado, None
            if self._item_con_papelera_en(event.position().toPoint()) is item:
                self.borrar_pedido.emit(item)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            item = self.currentItem()
            if item is not None and item.data(_ROL_INFO):
                self.borrar_pedido.emit(item)
            return
        super().keyPressEvent(event)

    def viewportEvent(self, event):
        if event.type() == QEvent.Type.ToolTip:
            if self._item_con_papelera_en(event.pos()) is not None:
                QToolTip.showText(event.globalPos(), "Borrar proyecto (Supr)", self)
                return True
        return super().viewportEvent(event)


class _BotonEntrada(QPushButton):
    """Botón grande de la izquierda: título, una línea que explica y, a la
    derecha, el atajo de teclado. `primario` lo pinta con el color de acento;
    el secundario solo lleva borde."""

    def __init__(self, titulo, subtitulo, atajo, primario, colors, parent=None):
        super().__init__(parent)
        c = colors
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(78)
        self.setAutoDefault(False)   # el Enter lo maneja keyPressEvent (abajo)

        if primario:
            tinta = _tinta_sobre(COLOR_ACCENT)
            suave = _rgba(tinta, 0.78)
            fondo, borde = COLOR_ACCENT, COLOR_ACCENT
            encima = QColor(COLOR_ACCENT).lighter(112).name()
            tecla_borde = _rgba(tinta, 0.45)
        else:
            tinta, suave = c["text"], c["muted"]
            fondo, borde = "transparent", c["border"]
            encima = _rgba(c["text"], 0.06)
            tecla_borde = c["border"]
        self.setStyleSheet(
            f"QPushButton {{ background-color: {fondo}; border: 1px solid {borde}; border-radius: 8px; }}"
            f"QPushButton:hover {{ background-color: {encima}; }}"
            f"QPushButton:focus {{ border: 2px solid {c['text']}; }}"
        )

        fila = QHBoxLayout(self)
        fila.setContentsMargins(22, 12, 18, 12)
        columna = QVBoxLayout()
        columna.setSpacing(3)
        lbl_titulo = QLabel(titulo)
        lbl_titulo.setFont(_serif(20))
        lbl_titulo.setStyleSheet(f"color: {tinta}; background: transparent; border: none;")
        lbl_sub = QLabel(subtitulo)
        lbl_sub.setWordWrap(True)
        lbl_sub.setStyleSheet(f"color: {suave}; background: transparent; border: none; font-size: 12px;")
        columna.addWidget(lbl_titulo)
        columna.addWidget(lbl_sub)
        fila.addLayout(columna, 1)
        lbl_tecla = QLabel(atajo)
        lbl_tecla.setStyleSheet(
            f"color: {suave}; background: transparent; border: 1px solid {tecla_borde};"
            "border-radius: 4px; padding: 2px 6px; font-size: 11px;")
        fila.addWidget(lbl_tecla, 0, Qt.AlignmentFlag.AlignTop)
        # Los textos no deben "tapar" el clic del botón.
        for lbl in (lbl_titulo, lbl_sub, lbl_tecla):
            lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click()
            return
        super().keyPressEvent(event)


class _SueltaAqui(QWidget):
    """Aviso que cubre la lista mientras se arrastra un .hadarproy sobre la ventana."""

    def __init__(self, colors, parent):
        super().__init__(parent)
        self.c = colors
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fondo = QColor(self.c["card"])
        fondo.setAlpha(245)
        p.fillRect(self.rect(), fondo)
        p.setPen(QPen(QColor(COLOR_ACCENT), 1.6, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(self.rect().adjusted(20, 20, -20, -20), 10, 10)
        p.setPen(QColor(self.c["text"]))
        p.setFont(_serif(22))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Suelta el proyecto para abrirlo")
        p.end()


# ----------------------------------------------------------------------------
# La pantalla
# ----------------------------------------------------------------------------
class PantallaInicio(QDialog):
    """Ventana de bienvenida: Nuevo Proyecto / Análisis Casual, con la lista
    de proyectos recientes siempre visible a la derecha.

    Al cerrar con éxito (self.exec() == Accepted), revisa:
        self.modo           -- "nuevo", "casual" o "continuar"
        self.ruta_proyecto  -- ruta del .hadarproy elegido (solo si modo == "continuar")

    "Análisis Casual" es a propósito desechable: nunca llega a guardarse
    como proyecto, por eso nunca tiene ML ni aparece en esta lista.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.modo = None
        self.ruta_proyecto = None
        self.colors = THEMES["tactical"]
        c = self.colors

        self.setWindowTitle("Hadar Data Analytics")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setMinimumSize(800, 540)
        self.resize(940, 600)
        self.setAcceptDrops(True)
        self.setStyleSheet(f"QDialog {{ background-color: {c['bg']}; }}")

        raiz = QHBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(0)
        raiz.addWidget(self._construir_izquierda(), 4)
        self._der = self._construir_derecha()
        raiz.addWidget(self._der, 6)

        self._suelta_aqui = _SueltaAqui(c, self)

        # Atajos de teclado (en Mac, Qt los muestra y usa como ⌘).
        QShortcut(QKeySequence("Ctrl+N"), self, self._elegir_nuevo)
        QShortcut(QKeySequence("Ctrl+Shift+N"), self, self._elegir_casual)
        QShortcut(QKeySequence("Ctrl+O"), self, self._buscar_otro_proyecto)
        QShortcut(QKeySequence("Ctrl+F"), self, self._enfocar_buscador)

        self._proyectos = []
        self._refrescar_lista()

    # -- construcción ----------------------------------------------------------
    def _construir_izquierda(self):
        c = self.colors
        izq = QWidget()
        izq.setObjectName("izq")
        izq.setMinimumWidth(340)
        izq.setStyleSheet(f"QWidget#izq {{ background-color: {c['bg']}; }}")
        lay = QVBoxLayout(izq)
        lay.setContentsMargins(48, 48, 40, 44)
        lay.setSpacing(0)

        if os.path.exists(LOGO_PNG_PATH):
            lbl_logo = QLabel()
            dpr = self.devicePixelRatioF()
            pixmap = QPixmap(LOGO_PNG_PATH).scaledToWidth(
                int(250 * dpr), Qt.TransformationMode.SmoothTransformation)
            pixmap.setDevicePixelRatio(dpr)
            lbl_logo.setPixmap(pixmap)
            lbl_logo.setStyleSheet("background: transparent;")
            lay.addWidget(lbl_logo, 0, Qt.AlignmentFlag.AlignLeft)
        else:
            # Si falta el archivo del logo, se muestra el nombre escrito
            marca = QLabel("Hadar")
            marca.setFont(_serif(46))
            marca.setStyleSheet(f"color: {c['text']}; background: transparent;")
            lay.addWidget(marca)
            sub = QLabel("Data Analytics")
            sub.setStyleSheet(f"color: {c['muted']}; background: transparent; font-size: 14px;")
            lay.addWidget(sub)

        lay.addStretch(1)

        nativo = QKeySequence.SequenceFormat.NativeText
        self._btn_nuevo = _BotonEntrada(
            "Nuevo Proyecto", "Para un análisis que vas a seguir usando.",
            QKeySequence("Ctrl+N").toString(nativo), True, c)
        self._btn_nuevo.clicked.connect(self._elegir_nuevo)
        lay.addWidget(self._btn_nuevo)
        lay.addSpacing(12)
        self._btn_casual = _BotonEntrada(
            "Análisis Casual", "Revisión rápida y puntual. No se guarda.",
            QKeySequence("Ctrl+Shift+N").toString(nativo), False, c)
        self._btn_casual.clicked.connect(self._elegir_casual)
        lay.addWidget(self._btn_casual)
        return izq

    def _construir_derecha(self):
        c = self.colors
        der = QWidget()
        der.setObjectName("der")
        der.setStyleSheet(
            f"QWidget#der {{ background-color: {c['card']}; border-left: 1px solid {c['border']}; }}")
        lay = QVBoxLayout(der)
        lay.setContentsMargins(0, 44, 0, 20)
        lay.setSpacing(0)

        # Encabezado: título a la izquierda, "Abrir archivo…" a la derecha.
        cab = QHBoxLayout()
        cab.setContentsMargins(_MARGEN, 0, _MARGEN, 0)
        titulo = QLabel("Proyectos recientes")
        titulo.setFont(_serif(24))
        titulo.setStyleSheet(f"color: {c['text']}; background: transparent; border: none;")
        cab.addWidget(titulo)
        cab.addStretch(1)
        abrir = QLabel(
            f'<a href="abrir" style="color:{c["muted"]}; text-decoration:underline">Abrir archivo…</a>')
        abrir.setStyleSheet("background: transparent; border: none; font-size: 13px;")
        abrir.setToolTip(QKeySequence("Ctrl+O").toString(QKeySequence.SequenceFormat.NativeText))
        abrir.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        abrir.linkActivated.connect(lambda _h: self._buscar_otro_proyecto())
        cab.addWidget(abrir, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(cab)

        # Buscador: solo aparece si la lista ya es larga.
        self._buscador = QLineEdit()
        self._buscador.setPlaceholderText("Buscar un proyecto")
        self._buscador.setClearButtonEnabled(True)
        self._buscador.setAcceptDrops(False)
        self._buscador.setStyleSheet(
            "QLineEdit { background: transparent; border: none; "
            f"border-bottom: 1px solid {c['border']}; padding: 7px 0; color: {c['text']}; font-size: 14px; }}"
            f"QLineEdit:focus {{ border-bottom: 1px solid {c['text']}; }}")
        self._buscador.textChanged.connect(self._poblar_lista)
        self._buscador.returnPressed.connect(self._abrir_primero)
        caja_buscador = QHBoxLayout()
        caja_buscador.setContentsMargins(_MARGEN, 14, _MARGEN, 0)
        caja_buscador.addWidget(self._buscador)
        self._caja_buscador = QWidget()
        self._caja_buscador.setLayout(caja_buscador)
        self._caja_buscador.setStyleSheet("background: transparent;")
        lay.addWidget(self._caja_buscador)

        # La lista.
        self._lista = _ListaProyectos()
        self._lista.setFrameShape(QFrame.Shape.NoFrame)
        self._lista.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._lista.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._lista.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._lista.setItemDelegate(_DelegadoProyectos(c, self._lista))
        self._lista.setStyleSheet(
            "QListWidget { background: transparent; border: none; outline: none; }"
            "QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }"
            f"QScrollBar::handle:vertical {{ background: {_rgba(c['muted'], 0.45)}; border-radius: 3px; min-height: 30px; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }")
        self._lista.itemClicked.connect(self._abrir_item)
        self._lista.itemActivated.connect(self._abrir_item)   # Enter
        self._lista.borrar_pedido.connect(self._pedir_borrar)
        lay.addSpacing(6)
        lay.addWidget(self._lista, 1)

        # Estado vacío / sin coincidencias.
        self._lbl_vacio = QLabel()
        self._lbl_vacio.setWordWrap(True)
        self._lbl_vacio.setStyleSheet(
            f"color: {c['muted']}; background: transparent; border: none;"
            f"font-family: {_SERIF_CSS}; font-size: 16px; font-style: italic;")
        caja_vacio = QHBoxLayout()
        caja_vacio.setContentsMargins(_MARGEN, 28, _MARGEN, 0)
        caja_vacio.addWidget(self._lbl_vacio)
        self._caja_vacio = QWidget()
        self._caja_vacio.setLayout(caja_vacio)
        self._caja_vacio.setStyleSheet("background: transparent;")
        lay.addWidget(self._caja_vacio)
        lay.addStretch(0)

        # Pie: la pista de arrastrar y soltar.
        pie = QLabel(f"También puedes arrastrar un archivo {EXTENSION} sobre esta ventana.")
        pie.setStyleSheet(
            f"color: {c['muted']}; background: transparent; border: none; font-size: 12px; "
            f"font-family: {_SERIF_CSS}; font-style: italic; padding: 8px {_MARGEN}px 0 {_MARGEN}px;")
        lay.addWidget(pie)

        self.setTabOrder(self._btn_nuevo, self._btn_casual)
        self.setTabOrder(self._btn_casual, self._buscador)
        self.setTabOrder(self._buscador, self._lista)
        return der

    # -- la lista ------------------------------------------------------------------
    def _refrescar_lista(self):
        self._proyectos = listar_proyectos_recientes()
        self._caja_buscador.setVisible(len(self._proyectos) >= _MIN_PARA_FILTRAR)
        self._buscador.clear()
        self._poblar_lista()

    def _poblar_lista(self, _texto=None):
        texto = self._buscador.text().strip().lower()
        visibles = [p for p in self._proyectos if texto in p["nombre_archivo"].lower()]

        self._lista.clear()
        grupo_actual = None
        for info in visibles:
            grupo = _grupo_de(info["modificado"])
            if grupo != grupo_actual:
                grupo_actual = grupo
                cabecera = QListWidgetItem()
                cabecera.setData(_ROL_GRUPO, grupo)
                cabecera.setFlags(Qt.ItemFlag.NoItemFlags)   # no se puede elegir ni recorrer con flechas
                self._lista.addItem(cabecera)
            item = QListWidgetItem()
            item.setData(_ROL_INFO, info)
            item.setToolTip(
                f"{info['ruta']}\nModificado: {datetime.fromtimestamp(info['modificado']).strftime('%d-%m-%Y %H:%M')}"
                + ("\nAprendizaje adaptativo (ML) activado" if info.get("ml_activado") else ""))
            self._lista.addItem(item)

        hay = bool(visibles)
        self._lista.setVisible(hay)
        self._caja_vacio.setVisible(not hay)
        if not self._proyectos:
            self._lbl_vacio.setText(
                "Todavía no hay proyectos guardados. Cuando guardes uno, aparecerá aquí para retomarlo con un clic.")
        elif not hay:
            self._lbl_vacio.setText(f"Ningún proyecto se llama como «{self._buscador.text().strip()}».")
        if hay:
            self._seleccionar_primero()

    def _seleccionar_primero(self):
        for i in range(self._lista.count()):
            item = self._lista.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsSelectable:
                self._lista.setCurrentItem(item)
                return

    def _abrir_item(self, item):
        info = item.data(_ROL_INFO)
        if info:
            self._elegir_continuar(info["ruta"])

    def _abrir_primero(self):
        self._seleccionar_primero()
        item = self._lista.currentItem()
        if item is not None:
            self._abrir_item(item)

    # -- borrar un proyecto ---------------------------------------------------------------
    def _estilo_cajas(self):
        c = self.colors
        return (
            f"QMessageBox {{ background-color: {c['card']}; }}"
            f"QMessageBox QLabel {{ color: {c['text']}; background: transparent; }}"
            f"QPushButton {{ color: {c['text']}; background: transparent; border: 1px solid {c['border']};"
            "  border-radius: 6px; padding: 6px 18px; min-width: 70px; }"
            f"QPushButton:hover {{ background-color: {_rgba(c['text'], 0.08)}; }}"
            f"QPushButton:focus {{ border: 2px solid {c['text']}; }}")

    def _confirmar_borrado(self, nombre):
        caja = QMessageBox(self)
        caja.setIcon(QMessageBox.Icon.Warning)
        caja.setWindowTitle("Borrar proyecto")
        caja.setText(f"¿Seguro que quieres borrar «{nombre}»?")
        caja.setInformativeText("El archivo se enviará a la papelera de reciclaje.")
        caja.setStyleSheet(self._estilo_cajas())
        btn_borrar = caja.addButton("Borrar", QMessageBox.ButtonRole.DestructiveRole)
        btn_borrar.setStyleSheet(
            f"QPushButton {{ background-color: {_COLOR_PELIGRO}; color: #FFFFFF; border: 1px solid {_COLOR_PELIGRO};"
            "  border-radius: 6px; padding: 6px 18px; min-width: 70px; }"
            "QPushButton:hover { background-color: #B91C1C; }"
            "QPushButton:focus { border: 2px solid #FFFFFF; }")
        btn_cancelar = caja.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        caja.setDefaultButton(btn_cancelar)   # Enter sin pensar NO borra
        caja.setEscapeButton(btn_cancelar)
        caja.exec()
        return caja.clickedButton() is btn_borrar

    @staticmethod
    def _enviar_a_papelera(ruta):
        """True si el archivo quedó en la papelera del sistema. Se usa la
        papelera (no borrar a secas) para que un clic equivocado se pueda deshacer."""
        try:
            res = QFile.moveToTrash(ruta)
        except Exception:
            return False
        ok = res[0] if isinstance(res, tuple) else res   # según la versión de PySide6
        return bool(ok) and not os.path.exists(ruta)

    def _pedir_borrar(self, item):
        info = item.data(_ROL_INFO)
        if not info or not self._confirmar_borrado(info["nombre_archivo"]):
            return
        if not self._enviar_a_papelera(info["ruta"]):
            aviso = QMessageBox(self)
            aviso.setIcon(QMessageBox.Icon.Warning)
            aviso.setWindowTitle("No se pudo borrar")
            aviso.setText("No se pudo enviar el proyecto a la papelera.")
            aviso.setInformativeText("Revisa que no esté abierto en otro programa e inténtalo de nuevo.")
            aviso.setStyleSheet(self._estilo_cajas())
            aviso.exec()
            return

        self._proyectos = [p for p in self._proyectos if p["ruta"] != info["ruta"]]
        if len(self._proyectos) < _MIN_PARA_FILTRAR:
            # El buscador se esconde: que no quede un filtro invisible puesto.
            self._caja_buscador.setVisible(False)
            self._buscador.clear()
        self._lista.fila_papelera_hover = -1   # la fila que estaba bajo el mouse ya no existe
        self._lista.viewport().unsetCursor()
        self._poblar_lista()
        if self._lista.isVisible() and self._lista.count():
            self._lista.setFocus()
        else:
            self._btn_nuevo.setFocus()

    def _enfocar_buscador(self):
        if self._caja_buscador.isVisible():
            self._buscador.setFocus()
            self._buscador.selectAll()

    # -- decisiones -------------------------------------------------------------------
    def _elegir_nuevo(self):
        self.modo = "nuevo"
        self.accept()

    def _elegir_casual(self):
        self.modo = "casual"
        self.ruta_proyecto = None
        self.accept()

    def _elegir_continuar(self, ruta):
        if self.modo is not None:   # ya se eligió (p. ej. un doble clic dispara dos señales)
            return
        self.modo = "continuar"
        self.ruta_proyecto = ruta
        self.accept()

    def _buscar_otro_proyecto(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Buscar proyecto", ruta_carpeta_proyectos(), f"Proyecto Hadar (*{EXTENSION})"
        )
        if path:
            self._elegir_continuar(path)

    # -- foco inicial ---------------------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._foco_inicial)

    def _foco_inicial(self):
        # Con proyectos, el foco queda en el más reciente (Enter lo retoma);
        # sin ellos, en "Nuevo Proyecto".
        if self._lista.isVisible() and self._lista.count():
            self._lista.setFocus()
        else:
            self._btn_nuevo.setFocus()

    # -- arrastrar y soltar un .hadarproy --------------------------------------------------
    @staticmethod
    def _proyecto_en(mime):
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            ruta = url.toLocalFile()
            if ruta.lower().endswith(EXTENSION):
                return ruta
        return None

    def dragEnterEvent(self, event):
        if self._proyecto_en(event.mimeData()):
            self._suelta_aqui.setGeometry(self._der.geometry())
            self._suelta_aqui.raise_()
            self._suelta_aqui.show()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._suelta_aqui.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._suelta_aqui.hide()
        ruta = self._proyecto_en(event.mimeData())
        if ruta:
            event.acceptProposedAction()
            self._elegir_continuar(ruta)
        else:
            event.ignore()