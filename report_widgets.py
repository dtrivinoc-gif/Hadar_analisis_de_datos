"""
Widgets gráficos de la pestaña "Reporte": el ítem de texto editable
(EditableTextItem) y las cajas de contenido movibles/redimensionables
(ReportBoxItem: texto, gráfico, indicador, imagen, cálculo, etc.).
"""
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPen, QBrush, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsPixmapItem,
    QMenu, QMessageBox,
)

from .config import COLOR_ACCENT_2

REPORTE_PAPEL_BG = "#FFFFFF"
REPORTE_PAPEL_BORDE = "#D8DCE8"
REPORTE_PAPEL_TINTA = "#1A1A1A"
REPORTE_PAPEL_MUTED = "#8A8F9C"
REPORTE_COLOR_ROJO = "#DC2626"

# Opciones de formato de texto deliberadamente acotadas (espíritu Hadar: lo
# necesario, no todo lo que existe en Word).
REPORTE_FUENTE_SERIFA = "Georgia"   # la misma serifa de la pestaña Gráficos
REPORTE_FUENTES = [REPORTE_FUENTE_SERIFA, "Arial", "Calibri", "Times New Roman"]
REPORTE_TAMANOS = [10, 12, 14, 18, 24, 32]

# Estilo Tufte: la "tinta" son los datos, no el marco. Con True, las cajas no
# llevan borde en reposo; el marco, el tirador de redimensionar y el botón de
# actualizar aparecen solo al pasar el mouse o al seleccionar la caja.
# Poner False para volver al marco fijo de antes.
REPORTE_CAJA_SOLO_AL_USAR = True


class EditableTextItem(QGraphicsTextItem):
    """Texto editable con doble clic; al perder el foco vuelve a modo
    'solo lectura' para no interferir con mover/redimensionar el recuadro."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_focus_in = None   # callback sin argumentos, para mostrar la barra de formato
        self.on_focus_out = None  # callback sin argumentos, para ocultarla

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self.on_focus_in:
            self.on_focus_in()

    def focusOutEvent(self, event):
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        if self.on_focus_out:
            self.on_focus_out()
        super().focusOutEvent(event)


class ReportBoxItem(QGraphicsRectItem):
    """Recuadro movible y redimensionable del lienzo de Reporte.

    kind='texto'  -> caja de texto editable con forma de fondo (rectángulo u óvalo).
    kind='imagen' -> foto (snapshot) de un gráfico, indicador o narrativa. Queda congelada
                     hasta que el usuario aprieta el botón de actualizar, que vuelve a
                     llamar a `regenerar` (una función sin argumentos que
                     entrega un QPixmap actualizado). Si se entrega `titulo`
                     (por ejemplo el nombre que el usuario le puso al gráfico),
                     se muestra como encabezado fijo sobre la imagen.
    """
    HANDLE = 14

    def __init__(self, kind, rect: QRectF, colors, ventana_principal=None, con_refresh=True, titulo=None):
        super().__init__(0, 0, rect.width(), rect.height())
        self._hover = False
        self.kind = kind
        self.shape_kind = "rect"
        self.colors = colors
        self.ventana_principal = ventana_principal
        self.regenerar = None  # función sin argumentos -> QPixmap (solo kind='imagen')

        self.setPos(rect.topLeft())
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        # El lienzo de Reporte es "papel": estos colores son fijos, no siguen
        # el tema claro/oscuro de la app.
        self.setPen(QPen(QColor(REPORTE_PAPEL_BORDE), 1))
        self.setBrush(QBrush(QColor(REPORTE_PAPEL_BG)))
        self.setAcceptHoverEvents(True)

        self.text_item = None
        self.pixmap_item = None
        self.titulo_item = None
        self.btn_refresh = None
        self._resizing = False
        self._imagen_original = None
        # Solo relevantes para kind == 'calculo' (Métrica/Indicador/Cálculo):
        # guardan el último título y valor numérico crudo, para poder elegir
        # este recuadro como fuente al combinar dos cálculos entre sí.
        self.titulo_calculo = None
        self.valor_crudo = None

        fuente_base = QFont(REPORTE_FUENTE_SERIFA, 12)

        if kind == "texto":
            self.text_item = EditableTextItem(self)
            self.text_item.setDefaultTextColor(QColor(REPORTE_PAPEL_TINTA))
            self.text_item.setFont(fuente_base)
            self.text_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            self.text_item.setPos(8, 8)
            self.text_item.setTextWidth(max(20, rect.width() - 16))
            self.text_item.setPlainText("Escribe aquí (doble clic para editar)…")
            if ventana_principal is not None:
                self.text_item.on_focus_in = lambda: ventana_principal._reporte_mostrar_barra_texto(self)
                self.text_item.on_focus_out = lambda: ventana_principal._reporte_ocultar_barra_texto(self)
        elif kind in ("calculo", "lista"):
            self.text_item = QGraphicsTextItem(self)
            self.text_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            self.text_item.setPos(8, 8)
            self.text_item.setTextWidth(max(20, rect.width() - 16))
            self._crear_boton_refresh(rect)
        else:  # 'imagen'
            self.pixmap_item = QGraphicsPixmapItem(self)
            if titulo:
                self.titulo_item = QGraphicsTextItem(titulo, self)
                self.titulo_item.setDefaultTextColor(QColor(REPORTE_PAPEL_TINTA))
                fuente_titulo = QFont(REPORTE_FUENTE_SERIFA)
                fuente_titulo.setPointSize(10)
                self.titulo_item.setFont(fuente_titulo)
                self.titulo_item.setPos(6, 2)
                self.titulo_item.setTextWidth(max(20, rect.width() - 30))
            self.pixmap_item.setPos(4, self._alto_titulo())
            if con_refresh:  # las imágenes subidas a mano no llevan botón de actualizar
                self._crear_boton_refresh(rect)
        self._actualizar_cromo()

    # -- "cromo" (marco, tirador, botón de actualizar): solo al usar ----
    def _cromo_visible(self):
        return (not REPORTE_CAJA_SOLO_AL_USAR) or self._hover or self.isSelected()

    def _actualizar_cromo(self):
        if self.btn_refresh is not None:
            self.btn_refresh.setVisible(self._cromo_visible())
        self.update()

    def hoverEnterEvent(self, event):
        self._hover = True
        self._actualizar_cromo()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self._actualizar_cromo()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._actualizar_cromo()
        return super().itemChange(change, value)

    def _alto_titulo(self):
        return 24 if self.titulo_item is not None else 4

    def _crear_boton_refresh(self, rect):
        self.btn_refresh = QGraphicsTextItem("↻", self)
        self.btn_refresh.setDefaultTextColor(QColor(COLOR_ACCENT_2))
        fuente = QFont()
        fuente.setPointSize(13)
        self.btn_refresh.setFont(fuente)
        self.btn_refresh.setPos(rect.width() - 26, 2)
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)

    # -- contenido -----------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap):
        if self.pixmap_item is None or pixmap is None or pixmap.isNull():
            return
        self._imagen_original = pixmap
        self._reescalar_imagen()

    def _reescalar_imagen(self):
        if self.pixmap_item is None or getattr(self, "_imagen_original", None) is None:
            return
        r = self.rect()
        top = self._alto_titulo()
        w = max(10, int(r.width() - 8))
        h = max(10, int(r.height() - top - 4))
        escalado = self._imagen_original.scaled(
            w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.pixmap_item.setPixmap(escalado)
        self.pixmap_item.setPos(4, top)

    def _reposicionar_contenido(self):
        r = self.rect()
        if self.text_item is not None:
            self.text_item.setTextWidth(max(20, r.width() - 16))
        if self.titulo_item is not None:
            self.titulo_item.setTextWidth(max(20, r.width() - 30))
        self._reescalar_imagen()
        if self.btn_refresh is not None:
            self.btn_refresh.setPos(r.width() - 26, 2)

    def set_calculo_text(self, titulo, valor_str, valor_crudo=None):
        if self.text_item is None:
            return
        self.titulo_calculo = titulo
        self.valor_crudo = valor_crudo
        color_titulo = REPORTE_PAPEL_MUTED
        color_valor = REPORTE_PAPEL_TINTA
        html = (
            f"<div style=\"text-align:center; font-family:'{REPORTE_FUENTE_SERIFA}'\">"
            f"<span style='font-size:9pt;color:{color_titulo}'>{titulo}</span><br>"
            f"<span style='font-size:24pt;color:{color_valor}'>{valor_str}</span>"
            "</div>"
        )
        self.text_item.setHtml(html)
        self.text_item.setTextWidth(max(20, self.rect().width() - 16))

    def set_lista_text(self, titulo, cuerpo_html):
        if self.text_item is None:
            return
        color_titulo = REPORTE_PAPEL_MUTED
        color_texto = REPORTE_PAPEL_TINTA
        html = (
            f"<div style=\"font-family:'{REPORTE_FUENTE_SERIFA}'\">"
            f"<span style='font-size:10pt;color:{color_titulo}'>{titulo}</span><br>"
            f"<span style='font-size:9pt;color:{color_texto}'>{cuerpo_html}</span>"
            "</div>"
        )
        self.text_item.setHtml(html)
        self.text_item.setTextWidth(max(20, self.rect().width() - 16))

    # -- refresco de recuadros tipo 'imagen'/'calculo'/'lista' -----------
    def hacer_refresh(self):
        if self.regenerar is None:
            return
        try:
            resultado = self.regenerar()
        except RuntimeError:
            QMessageBox.warning(
                self.ventana_principal, "No disponible",
                "El elemento original ya no existe (por ejemplo, borraste ese gráfico)."
            )
            return
        if self.kind == "calculo":
            titulo, valor_str, valor_crudo = resultado
            self.set_calculo_text(titulo, valor_str, valor_crudo)
        elif self.kind == "lista":
            titulo, cuerpo = resultado
            self.set_lista_text(titulo, cuerpo)
        else:
            self.set_pixmap(resultado)

    # -- interacción: mover / redimensionar / refrescar -----------------
    def mousePressEvent(self, event):
        pos = event.pos()
        if self.btn_refresh is not None and self.btn_refresh.isVisible():
            zona_boton = self.btn_refresh.mapToParent(self.btn_refresh.boundingRect()).boundingRect()
            if zona_boton.contains(pos):
                self.hacer_refresh()
                event.accept()
                return
        r = self.rect()
        en_esquina = (r.width() - self.HANDLE) <= pos.x() <= r.width() + 4 and \
                     (r.height() - self.HANDLE) <= pos.y() <= r.height() + 4
        if en_esquina:
            self._resizing = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            nuevo = event.pos()
            ancho = max(60, nuevo.x())
            alto = max(40, nuevo.y())
            self.prepareGeometryChange()
            self.setRect(0, 0, ancho, alto)
            self._reposicionar_contenido()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.kind == "texto" and self.text_item is not None:
            self.text_item.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
            self.text_item.setFocus(Qt.FocusReason.MouseFocusReason)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        accion_frente = menu.addAction("Traer al frente")
        accion_atras = menu.addAction("Enviar atrás")
        accion_forma = menu.addAction("Cambiar forma") if self.kind == "texto" else None
        menu.addSeparator()
        accion_eliminar = menu.addAction("Eliminar")
        elegido = menu.exec(event.screenPos())
        if elegido is None:
            return
        if elegido == accion_frente:
            self.setZValue(self.zValue() + 1)
        elif elegido == accion_atras:
            self.setZValue(self.zValue() - 1)
        elif accion_forma is not None and elegido == accion_forma:
            self.shape_kind = "ellipse" if self.shape_kind == "rect" else "rect"
            self.update()
        elif elegido == accion_eliminar:
            if self.scene() is not None:
                self.scene().removeItem(self)

    def paint(self, painter, option, widget=None):
        activo = self._cromo_visible()
        es_elipse = self.shape_kind == "ellipse"
        # El marco se ve: en reposo solo si la forma ES el contenido (óvalo) o si
        # se desactivó el modo limpio; al usar la caja, fino y discreto.
        if self.isSelected():
            pluma = QPen(QColor(COLOR_ACCENT_2), 1)
        elif self._hover:
            pluma = QPen(QColor(REPORTE_PAPEL_BORDE), 1)
        elif es_elipse or not REPORTE_CAJA_SOLO_AL_USAR:
            pluma = self.pen()
        else:
            pluma = QPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.brush())
        painter.setPen(pluma)
        if es_elipse:
            painter.drawEllipse(self.rect())
        else:
            painter.drawRect(self.rect())
        if not activo:
            return
        # Tirador de redimensionar, esquina inferior derecha.
        r = self.rect()
        painter.setPen(QPen(QColor(REPORTE_PAPEL_MUTED), 1))
        painter.drawLine(int(r.width() - 10), int(r.height() - 2), int(r.width() - 2), int(r.height() - 10))
        painter.drawLine(int(r.width() - 6), int(r.height() - 2), int(r.width() - 2), int(r.height() - 6))


# ----------------------------------------------------------------------------
# Ventana principal
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Memoria organizacional persistente: las anomalías detectadas dejan de ser
# "una foto" -- quedan guardadas en un SQLite local (un solo archivo, sin
# servidor) asociadas al "proceso de negocio" al que pertenecen, identificado
# por la ESTRUCTURA del dataset (nombres de columna normalizados + tipo
# genérico), no por el nombre del archivo. Así, la próxima vez que se cargue
# un dataset con la misma estructura, se puede saber si un problema ya
# existía antes y sigue sin resolverse.
# ----------------------------------------------------------------------------