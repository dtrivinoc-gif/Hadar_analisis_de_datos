"""
ontologia_ui.py — Ventana del "Esquema sugerido" (diagrama de tablas y
relaciones) para Hadar.

Qué es esto, en simple:
El usuario carga 2+ tablas (ej. Clientes y Pedidos). Esta ventana le
muestra una cajita por tabla, con sus columnas adentro, y líneas que
conectan las tablas que Hadar cree que están relacionadas (calculado
por ontologia.inferir_relaciones). El usuario puede:
- arrastrar las cajitas para acomodarlas a su gusto,
- borrar una línea que esté mal (clic derecho -> Eliminar),
- agregar una relación a mano si Hadar no la encontró sola.

Nada de esto obliga al usuario a saber de bases de datos: si no toca
nada, el esquema sugerido automático ya queda funcionando.

Este módulo no toca el DataFrame activo de la app (self.df en
HadarApp) ni ninguna otra pestaña — es autocontenido. La ventana
principal solo necesita: crear el diálogo pasándole las tablas y el
tema de colores, mostrarlo, y (más adelante, en el Paso C) leer
`dialogo.relaciones` para cruzar información en Narrativa.

--- Cambios de esta versión ---
El diagrama se veía "desprolijo" con muchas curvas cruzándose. Tres
cambios, inspirados en cómo Power BI dibuja sus relaciones:

1. Líneas RECTAS en vez de curvas (antes: cubicTo / Bézier).

2. UNA sola línea por CADA PAR DE TABLAS, en vez de una línea por
   cada par de columnas que calzó. Si Hadar encontró, por ejemplo,
   2 columnas en común entre "Inventario" y "Productos", antes se
   dibujaban 2 curvas separadas; ahora se dibuja 1 sola línea recta
   entre esas dos cajas, y el detalle de qué columnas coinciden queda
   en el tooltip (al pasar el mouse) y en los puntitos de color junto
   a cada columna involucrada (eso no cambió). Con esto, el número de
   líneas en el diagrama pasa de "una por columna que calzó" a "una
   por par de tablas relacionadas", que es exactamente como se ve un
   diagrama de Power BI.

3. Etiquetas de cardinalidad "1" / "*" en las puntas de cada línea
   (el "algo en medio" que se ve en los diagramas de Power BI): el
   extremo "1" señala la tabla principal (la lista maestra, ej.
   Productos) y el extremo "*" señala la tabla que la referencia
   muchas veces (ej. Inventario). Si Hadar no está seguro de cuál es
   cuál, no se dibuja ninguna etiqueta y la línea queda punteada,
   igual que antes, invitando a que el usuario lo confirme con clic
   derecho.

La clase _LineaRelacion ahora representa un GRUPO de una o más
relaciones (ontologia.RelacionSugerida) entre las mismas dos tablas,
en vez de una sola. El clic derecho para eliminar borra todo el
grupo (todas las columnas que esa línea resume) de una vez; sería
más fino poder borrar solo una columna del grupo, pero en la práctica
casi todos los grupos terminan siendo de 1 sola relación gracias al
ajuste de umbrales en ontologia.py, así que se dejó así por simpleza.

--- Cambios de esta segunda ronda ---
1. Líneas ORTOGONALES (solo tramos norte/sur/este/oeste, con un único
   quiebre en el medio) en vez de líneas rectas en diagonal. Cada caja
   ahora tiene 4 puntos de anclaje (arriba/abajo/izquierda/derecha) en
   vez de solo 2, y _LineaRelacion elige automáticamente si conectar
   por los lados (cuando dos tablas están más separadas en horizontal)
   o por arriba/abajo (cuando están más separadas en vertical).
2. La ventana ahora ajusta el zoom automáticamente para que todo el
   esquema entre en la pantalla al abrirse (antes, con esquemas
   grandes, quedaba cortado y había que usar la barra de scroll). Hay
   además un botón "Ajustar vista" para volver a centrar después de
   mover las cajas a mano.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem,
    QGraphicsTextItem, QGraphicsPathItem, QMenu, QMessageBox, QFrame,
    QDialogButtonBox, QToolTip,
)

from .config import COLOR_ACCENT, COLOR_ACCENT_2, COLOR_ACCENT_3, COLOR_DANGER
from .ontologia import inferir_relaciones, RelacionSugerida

# ----------------------------------------------------------------------
# Medidas del diagrama (en píxeles de la escena)
# ----------------------------------------------------------------------

ANCHO_CAJA = 220
ALTO_FILA_COLUMNA = 26
ALTO_TITULO = 40
ESPACIO_ENTRE_CAJAS_X = 90
ESPACIO_ENTRE_CAJAS_X_MAX_POR_FILA = 3   # cuántas cajas por fila antes de bajar
ESPACIO_ENTRE_CAJAS_Y = 60
MAX_COLUMNAS_VISIBLES = 14               # si una tabla tiene más columnas, se recorta la vista (con "+N más")
DISTANCIA_ETIQUETA_CARDINALIDAD = 16     # qué tan lejos de la caja se dibuja el "1" / "*"
MAX_LADO_EXPORTACION = 4096                # tope de ancho/alto del pixmap exportado (px)
MAX_PIXELES_EXPORTACION = 6_000_000      # tope de área (~24 MB a 4 bytes/px); se escala proporcional
FACTOR_ZOOM = 1.15                       # cuánto acerca/aleja cada clic en los botones - / +


# ----------------------------------------------------------------------
# Cajita de una tabla
# ----------------------------------------------------------------------

class _CajaTabla(QGraphicsRectItem):
    """Representa una tabla en el diagrama: título + lista de columnas.
    Se puede arrastrar; cuando se mueve, avisa a las líneas conectadas
    para que se redibujen."""

    def __init__(self, nombre_tabla: str, columnas: list[str], colors: dict):
        columnas_visibles = columnas[:MAX_COLUMNAS_VISIBLES]
        self._hay_columnas_ocultas = len(columnas) > MAX_COLUMNAS_VISIBLES
        n_filas = len(columnas_visibles) + (1 if self._hay_columnas_ocultas else 0)
        alto = ALTO_TITULO + n_filas * ALTO_FILA_COLUMNA + 10

        super().__init__(0, 0, ANCHO_CAJA, alto)
        self.nombre_tabla = nombre_tabla
        self.columnas = columnas
        self.colors = colors
        self.lineas_conectadas: list["_LineaRelacion"] = []
        # Callback que pone el diálogo dueño para agrupar actualizaciones de
        # líneas (una vez por ciclo de eventos). None -> se actualiza al tiro.
        self.programar_lineas = None
        self._filas_columna: dict[str, QRectF] = {}

        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setZValue(1)
        self.setPen(QPen(QColor(colors["border"]), 1.5))
        self.setBrush(QBrush(QColor(colors["card"])))
        self.setCursor(Qt.OpenHandCursor)

        # OJO: PySide6 no protege automáticamente de la basura (garbage
        # collector) de Python a los QGraphicsItem hijos solo por tener un
        # "padre gráfico" — hay que guardar una referencia dura acá, o los
        # textos/puntos desaparecen del diagrama sin avisar ni dar error.
        self._items_hijos: list = []

        # --- título ---
        titulo = QGraphicsTextItem(nombre_tabla, self)
        self._items_hijos.append(titulo)
        fuente_titulo = QFont()
        fuente_titulo.setBold(True)
        fuente_titulo.setPointSize(11)
        titulo.setFont(fuente_titulo)
        titulo.setDefaultTextColor(QColor(COLOR_ACCENT))
        titulo.setPos(10, 8)

        subtitulo = QGraphicsTextItem(f"{len(columnas)} columna(s)", self)
        self._items_hijos.append(subtitulo)
        fuente_sub = QFont()
        fuente_sub.setPointSize(8)
        subtitulo.setFont(fuente_sub)
        subtitulo.setDefaultTextColor(QColor(colors["muted"]))
        subtitulo.setPos(10, 24)

        # línea separadora bajo el título
        separador = QGraphicsRectItem(0, ALTO_TITULO, ANCHO_CAJA, 1, self)
        separador.setPen(QPen(QColor(colors["border"])))
        separador.setBrush(QBrush(QColor(colors["border"])))
        self._items_hijos.append(separador)

        # --- filas de columnas ---
        y = ALTO_TITULO + 5
        for col in columnas_visibles:
            texto = QGraphicsTextItem(str(col), self)
            self._items_hijos.append(texto)
            fuente_col = QFont()
            fuente_col.setPointSize(9)
            texto.setFont(fuente_col)
            texto.setDefaultTextColor(QColor(colors["text"]))
            texto.setPos(18, y)
            # recorta nombres muy largos para que no se salgan de la caja
            if texto.boundingRect().width() > ANCHO_CAJA - 30:
                texto.setPlainText(str(col)[:22] + "…")
            self._filas_columna[col] = QRectF(0, y, ANCHO_CAJA, ALTO_FILA_COLUMNA)
            y += ALTO_FILA_COLUMNA

        if self._hay_columnas_ocultas:
            n_ocultas = len(columnas) - len(columnas_visibles)
            extra = QGraphicsTextItem(f"… +{n_ocultas} columna(s) más", self)
            self._items_hijos.append(extra)
            fuente_extra = QFont()
            fuente_extra.setPointSize(8)
            fuente_extra.setItalic(True)
            extra.setFont(fuente_extra)
            extra.setDefaultTextColor(QColor(colors["muted"]))
            extra.setPos(18, y)

    # -- puntos de anclaje para las líneas --------------------------------

    def tiene_columna_visible(self, nombre_columna: str) -> bool:
        return nombre_columna in self._filas_columna

    def centro_escena(self) -> QPointF:
        """Centro de la caja en coordenadas de escena. Se usa para decidir
        si dos tablas están más separadas en horizontal o en vertical, y
        así elegir por qué lado sale la línea que las conecta."""
        return self.mapToScene(QPointF(self.rect().width() / 2, self.rect().height() / 2))

    def punto_borde(self, lado: str) -> QPointF:
        """
        Punto (en coordenadas de ESCENA) en la mitad de uno de los 4
        bordes de la caja. Ya no apunta a una columna específica -- así
        todas las relaciones entre las mismas dos tablas comparten un
        único punto de salida/llegada, como en Power BI, en vez de
        abrirse en abanico desde cada fila.

        lado: 'izquierda', 'derecha', 'arriba' o 'abajo'.
        """
        ancho, alto = self.rect().width(), self.rect().height()
        puntos_locales = {
            "izquierda": QPointF(0, alto / 2),
            "derecha": QPointF(ancho, alto / 2),
            "arriba": QPointF(ancho / 2, 0),
            "abajo": QPointF(ancho / 2, alto),
        }
        return self.mapToScene(puntos_locales[lado])

    def resaltar_columna(self, nombre_columna: str, color: QColor):
        """Dibuja un puntito de color junto a una columna involucrada en
        una relación (pista visual rápida de qué columnas calzaron, sin
        tener que abrir el tooltip de la línea)."""
        fila = self._filas_columna.get(nombre_columna)
        if fila is None:
            return
        punto = QGraphicsRectItem(6, fila.top() + fila.height() / 2 - 3, 6, 6, self)
        punto.setBrush(QBrush(color))
        punto.setPen(QPen(Qt.NoPen))
        self._items_hijos.append(punto)

    # -- mover la caja actualiza las líneas --------------------------------

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            if self.programar_lineas is not None:
                self.programar_lineas(self.lineas_conectadas)
            else:
                for linea in self.lineas_conectadas:
                    linea.actualizar_geometria()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)


def _punto_desplazado(origen: QPointF, hacia: QPointF, distancia: float) -> QPointF:
    """Punto ubicado a `distancia` píxeles de `origen`, en línea recta
    hacia `hacia`. Se usa para poner las etiquetas "1" / "*" un poco
    separadas de la caja, no pegadas al borde."""
    dx = hacia.x() - origen.x()
    dy = hacia.y() - origen.y()
    largo = math.hypot(dx, dy) or 1.0
    return QPointF(origen.x() + dx / largo * distancia, origen.y() + dy / largo * distancia)


# ----------------------------------------------------------------------
# Línea que conecta dos tablas (puede resumir más de una relación/columna)
# ----------------------------------------------------------------------

class _LineaRelacion(QGraphicsPathItem):
    """Línea recta entre dos tablas. Representa UNA O MÁS relaciones
    (ontologia.RelacionSugerida) entre las mismas dos tablas -- si Hadar
    encontró varias columnas en común entre "A" y "B", todas comparten
    esta misma línea en vez de dibujarse por separado.

    Se puede eliminar con clic derecho (borra TODAS las relaciones que
    resume). El grosor/color reflejan la confianza de la relación más
    fuerte del grupo."""

    def __init__(self, relaciones: list[RelacionSugerida], caja_origen: _CajaTabla,
                 caja_destino: _CajaTabla, colors: dict, on_eliminar, on_confirmar_direccion=None):
        super().__init__()
        self.relaciones = relaciones
        self.caja_origen = caja_origen
        self.caja_destino = caja_destino
        self.colors = colors
        self._on_eliminar = on_eliminar
        self._on_confirmar_direccion = on_confirmar_direccion
        self._tooltip_texto = ""

        self.setZValue(0)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)

        caja_origen.lineas_conectadas.append(self)
        caja_destino.lineas_conectadas.append(self)

        # Etiquetas de cardinalidad ("1" en la tabla principal, "*" en la
        # que la referencia). Mismo cuidado de referencia dura que en
        # _CajaTabla, para que PySide6 no las bote de la memoria.
        self._items_hijos: list = []
        self._etiqueta_uno = QGraphicsTextItem("1", self)
        self._etiqueta_muchos = QGraphicsTextItem("*", self)
        for etiqueta in (self._etiqueta_uno, self._etiqueta_muchos):
            fuente = QFont()
            fuente.setBold(True)
            fuente.setPointSize(9)
            etiqueta.setFont(fuente)
            etiqueta.setZValue(2)
            self._items_hijos.append(etiqueta)

        self._hover = False
        self.actualizar_geometria()
        self._tooltip_texto = self._armar_tooltip()
        self.setToolTip(self._tooltip_texto)

    # -- cuál de las relaciones del grupo manda sobre el color/estilo -----

    def _representativa(self) -> RelacionSugerida:
        """La relación más "fuerte" del grupo, usada para decidir el
        color, el grosor, el estilo de línea y la dirección (1/*) que se
        dibuja. Se prioriza una relación detectada automáticamente (con
        más confianza) por sobre una agregada a mano, porque la manual
        no trae información de cuál tabla es la principal."""
        automaticas = [r for r in self.relaciones if not getattr(r, "manual", False)]
        if automaticas:
            return max(automaticas, key=lambda r: r.confianza)
        return self.relaciones[0]

    def _color_por_confianza(self) -> QColor:
        r = self._representativa()
        if getattr(r, "manual", False):
            return QColor(COLOR_ACCENT_3)  # relación agregada a mano por el usuario
        if getattr(r, "direccion_pendiente", False):
            return QColor(COLOR_DANGER)    # dirección padre/hijo sin confirmar: salta a la vista
        # (una relación 1:1 no tiene dirección que confirmar: se pinta por confianza)
        c = r.confianza
        if c >= 0.75:
            return QColor(COLOR_ACCENT)
        if c >= 0.5:
            return QColor(COLOR_ACCENT_2)
        return QColor(self.colors["muted"])

    def actualizar_geometria(self):
        r = self._representativa()

        # Elige por qué par de lados sale la línea: si las dos tablas
        # están más separadas en horizontal, sale por izquierda/derecha;
        # si están más separadas en vertical (ej. una fila de cajas
        # arriba y otra abajo), sale por arriba/abajo. Así la línea nunca
        # tiene que cruzar en diagonal por encima de una caja para llegar
        # a un lado que no le queda de frente.
        centro_o = self.caja_origen.centro_escena()
        centro_d = self.caja_destino.centro_escena()
        dx = centro_d.x() - centro_o.x()
        dy = centro_d.y() - centro_o.y()

        if abs(dx) >= abs(dy):
            if dx >= 0:
                p1 = self.caja_origen.punto_borde("derecha")
                p2 = self.caja_destino.punto_borde("izquierda")
            else:
                p1 = self.caja_origen.punto_borde("izquierda")
                p2 = self.caja_destino.punto_borde("derecha")
            separacion = abs(p2.x() - p1.x())
            margen = min(18.0, separacion / 3.0)
            medio = (p1.x() + p2.x()) / 2
            # Evita un codo prácticamente pegado al borde en separaciones cortas.
            if separacion >= 2 * margen:
                medio = max(p1.x() + margen, min(medio, p2.x() - margen)) if p2.x() >= p1.x() else max(p2.x() + margen, min(medio, p1.x() - margen))
            quiebre1 = QPointF(medio, p1.y())
            quiebre2 = QPointF(medio, p2.y())
        else:
            if dy >= 0:
                p1 = self.caja_origen.punto_borde("abajo")
                p2 = self.caja_destino.punto_borde("arriba")
            else:
                p1 = self.caja_origen.punto_borde("arriba")
                p2 = self.caja_destino.punto_borde("abajo")
            separacion = abs(p2.y() - p1.y())
            margen = min(18.0, separacion / 3.0)
            medio = (p1.y() + p2.y()) / 2
            if separacion >= 2 * margen:
                medio = max(p1.y() + margen, min(medio, p2.y() - margen)) if p2.y() >= p1.y() else max(p2.y() + margen, min(medio, p1.y() - margen))
            quiebre1 = QPointF(p1.x(), medio)
            quiebre2 = QPointF(p2.x(), medio)

        # Camino ORTOGONAL: solo tramos norte/sur/este/oeste con un único
        # quiebre en el medio (antes era una línea recta en diagonal).
        camino = QPainterPath(p1)
        camino.lineTo(quiebre1)
        camino.lineTo(quiebre2)
        camino.lineTo(p2)
        self.setPath(camino)

        color = self._color_por_confianza()
        grosor = 1.0 + 2.5 * r.confianza + (1.5 if self._hover else 0.0)
        pen = QPen(color, grosor)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        if getattr(r, "manual", False):
            pen.setStyle(Qt.DashLine)
        elif getattr(r, "direccion_pendiente", False):
            pen.setStyle(Qt.DotLine)
        self.setPen(pen)

        # Etiquetas de cardinalidad: solo se dibujan si Hadar (o el
        # usuario) ya sabe cuál tabla es la principal. Si no, se ocultan
        # y la línea punteada ya avisa que falta confirmarlo. El
        # desplazamiento se hace hacia el primer quiebre (no hacia la
        # punta lejana), para que la etiqueta quede pegada al primer
        # tramo -- siempre horizontal o vertical, nunca en diagonal.
        texto_muchos = "*"
        if r.tabla_principal == self.caja_origen.nombre_tabla:
            punto_uno, vecino_uno = p1, quiebre1
            punto_muchos, vecino_muchos = p2, quiebre2
        elif r.tabla_principal == self.caja_destino.nombre_tabla:
            punto_uno, vecino_uno = p2, quiebre2
            punto_muchos, vecino_muchos = p1, quiebre1
        elif getattr(r, "cardinalidad", "") == "uno_a_uno" and not getattr(r, "manual", False):
            # 1:1: no hay padre/hijo; se rotula "1" en ambas puntas, sin alarma.
            punto_uno, vecino_uno = p1, quiebre1
            punto_muchos, vecino_muchos = p2, quiebre2
            texto_muchos = "1"
        else:
            punto_uno = punto_muchos = None

        if punto_uno is not None:
            pos_uno = _punto_desplazado(punto_uno, vecino_uno, DISTANCIA_ETIQUETA_CARDINALIDAD)
            pos_muchos = _punto_desplazado(punto_muchos, vecino_muchos, DISTANCIA_ETIQUETA_CARDINALIDAD)
            self._etiqueta_uno.setPlainText("1")
            self._etiqueta_uno.setDefaultTextColor(color)
            self._etiqueta_uno.setPos(pos_uno.x() - 5, pos_uno.y() - 10)
            self._etiqueta_muchos.setPlainText(texto_muchos)
            self._etiqueta_muchos.setDefaultTextColor(color)
            self._etiqueta_muchos.setPos(pos_muchos.x() - 5, pos_muchos.y() - 12)
            self._etiqueta_uno.setVisible(True)
            self._etiqueta_muchos.setVisible(True)
        else:
            self._etiqueta_uno.setVisible(False)
            self._etiqueta_muchos.setVisible(False)

    # -- tooltip: resume todas las columnas que esta línea agrupa ---------

    def _armar_tooltip(self) -> str:
        lineas = []
        for r in self.relaciones:
            if getattr(r, "manual", False):
                etiqueta = "Agregada manualmente"
            else:
                etiqueta = f"{int(r.confianza * 100)}% de confianza"
            aviso = ""
            cardinalidad = getattr(r, "cardinalidad", "sin_definir")
            if cardinalidad == "muchos_a_muchos":
                aviso = " ↔ relación muchos a muchos (M:N)"
            elif cardinalidad == "uno_a_uno" and not getattr(r, "manual", False):
                aviso = " ↔ relación uno a uno (1:1)"
            elif getattr(r, "direccion_pendiente", False):
                aviso = " ⚠ dirección sin confirmar"
            lineas.append(
                f"{r.tabla_origen}.{r.columna_origen} ↔ {r.tabla_destino}.{r.columna_destino} "
                f"({etiqueta}){aviso}"
            )
            lineas.append(f"   {r.razon}")
        lineas.append("(clic derecho para más opciones)")
        return "\n".join(lineas)

    def hoverEnterEvent(self, event):
        # El tooltip estático lo gestiona Qt. El engrosado es un ESTADO
        # (_hover) que actualizar_geometria() aplica siempre igual; antes se
        # sumaba grosor al pen actual y, si faltaba el hoverLeave, quedaba pegado.
        self._hover = True
        self.actualizar_geometria()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.actualizar_geometria()
        super().hoverLeaveEvent(event)

    def limpiar_hover(self):
        """Apaga el resaltado y redibuja. El diálogo lo usa en las actualizaciones
        diferidas: si una caja se está moviendo, el mouse está en la caja, no en la
        línea, aunque Qt no haya enviado hoverLeaveEvent."""
        self._hover = False
        self.actualizar_geometria()

    def contextMenuEvent(self, event):
        no_manuales = [r for r in self.relaciones if not getattr(r, "manual", False)]
        menu = QMenu()
        if len(self.relaciones) == 1:
            accion_eliminar = menu.addAction("Eliminar esta relación")
        else:
            accion_eliminar = menu.addAction(f"Eliminar esta relación ({len(self.relaciones)} columnas)")

        accion_principal_origen = None
        accion_principal_destino = None
        r = self._representativa()
        es_mn = getattr(r, "cardinalidad", "sin_definir") == "muchos_a_muchos"
        es_1a1 = getattr(r, "cardinalidad", "sin_definir") == "uno_a_uno"
        if no_manuales and self._on_confirmar_direccion is not None:
            menu.addSeparator()
            if es_mn:
                menu.addAction("Ambas columnas repiten valores (M:N). Si sabes cuál es la principal, elígela:").setEnabled(False)
            elif es_1a1:
                menu.addAction("Relación 1:1 (no requiere dirección). Si una tabla es en realidad la principal, elígela:").setEnabled(False)
            marca_o = " ✓" if r.tabla_principal == self.caja_origen.nombre_tabla else ""
            marca_d = " ✓" if r.tabla_principal == self.caja_destino.nombre_tabla else ""
            accion_principal_origen = menu.addAction(f'"{self.caja_origen.nombre_tabla}" es la tabla principal{marca_o}')
            accion_principal_destino = menu.addAction(f'"{self.caja_destino.nombre_tabla}" es la tabla principal{marca_d}')

        elegida = menu.exec(event.screenPos())
        if elegida == accion_eliminar:
            self._on_eliminar(list(self.relaciones))
        elif elegida is not None and elegida == accion_principal_origen:
            self._on_confirmar_direccion(self.caja_origen.nombre_tabla, no_manuales)
        elif elegida is not None and elegida == accion_principal_destino:
            self._on_confirmar_direccion(self.caja_destino.nombre_tabla, no_manuales)


# ----------------------------------------------------------------------
# Diálogo principal
# ----------------------------------------------------------------------

class DialogoEsquemaOntologia(QDialog):
    """
    Ventana del esquema sugerido. Uso típico desde HadarApp:

        dialogo = DialogoEsquemaOntologia(self.tablas, self.colors, self)
        dialogo.exec()
        # más adelante (Paso C): dialogo.relaciones tiene la lista final
        # (con lo que el usuario haya agregado/borrado a mano).
    """

    def __init__(self, tablas: dict, colors: dict, parent=None, relaciones_iniciales=None):
        super().__init__(parent)
        self.tablas = tablas
        self.colors = colors
        self.setWindowTitle("Esquema sugerido — Hadar")
        self.resize(1000, 640)
        self.setModal(True)

        # Si ya había un esquema confirmado antes (relaciones_iniciales), se
        # parte de ahí -- así reabrir este diálogo no borra las relaciones
        # que el usuario agregó, eliminó o corrigió la última vez. Solo se
        # vuelve a detectar todo desde cero la primerísima vez que se abre.
        if relaciones_iniciales is not None:
            self.relaciones: list[RelacionSugerida] = list(relaciones_iniciales)
        else:
            self.relaciones: list[RelacionSugerida] = inferir_relaciones(tablas)

        self._cajas: dict[str, _CajaTabla] = {}
        self._lineas: list[_LineaRelacion] = []

        # Actualización agrupada de líneas: propia de ESTE diálogo (antes era
        # global de módulo y dos diálogos se pisaban).
        self._lineas_pendientes: set = set()
        self._timer_lineas = QTimer(self)
        self._timer_lineas.setSingleShot(True)
        self._timer_lineas.setInterval(0)
        self._timer_lineas.timeout.connect(self._actualizar_lineas_pendientes)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # --- encabezado ---
        titulo = QLabel("Esquema sugerido entre tus tablas")
        titulo.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {colors['text']};")
        layout.addWidget(titulo)

        ayuda = QLabel(
            "Hadar detectó estas conexiones solo. Cada línea une dos tablas (puede resumir "
            "más de una columna en común); el extremo marcado \"1\" es la tabla principal y "
            "el extremo \"*\" es la que la referencia muchas veces. Arrastra las tablas para "
            "acomodarlas. Clic derecho sobre una línea para borrarla o corregir cuál tabla es "
            "la principal. Si falta una relación, agrégala con el botón de abajo."
        )
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet(f"color: {colors['muted']}; font-size: 11px;")
        layout.addWidget(ayuda)

        # --- barra de herramientas ---
        barra = QHBoxLayout()
        self.btn_agregar = QPushButton("+ Agregar relación manual")
        self.btn_agregar.clicked.connect(self._abrir_dialogo_agregar_relacion)
        barra.addWidget(self.btn_agregar)

        self.btn_redetectar = QPushButton("Detectar automáticamente de nuevo")
        self.btn_redetectar.setToolTip(
            "Vuelve a analizar las tablas. No borra las relaciones que agregaste a mano."
        )
        self.btn_redetectar.clicked.connect(self._redetectar)
        barra.addWidget(self.btn_redetectar)

        self.btn_zoom_menos = QPushButton("-")
        self.btn_zoom_menos.setFixedWidth(30)
        self.btn_zoom_menos.setToolTip("Alejar")
        self.btn_zoom_menos.clicked.connect(lambda: self._zoom(1 / FACTOR_ZOOM))
        barra.addWidget(self.btn_zoom_menos)

        self.btn_zoom_mas = QPushButton("+")
        self.btn_zoom_mas.setFixedWidth(30)
        self.btn_zoom_mas.setToolTip("Acercar")
        self.btn_zoom_mas.clicked.connect(lambda: self._zoom(FACTOR_ZOOM))
        barra.addWidget(self.btn_zoom_mas)

        barra.addStretch()
        self.lbl_resumen = QLabel("")
        self.lbl_resumen.setStyleSheet(f"color: {colors['muted']}; font-size: 11px;")
        barra.addWidget(self.lbl_resumen)
        layout.addLayout(barra)

        # --- lienzo ---
        self.escena = QGraphicsScene(self)
        self.escena.setBackgroundBrush(QBrush(QColor(colors["bg"])))
        self.vista = QGraphicsView(self.escena)
        self.vista.setRenderHint(QPainter.Antialiasing)
        self.vista.setDragMode(QGraphicsView.RubberBandDrag)
        self.vista.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        layout.addWidget(self.vista, stretch=1)

        # --- leyenda ---
        leyenda = QLabel(
            f"<span style='color:{COLOR_ACCENT}'>●</span> confianza alta&nbsp;&nbsp;"
            f"<span style='color:{COLOR_ACCENT_2}'>●</span> confianza media&nbsp;&nbsp;"
            f"<span style='color:{colors['muted']}'>●</span> confianza baja&nbsp;&nbsp;"
            f"<span style='color:{COLOR_ACCENT_3}'>- - -</span> agregada por ti&nbsp;&nbsp;"
            f"<span style='color:{COLOR_DANGER}'>···</span> dirección sin confirmar (clic derecho)&nbsp;&nbsp;"
            f"<b>1 / *</b> tabla principal / tabla referenciada&nbsp;&nbsp;"
            f"<b>1 — 1</b> relación uno a uno (sin principal)"
        )
        leyenda.setWordWrap(True)
        leyenda.setStyleSheet("font-size: 11px;")
        layout.addWidget(leyenda)

        # --- botones de cierre ---
        botones = QDialogButtonBox(QDialogButtonBox.Close)
        botones.rejected.connect(self.reject)
        botones.accepted.connect(self.accept)
        botones.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        layout.addWidget(botones)

        self._dibujar_todo()

    # ------------------------------------------------------------------

    def _programar_actualizacion_lineas(self, lineas):
        self._lineas_pendientes.update(lineas)
        if not self._timer_lineas.isActive():
            self._timer_lineas.start()

    def _actualizar_lineas_pendientes(self):
        pendientes = list(self._lineas_pendientes)
        self._lineas_pendientes.clear()
        for linea in pendientes:
            try:
                linea.limpiar_hover()
            except RuntimeError:
                continue  # Qt ya destruyó el objeto gráfico

    def _dibujar_todo(self):
        self._timer_lineas.stop()
        self._lineas_pendientes.clear()
        self.escena.clear()
        self._cajas.clear()
        self._lineas.clear()
        self._colocar_cajas()
        self._dibujar_relaciones()
        self._actualizar_resumen()
        self.vista.setSceneRect(self.escena.itemsBoundingRect().adjusted(-40, -40, 40, 40))

    def _ajustar_vista(self):
        """Acerca/aleja el zoom para que todo el esquema quepa en la
        ventana de una vez, en vez de quedar cortado con barras de
        scroll (como pasaba con esquemas de varias tablas). Se llama
        solo automáticamente (al abrir la ventana y al redetectar); el
        usuario ajusta el zoom a mano con los botones - / +."""
        rect = self.escena.itemsBoundingRect()
        if rect.isEmpty():
            return
        self.vista.fitInView(rect.adjusted(-30, -30, 30, 30), Qt.KeepAspectRatio)

    def _zoom(self, factor: float):
        self.vista.scale(factor, factor)

    def showEvent(self, event):
        super().showEvent(event)
        # fitInView necesita que la vista ya tenga un tamaño real en
        # pantalla -- llamarlo apenas se construye el diálogo (antes de
        # mostrarse) daría un zoom incorrecto. Por eso se hace acá y no
        # en _dibujar_todo.
        self._ajustar_vista()

    def _colocar_cajas(self):
        x, y = 0, 0
        max_alto_fila = 0
        for i, (nombre_tabla, df) in enumerate(self.tablas.items()):
            caja = _CajaTabla(nombre_tabla, list(df.columns), self.colors)
            caja.setPos(x, y)
            caja.programar_lineas = self._programar_actualizacion_lineas
            self.escena.addItem(caja)
            self._cajas[nombre_tabla] = caja

            max_alto_fila = max(max_alto_fila, caja.rect().height())
            x += ANCHO_CAJA + ESPACIO_ENTRE_CAJAS_X
            if (i + 1) % ESPACIO_ENTRE_CAJAS_X_MAX_POR_FILA == 0:
                x = 0
                y += max_alto_fila + ESPACIO_ENTRE_CAJAS_Y
                max_alto_fila = 0

    def _dibujar_relaciones(self):
        # Agrupa todas las relaciones que conectan el MISMO par de tablas
        # en un solo grupo, para dibujar una sola línea por par (en vez de
        # una por cada columna que calzó). El orden de tabla_origen /
        # tabla_destino de cada relación dentro del grupo no importa acá:
        # _LineaRelacion compara por NOMBRE de tabla, no por rol.
        grupos: dict[frozenset, list[RelacionSugerida]] = {}
        for r in self.relaciones:
            clave = frozenset((r.tabla_origen, r.tabla_destino))
            grupos.setdefault(clave, []).append(r)

        for relaciones_grupo in grupos.values():
            primera = relaciones_grupo[0]
            caja_o = self._cajas.get(primera.tabla_origen)
            caja_d = self._cajas.get(primera.tabla_destino)
            if not caja_o or not caja_d:
                continue

            linea = _LineaRelacion(
                relaciones_grupo, caja_o, caja_d, self.colors,
                self._eliminar_relaciones, self._confirmar_direccion_grupo,
            )
            self.escena.addItem(linea)
            self._lineas.append(linea)

            color = linea._color_por_confianza()
            for r in relaciones_grupo:
                caja_r_origen = self._cajas.get(r.tabla_origen)
                caja_r_destino = self._cajas.get(r.tabla_destino)
                if caja_r_origen:
                    caja_r_origen.resaltar_columna(r.columna_origen, color)
                if caja_r_destino:
                    caja_r_destino.resaltar_columna(r.columna_destino, color)

    def _actualizar_resumen(self):
        n = len(self.relaciones)
        n_lineas = len(self._lineas)
        manuales = sum(1 for r in self.relaciones if getattr(r, "manual", False))
        if n == 0:
            self.lbl_resumen.setText("No se detectaron relaciones todavía.")
        else:
            extra = f" ({manuales} agregada(s) por ti)" if manuales else ""
            detalle_columnas = f" en {n} columna(s)" if n != n_lineas else ""
            self.lbl_resumen.setText(f"{n_lineas} relación(es) entre tablas{detalle_columnas}{extra}")

    # ------------------------------------------------------------------
    # Acciones del usuario
    # ------------------------------------------------------------------

    def _eliminar_relaciones(self, relaciones: list[RelacionSugerida]):
        if len(relaciones) == 1:
            r = relaciones[0]
            pregunta = (
                f"¿Eliminar la relación entre \"{r.tabla_origen}.{r.columna_origen}\" "
                f"y \"{r.tabla_destino}.{r.columna_destino}\"?"
            )
        else:
            detalle = "\n".join(
                f"- {r.tabla_origen}.{r.columna_origen} ↔ {r.tabla_destino}.{r.columna_destino}"
                for r in relaciones
            )
            pregunta = f"¿Eliminar estas {len(relaciones)} relaciones?\n{detalle}"

        respuesta = QMessageBox.question(self, "Eliminar relación", pregunta)
        if respuesta != QMessageBox.Yes:
            return
        for r in relaciones:
            if r in self.relaciones:
                self.relaciones.remove(r)
        self._dibujar_todo()

    def _confirmar_direccion_grupo(self, tabla_elegida: str, relaciones_afectadas: list[RelacionSugerida]):
        """El usuario eligió a mano, desde el clic derecho, cuál tabla es
        la principal para esta línea (para corregir un 'sin_definir', o
        simplemente porque el motor adivinó mal). Se aplica a todas las
        relaciones automáticas que comparten esa línea, para que la
        cardinalidad mostrada sea consistente."""
        for r in relaciones_afectadas:
            if r.tabla_origen == tabla_elegida:
                r.tabla_principal = tabla_elegida
                r.columna_principal = r.columna_origen
            elif r.tabla_destino == tabla_elegida:
                r.tabla_principal = tabla_elegida
                r.columna_principal = r.columna_destino
            else:
                continue
            r.certeza_direccion = "alta"
            r.direccion_confirmada = True
            if getattr(r, "cardinalidad", "") in ("muchos_a_muchos", "uno_a_uno"):
                r.cardinalidad = "uno_a_muchos"   # el usuario decidió quién es el padre
        self._dibujar_todo()

    def _redetectar(self):
        """Vuelve a correr el motor automático, conservando las relaciones
        que el usuario agregó a mano (y sin resucitar las que borró, salvo
        que efectivamente sigan siendo válidas y el usuario no las haya
        tocado — se identifican por tabla+columna en ambos extremos), y
        respetando cualquier dirección padre/hijo que el usuario haya
        confirmado a mano en una relación detectada automáticamente."""
        manuales = [r for r in self.relaciones if getattr(r, "manual", False)]
        # Este diccionario se reconstruye desde las relaciones vigentes en cada
        # ejecución; no se guarda como estado persistente, por lo que no acumula
        # direcciones huérfanas entre redetecciones.
        direcciones_confirmadas = {
            (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino): r
            for r in self.relaciones
            if getattr(r, "direccion_confirmada", False) and not getattr(r, "manual", False)
        }

        nuevas_auto = inferir_relaciones(self.tablas)

        claves_manuales = {
            (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino) for r in manuales
        }
        nuevas_auto = [
            r for r in nuevas_auto
            if (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino) not in claves_manuales
        ]

        for r in nuevas_auto:
            clave = (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino)
            vieja = direcciones_confirmadas.get(clave)
            if vieja is not None:
                r.tabla_principal = vieja.tabla_principal
                r.columna_principal = vieja.columna_principal
                r.certeza_direccion = "alta"
                r.direccion_confirmada = True
                if getattr(r, "cardinalidad", "") in ("muchos_a_muchos", "uno_a_uno"):
                    r.cardinalidad = "uno_a_muchos"   # el usuario decidió quién es el padre

        self.relaciones = nuevas_auto + manuales
        self._dibujar_todo()
        # _colocar_cajas() vuelve a poner las tablas en su cuadrícula por
        # defecto, así que el zoom que el usuario tenía ya no calza --
        # se reajusta para que todo vuelva a entrar en pantalla.
        self._ajustar_vista()

    def _abrir_dialogo_agregar_relacion(self):
        dialogo = _DialogoAgregarRelacion(self.tablas, self)
        if dialogo.exec() == QDialog.DialogCode.Accepted:
            t1, c1, t2, c2 = dialogo.seleccion()
            if t1 == t2:
                QMessageBox.warning(self, "Relación inválida", "Elige dos tablas distintas.")
                return
            ya_existe = any(
                (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino) == (t1, c1, t2, c2)
                or (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino) == (t2, c2, t1, c1)
                for r in self.relaciones
            )
            if ya_existe:
                QMessageBox.information(self, "Ya existe", "Esa relación ya está en el esquema.")
                return

            nueva = RelacionSugerida(
                tabla_origen=t1, columna_origen=c1,
                tabla_destino=t2, columna_destino=c2,
                confianza=1.0, razon="Agregada manualmente por el usuario",
            )
            nueva.manual = True
            self.relaciones.append(nueva)
            self._dibujar_todo()

    # ------------------------------------------------------------------

    def obtener_relaciones_confirmadas(self) -> list[RelacionSugerida]:
        """Para usar más adelante (Paso C): el estado final del esquema,
        tal como quedó después de que el usuario lo revisó."""
        return list(self.relaciones)

    def render_a_pixmap(self) -> QPixmap:
        """Imagen (QPixmap) del esquema tal como está dibujado ahora mismo
        -- para llevarlo a la pestaña Reporte sin necesidad de abrir esta
        ventana ni pedirle nada al usuario."""
        # Si una caja se movió en este mismo ciclo, las líneas aún no se redibujaron.
        self._actualizar_lineas_pendientes()
        rect = self.escena.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        ancho_f = max(1.0, rect.width())
        alto_f = max(1.0, rect.height())
        # Esquemas de 20-30 tablas pueden pasar de 40 MB: se escala proporcionalmente
        # (nunca se agranda) para respetar tope de lado y de área.
        escala = min(
            1.0,
            MAX_LADO_EXPORTACION / max(ancho_f, alto_f),
            math.sqrt(MAX_PIXELES_EXPORTACION / (ancho_f * alto_f)),
        )
        ancho = max(1, int(ancho_f * escala))
        alto = max(1, int(alto_f * escala))
        pixmap = QPixmap(ancho, alto)
        pixmap.fill(QColor(self.colors["bg"]))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        self.escena.render(painter, target=pixmap.rect(), source=rect)
        painter.end()
        return pixmap


class _DialogoAgregarRelacion(QDialog):
    """Formulario chico: elegir tabla+columna de un lado y del otro."""

    def __init__(self, tablas: dict, parent=None):
        super().__init__(parent)
        self.tablas = tablas
        self.setWindowTitle("Agregar relación manual")

        layout = QVBoxLayout(self)

        fila1 = QHBoxLayout()
        self.combo_tabla1 = QComboBox()
        self.combo_tabla1.addItems(list(tablas.keys()))
        self.combo_col1 = QComboBox()
        fila1.addWidget(QLabel("Tabla:"))
        fila1.addWidget(self.combo_tabla1)
        fila1.addWidget(QLabel("Columna:"))
        fila1.addWidget(self.combo_col1)
        layout.addLayout(fila1)

        layout.addWidget(QLabel("se relaciona con"))

        fila2 = QHBoxLayout()
        self.combo_tabla2 = QComboBox()
        self.combo_tabla2.addItems(list(tablas.keys()))
        if len(tablas) > 1:
            self.combo_tabla2.setCurrentIndex(1)
        self.combo_col2 = QComboBox()
        fila2.addWidget(QLabel("Tabla:"))
        fila2.addWidget(self.combo_tabla2)
        fila2.addWidget(QLabel("Columna:"))
        fila2.addWidget(self.combo_col2)
        layout.addLayout(fila2)

        self.combo_tabla1.currentTextChanged.connect(self._refrescar_col1)
        self.combo_tabla2.currentTextChanged.connect(self._refrescar_col2)
        self._refrescar_col1(self.combo_tabla1.currentText())
        self._refrescar_col2(self.combo_tabla2.currentText())

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _refrescar_col1(self, nombre_tabla):
        self.combo_col1.clear()
        if nombre_tabla in self.tablas:
            self.combo_col1.addItems([str(c) for c in self.tablas[nombre_tabla].columns])

    def _refrescar_col2(self, nombre_tabla):
        self.combo_col2.clear()
        if nombre_tabla in self.tablas:
            self.combo_col2.addItems([str(c) for c in self.tablas[nombre_tabla].columns])

    def seleccion(self):
        return (
            self.combo_tabla1.currentText(), self.combo_col1.currentText(),
            self.combo_tabla2.currentText(), self.combo_col2.currentText(),
        )