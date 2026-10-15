"""
Modelo de tabla (QAbstractTableModel) que conecta un DataFrame de pandas
con la vista QTableView de Qt, incluyendo el sistema de notas por celda
(usado por el detector de anomalías para resaltar valores problemáticos).
"""
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor, QPixmap, QPainter, QBrush

from .config import COLOR_DANGER, COLOR_ACCENT, COLOR_ACCENT_3
from .io_datos import cast_valor_a_dtype

# Rojo oscuro para celdas que SOLO marcó el chequeo de combinación (ML) --
# ver agregar_nota(exclusiva_ml=...). Si la misma celda también la marcó
# cualquier otro chequeo, se pinta con el rojo de siempre (COLOR_DANGER):
# el color distinto es para decir "esto SOLO lo vio el ML", no para decir
# "esto lo vio el ML también".
COLOR_ML_EXCLUSIVO = "#8B0000"


class PandasTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame = None):
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()
        # callback opcional: on_edit(row_label, col_name, nuevo_valor, valor_anterior)
        # que HadarApp usa para propagar la edición al DataFrame maestro.
        self.on_edit = None
        self.editable = False

        # ===== Notas por celda =====
        # Claves por (row_label, col_name) -- NO por posición (row, col) --
        # así una nota sigue apuntando a la celda correcta aunque cambien los
        # filtros y esa fila quede en otra posición visual de la tabla.
        self.notas = {}                  # {(row_label, col_name): "texto"}
        self.anomalias = {}              # {clave: exclusiva_ml} -- ver agregar_nota
        self._notas_narrativa_keys = set()  # cuáles vinieron de la pestaña Narrativa

        # ===== Limpieza sugerida =====
        # A propósito es un sistema APARTE de notas/anomalias: una anomalía
        # (ej. un precio mucho más alto que el resto) puede ser un problema
        # real de negocio que hay que investigar, no basura para borrar. Lo
        # que va acá es suciedad puramente técnica (duplicados, espacios de
        # más, formatos inconsistentes) -- la corrección es casi siempre
        # obvia, pero igual nunca se aplica sola, solo se señala.
        self.limpieza = {}               # {(row_label, col_name): "texto"}

        # ===== Columnas calculadas en Hadar =====
        # {nombre_columna: "fórmula tal como la escribió el usuario"}. Solo
        # cambia cómo se ve el ENCABEZADO (marca "ƒx" + tooltip); el nombre
        # real de la columna en el DataFrame no se toca.
        self._columnas_calculadas = {}
        # {nombre_columna: aviso} de las calculadas que pudieron quedar
        # desactualizadas (se modificaron datos de los que salen).
        self._columnas_desactualizadas = {}
        # {nombre_columna: texto} de las columnas que son PREDICCIONES de la
        # pestaña Predicción (ver tablero.py). No son datos del usuario: se
        # muestran de otro color, con aviso en el encabezado, y no se pueden
        # editar (no existen en el DataFrame maestro).
        self._columnas_predichas = {}
        # Iconos: se crean una sola vez (no en cada data(), que se llama por
        # cada celda visible en cada repintado).
        self._icono_anomalia = self._crear_icono_punto(COLOR_DANGER)
        self._icono_anomalia_ml = self._crear_icono_punto(COLOR_ML_EXCLUSIVO)
        self._icono_nota = self._crear_icono_punto(COLOR_ACCENT_3)

    @staticmethod
    def _crear_icono_punto(color_hex):
        pixmap = QPixmap(14, 14)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(color_hex)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, 10, 10)
        painter.end()
        return pixmap

    def icono_anomalia(self):
        """Getter público del punto rojo que ya se usa en la pestaña Datos
        para anomalías -- lo reutiliza, por ejemplo, la sub-pestaña Linaje
        de Narrativa, para que el mismo color signifique lo mismo en toda
        la app."""
        return self._icono_anomalia

    def _clave(self, row, col):
        try:
            return (self._df.index[row], self._df.columns[col])
        except IndexError:
            return None

    def agregar_nota(self, row_label, col_name, texto, es_anomalia=False,
                      origen_narrativa=False, exclusiva_ml=False, emitir=True):
        """Agrega o actualiza la nota de una celda, identificada por su
        etiqueta de índice real y nombre de columna (no por posición).
        exclusiva_ml=True pinta el círculo en rojo oscuro (COLOR_ML_EXCLUSIVO)
        en vez del rojo de siempre -- pensado para cuando ESA celda solo la
        marcó el chequeo de combinación (ML) y ningún otro chequeo la habría
        marcado por su cuenta. Si la misma celda recibe más de un llamado
        (varias anomalías tocándola), el último valor de exclusiva_ml es el
        que queda -- anomalias_a_notas_celda() ya se encarga de que todos
        los llamados para una misma celda traigan el mismo valor correcto."""
        clave = (row_label, col_name)
        self.notas[clave] = texto
        if es_anomalia:
            self.anomalias[clave] = exclusiva_ml
        elif clave in self.anomalias:
            del self.anomalias[clave]
        if origen_narrativa:
            self._notas_narrativa_keys.add(clave)
        if emitir:
            self._emitir_cambio_celda(row_label, col_name)

    def quitar_nota(self, row_label, col_name):
        clave = (row_label, col_name)
        self.notas.pop(clave, None)
        self.anomalias.pop(clave, None)
        self._notas_narrativa_keys.discard(clave)
        self._emitir_cambio_celda(row_label, col_name)

    def limpiar_notas_narrativa(self):
        """Quita solo las notas que puso automáticamente la pestaña
        Narrativa (para no acumular versiones viejas al regenerarla),
        dejando intactas las notas que el usuario escribió a mano."""
        if not self._notas_narrativa_keys:
            return
        for clave in list(self._notas_narrativa_keys):
            self.notas.pop(clave, None)
            self.anomalias.pop(clave, None)
        self._notas_narrativa_keys.clear()
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
            )

    def limpiar_todas_las_notas(self):
        """Se llama al cargar un archivo NUEVO (no al filtrar): las notas de
        un dataset anterior no deberían seguir apareciendo en otro distinto."""
        self.notas.clear()
        self.anomalias.clear()
        self._notas_narrativa_keys.clear()
        self.limpieza.clear()

    def marcar_limpieza(self, row_label, col_name, texto, emitir=True):
        """Marca una celda como parte de un hallazgo de limpieza sugerida
        (duplicado, espacio de más, formato inconsistente). Independiente
        de notas/anomalías -- una celda puede tener ambas cosas a la vez."""
        self.limpieza[(row_label, col_name)] = texto
        if emitir:
            self._emitir_cambio_celda(row_label, col_name)

    def limpiar_marcas_de_limpieza(self):
        """Quita todas las marcas de limpieza sugerida (para volver a
        calcular desde cero, o si el usuario cierra el panel)."""
        if not self.limpieza:
            return
        self.limpieza.clear()
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
            )

    def notas_manuales(self):
        """Solo las notas que escribió el usuario a mano (excluye las
        automáticas de Narrativa) -- usado al guardar un proyecto, ya que
        las de Narrativa se regeneran solas al volver a generar el informe."""
        return {
            clave: texto for clave, texto in self.notas.items()
            if clave not in self._notas_narrativa_keys
        }

    def cargar_notas_manuales(self, notas):
        """Repone notas manuales guardadas en un proyecto. Reemplaza las
        notas manuales actuales (deja intactas las automáticas de
        Narrativa, si las hubiera)."""
        for clave in list(self.notas.keys()):
            if clave not in self._notas_narrativa_keys:
                del self.notas[clave]
                self.anomalias.pop(clave, None)
        for (row_label, col_name), texto in (notas or {}).items():
            self.notas[(row_label, col_name)] = texto
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
            )

    def _emitir_cambio_celda(self, row_label, col_name):
        try:
            row = self._df.index.get_loc(row_label)
            col = self._df.columns.get_loc(col_name)
        except (KeyError, TypeError):
            return  # la celda no está visible con el filtro/orden actual
        if isinstance(row, slice) or not isinstance(row, (int, np.integer)):
            return  # índice repetido/no único: no hay una única celda que marcar
        if isinstance(col, slice) or not isinstance(col, (int, np.integer)):
            return
        idx = self.index(int(row), int(col))
        self.dataChanged.emit(idx, idx, [Qt.DecorationRole, Qt.BackgroundRole, Qt.ToolTipRole])

    def set_columnas_calculadas(self, mapa, avisos=None):
        """mapa: {nombre_columna: fórmula}. avisos: {nombre_columna: texto} solo
        para las que pudieron quedar desactualizadas. Refresca solo los encabezados."""
        self._columnas_calculadas = dict(mapa or {})
        self._columnas_desactualizadas = dict(avisos or {})
        n = self.columnCount()
        if n:
            self.headerDataChanged.emit(Qt.Horizontal, 0, n - 1)

    def set_columnas_predichas(self, mapa):
        """mapa: {nombre_columna: texto del aviso}. Vacío o None = ninguna."""
        mapa = dict(mapa or {})
        if mapa == self._columnas_predichas:
            return
        self._columnas_predichas = mapa
        n = self.columnCount()
        if n:
            self.headerDataChanged.emit(Qt.Horizontal, 0, n - 1)
            self.dataChanged.emit(self.index(0, 0), self.index(max(self.rowCount() - 1, 0), n - 1),
                                  [Qt.ForegroundRole])

    def _es_columna_predicha(self, col):
        if not self._columnas_predichas:
            return False
        try:
            return str(self._df.columns[col]) in self._columnas_predichas
        except IndexError:
            return False

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()

        if role == Qt.DecorationRole:
            clave = self._clave(row, col)
            if clave in self.anomalias:
                return self._icono_anomalia_ml if self.anomalias[clave] else self._icono_anomalia
            if clave in self.notas:
                return self._icono_nota
            return None

        if role == Qt.BackgroundRole:
            clave = self._clave(row, col)
            if clave in self.anomalias:
                return QColor(160, 160, 160)
            if clave in self.limpieza:
                color = QColor(COLOR_DANGER)
                color.setAlpha(55)  # rojo suave -- distinto del gris sólido de anomalías
                return color
            return None

        if role == Qt.ForegroundRole:
            return QColor(COLOR_ACCENT) if self._es_columna_predicha(col) else None

        if role == Qt.ToolTipRole:
            clave = self._clave(row, col)
            valor = self._df.iat[row, col]
            texto_valor = "" if pd.isna(valor) else str(valor)
            nota = self.notas.get(clave)
            limpieza_texto = self.limpieza.get(clave)
            # Junta nota (si hay) + aviso de limpieza (si hay) + el contenido
            # COMPLETO de la celda -- sin esto, un texto largo que la columna
            # corta visualmente queda ilegible sin agrandar la columna a mano.
            partes = []
            if nota:
                partes.append(f"📌 {nota}")
            if limpieza_texto:
                partes.append(f"Limpieza sugerida: {limpieza_texto}")
            if texto_valor:
                partes.append(texto_valor)
            return "\n\n".join(partes) if partes else None

        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        value = self._df.iat[row, col]
        if pd.isna(value):
            return ""
        return str(value)

    def flags(self, index):
        base = super().flags(index)
        if not index.isValid():
            return base
        if self.editable and not self._es_columna_predicha(index.column()):
            return base | Qt.ItemIsEditable
        return base

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid() or not self.editable:
            return False
        row, col = index.row(), index.column()
        if self._es_columna_predicha(col):
            return False
        col_name = self._df.columns[col]
        dtype = self._df[col_name].dtype
        nuevo_valor = cast_valor_a_dtype(str(value), dtype)
        row_label = self._df.index[row]

        valor_anterior = self._df.iat[row, col]
        self._df.iat[row, col] = nuevo_valor
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])

        if self.on_edit is not None:
            self.on_edit(row_label, col_name, nuevo_valor, valor_anterior)
        return True

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            if section < 0 or section >= len(self._df.columns):
                return None
            nombre = str(self._df.columns[section])
            # EditRole devuelve SIEMPRE el nombre real (sin la marca "ƒx"):
            # es lo que debe usar cualquier código que necesite el nombre de
            # la columna (ej. la barra de estado al hacer clic en una celda).
            if role == Qt.EditRole:
                return nombre
            if nombre in self._columnas_predichas:
                if role == Qt.ToolTipRole:
                    return self._columnas_predichas[nombre]
                if role == Qt.DisplayRole:
                    return nombre
            formula = self._columnas_calculadas.get(nombre)
            aviso = self._columnas_desactualizadas.get(nombre)
            if role == Qt.DisplayRole:
                if formula is None:
                    return nombre
                return f"ƒx ⚠ {nombre}" if aviso else f"ƒx {nombre}"
            if role == Qt.ToolTipRole and formula is not None:
                texto = (
                    "Columna calculada en Hadar\n"
                    f"= {formula}\n"
                    "Se calculó una sola vez: no se actualiza sola si cambias los datos."
                )
                if aviso:
                    texto += f"\n\nOjo: {aviso}"
                return texto
            return None
        if role != Qt.DisplayRole:
            return None
        # OJO: se usa el índice REAL de la fila (self._df.index), no su
        # posición visual -- si no, en cuanto hay un filtro activo (por
        # columna, por fecha, o el nuevo buscador de filas) la numeración
        # se corre y deja de coincidir con la fila que reportan Narrativa
        # o el buscador. Sin ningún filtro, index y posición son lo mismo,
        # así que esto no cambia nada para el caso sin filtrar.
        try:
            return str(int(self._df.index[section]) + 1)
        except (IndexError, TypeError, ValueError):
            return str(section + 1)