"""
retorno_ui.py — Pantalla del circuito de retorno: elegir dónde está lo real y
ver cuánto acertó Hadar. Todo nativo de PySide6. La lógica está en retorno.py.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QComboBox, QLabel, QMessageBox, QVBoxLayout,
)

from .retorno import TITULAR_NIVEL, comparar, leer_tabla_archivo

_ARCHIVO = "Abrir un archivo…"
_MISMO_ORDEN = "(mismo orden de filas, sin columna id)"


class DialogoComparar(QDialog):
    """Pide dónde está lo real y compara. Si todo sale bien, deja el resultado en `self.resultado`."""

    def __init__(self, parent, tablas: dict, pub):
        super().__init__(parent)
        self.setWindowTitle("Comparar con lo real")
        self.setMinimumWidth(520)
        self._tablas = dict(tablas or {})
        self._pub = pub
        self._archivos = {}            # {nombre que se ve: DataFrame leído de un archivo}
        self._anterior = None
        self.resultado = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            f"Compara lo que Hadar predijo para «{pub.objetivo}» (el {pub.calculado_en}) con lo que de verdad "
            "pasó. Elige la tabla —o un archivo— donde ya está el resultado real.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        form = QFormLayout()
        self.combo_fuente = QComboBox()
        self.combo_fuente.addItems(list(self._tablas) + [_ARCHIVO])
        self.combo_real = QComboBox()
        self.combo_id = QComboBox()
        form.addRow("Dónde está lo real", self.combo_fuente)
        form.addRow("Columna con lo real", self.combo_real)
        form.addRow("Cómo emparejar las filas", self.combo_id)
        lay.addLayout(form)

        self.lbl_ayuda = QLabel("")
        self.lbl_ayuda.setWordWrap(True)
        lay.addWidget(self.lbl_ayuda)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.button(QDialogButtonBox.Ok).setText("Comparar")
        botones.button(QDialogButtonBox.Cancel).setText("Cancelar")
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        lay.addWidget(botones)

        self.combo_fuente.currentIndexChanged.connect(self._cambio_fuente)
        if self._tablas:
            # lo más probable: otra tabla distinta de la que se predijo
            otras = [n for n in self._tablas if n != pub.tabla]
            self.combo_fuente.setCurrentText((otras or list(self._tablas))[0])
        self._cambio_fuente()

    # ------------------------------------------------------------------
    def _df_actual(self):
        nombre = self.combo_fuente.currentText()
        if nombre in self._archivos:
            return self._archivos[nombre]
        return self._tablas.get(nombre)

    def _cambio_fuente(self):
        if self.combo_fuente.currentText() == _ARCHIVO:
            ruta, _ = QFileDialog.getOpenFileName(
                self, "Archivo con lo real", "", "Datos (*.csv *.txt *.xlsx *.xls *.parquet);;Todos (*)")
            if not ruta:
                self.combo_fuente.blockSignals(True)
                self.combo_fuente.setCurrentText(self._anterior) if self._anterior else self.combo_fuente.setCurrentIndex(0)
                self.combo_fuente.blockSignals(False)
                return
            try:
                df = leer_tabla_archivo(ruta)
            except ValueError as e:
                QMessageBox.warning(self, "No se pudo abrir el archivo", str(e))
                self.combo_fuente.blockSignals(True)
                self.combo_fuente.setCurrentText(self._anterior) if self._anterior else self.combo_fuente.setCurrentIndex(0)
                self.combo_fuente.blockSignals(False)
                return
            nombre = ruta.replace("\\", "/").rsplit("/", 1)[-1]
            self._archivos[nombre] = df
            self.combo_fuente.blockSignals(True)
            if self.combo_fuente.findText(nombre) < 0:
                self.combo_fuente.insertItem(self.combo_fuente.count() - 1, nombre)
            self.combo_fuente.setCurrentText(nombre)
            self.combo_fuente.blockSignals(False)
        self._anterior = self.combo_fuente.currentText()
        self._llenar_columnas()

    def _llenar_columnas(self):
        df = self._df_actual()
        columnas = [str(c) for c in df.columns] if df is not None else []
        self.combo_real.clear()
        self.combo_real.addItems(columnas)
        if self._pub.objetivo in columnas:
            self.combo_real.setCurrentText(self._pub.objetivo)
        self.combo_id.clear()
        self.combo_id.addItem(_MISMO_ORDEN)
        if self._pub.ids is not None:
            self.combo_id.addItems(columnas)
            if self._pub.id_nombre in columnas:
                self.combo_id.setCurrentText(self._pub.id_nombre)
            self.lbl_ayuda.setText(
                f"Lo más seguro es emparejar por «{self._pub.id_nombre}», la columna que identificaba cada fila "
                "cuando se predijo.")
        else:
            self.lbl_ayuda.setText(
                "Estas predicciones no guardaron una columna que identifique cada fila: solo se pueden emparejar "
                "si la tabla con lo real tiene las mismas filas, en el mismo orden.")

    # ------------------------------------------------------------------
    def accept(self):
        df = self._df_actual()
        if df is None or not self.combo_real.currentText():
            QMessageBox.warning(self, "Falta un dato", "Elige dónde está lo real y qué columna lo trae.")
            return
        col_id = self.combo_id.currentText()
        col_id = None if col_id == _MISMO_ORDEN else col_id
        try:
            self.resultado = comparar(self._pub, df, self.combo_real.currentText(), col_id)
        except ValueError as e:
            QMessageBox.warning(self, "No se pudo comparar", str(e))
            return
        super().accept()


def mostrar_resultado(parent, d: dict):
    """El resultado de comparar, en palabras simples."""
    texto = TITULAR_NIVEL.get(d.get("nivel"), "") + "\n\n" + "\n".join("• " + l for l in d.get("lineas", []))
    QMessageBox.information(parent, f"Lo predicho contra lo real — {d.get('objetivo', '')}", texto)
