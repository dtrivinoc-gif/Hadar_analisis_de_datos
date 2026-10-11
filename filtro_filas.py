"""
Filtro de filas compartido: "calcula esto solo con las filas donde...".

Sirve para que Indicadores, Métricas y Gráficos puedan trabajar sobre una
parte de los datos (una máquina, un producto, un servicio, las lecturas sobre
cierto valor...) en vez de sobre la tabla completa -- útil cuando una misma
base mezcla cosas muy distintas entre sí.

Mismo estilo que el filtro de la pestaña Alarma: se elige una columna y se
marcan los valores (los más frecuentes primero). Además se puede comparar
números (mayor que, menor que...) y combinar hasta 3 condiciones (todas deben
cumplirse a la vez).

Un filtro es una lista de condiciones, guardable como JSON:
    [{"columna": "máquina", "operador": "es", "valores": ["A", "C"]},
     {"columna": "temperatura", "operador": ">", "valores": ["90"]}]
Lista vacía o None = todas las filas.

Este módulo tiene dos partes: las funciones puras de arriba (no saben nada de
Qt, se pueden probar solas) y las ventanas de abajo.
"""
import re

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QDialog, QDialogButtonBox,
    QMessageBox, QScrollArea, QFrame,
)

OPERADORES_FILTRO = {
    "es": "es alguno de",
    "no_es": "no es ninguno de",
    ">": "es mayor que",
    ">=": "es mayor o igual a",
    "<": "es menor que",
    "<=": "es menor o igual a",
    "vacio": "está vacío",
    "no_vacio": "no está vacío",
}
_NUMERICOS = (">", ">=", "<", "<=")
_SIN_VALOR = ("vacio", "no_vacio")
_CON_LISTA = ("es", "no_es")
MAX_CONDICIONES = 3
MAX_VALORES_FILTRO = 200   # como en Alarma: se listan los más frecuentes


# ----------------------------------------------------------------------------
# Lógica pura (sin Qt)
# ----------------------------------------------------------------------------
def _numero(texto):
    """Lee un número escrito a la chilena (1.234,5) o a la americana (1234.5).
    None si no es un número."""
    t = str(texto).strip().replace(" ", "")
    if not t:
        return None
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _mascara(df, cond):
    col, op = cond.get("columna"), cond.get("operador", "es")
    valores = cond.get("valores") or []
    nada = np.zeros(len(df), dtype=bool)
    if col not in df.columns:
        return nada
    serie = df[col]
    if op == "es":
        return serie.astype(str).isin([str(v) for v in valores]).to_numpy()
    if op == "no_es":
        return (~serie.astype(str).isin([str(v) for v in valores])).to_numpy()
    if op in _SIN_VALOR:
        vacio = (serie.isna() | (serie.astype(str).str.strip() == "")).to_numpy()
        return vacio if op == "vacio" else ~vacio
    if op in _NUMERICOS:
        n = _numero(valores[0]) if valores else None
        if n is None:
            return nada
        x = pd.to_numeric(serie, errors="coerce")
        cmp = {">": x > n, ">=": x >= n, "<": x < n, "<=": x <= n}[op]
        return cmp.fillna(False).to_numpy()
    return nada


def aplicar_filtro(df, filtro):
    """El df con solo las filas que cumplen TODAS las condiciones. Sin filtro,
    devuelve el mismo df. Si una columna del filtro no existe en este df,
    no pasa ninguna fila (en vez de fallar)."""
    if df is None or not filtro:
        return df
    mascara = np.ones(len(df), dtype=bool)
    for cond in filtro:
        mascara &= _mascara(df, cond)
    return df[mascara]


def columnas_del_filtro(filtro):
    return {c.get("columna") for c in (filtro or []) if c.get("columna")}


def describir_filtro(filtro):
    """'máquina = A, C y temperatura > 90' o '' si no hay filtro."""
    partes = []
    for c in filtro or []:
        col, op, vals = c.get("columna"), c.get("operador", "es"), [str(v) for v in (c.get("valores") or [])]
        if op in _CON_LISTA:
            lista = ", ".join(vals) if len(vals) <= 3 else f"{len(vals)} valores"
            partes.append(f"{col} {'=' if op == 'es' else '≠'} {lista}")
        elif op in _NUMERICOS:
            signo = {">": ">", ">=": "≥", "<": "<", "<=": "≤"}[op]
            partes.append(f"{col} {signo} {vals[0] if vals else '?'}")
        elif op == "vacio":
            partes.append(f"{col} está vacío")
        elif op == "no_vacio":
            partes.append(f"{col} no está vacío")
    return " y ".join(partes)


def _fechas(serie):
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie
    muestra = serie.dropna().astype(str).head(20)
    iso = len(muestra) > 0 and muestra.str.match(r"^\d{4}-\d{2}-\d{2}").all()
    return pd.to_datetime(serie, dayfirst=not iso, errors="coerce")


def aplicar_ventana(df, ventana):
    """Deja solo los últimos N días, contando hacia atrás desde la fecha más
    reciente que hay en los datos (no desde hoy: así sirve también con bases
    antiguas). ventana = {"columna": ..., "dias": N}."""
    if df is None or not ventana or not ventana.get("columna") or not ventana.get("dias"):
        return df
    col = ventana["columna"]
    if col not in df.columns:
        return df.iloc[0:0]
    fechas = _fechas(df[col])
    fin = fechas.max()
    if pd.isna(fin):
        return df.iloc[0:0]
    desde = fin - pd.Timedelta(days=int(ventana["dias"]))
    return df[(fechas > desde).fillna(False).to_numpy()]


def describir_ventana(ventana):
    if not ventana or not ventana.get("columna") or not ventana.get("dias"):
        return ""
    return f"últimos {int(ventana['dias'])} días ({ventana['columna']})"


# ----------------------------------------------------------------------------
# Ventanas
# ----------------------------------------------------------------------------
class _Condicion(QWidget):
    """Una condición: columna + cómo comparar + con qué valores."""
    quitar_pedido = Signal(object)

    def __init__(self, df, columnas, cond=None, parent=None):
        super().__init__(parent)
        self._df = df
        self._cols = [str(c) for c in columnas]

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(4)

        fila = QHBoxLayout()
        self.combo_col = QComboBox()
        # Mismo criterio que en Gráficos: un nombre de columna muy largo no
        # debe ensanchar el combo y empujar los botones fuera de la ventana.
        self.combo_col.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_col.setMinimumContentsLength(12)
        self.combo_col.addItems(self._cols)
        fila.addWidget(self.combo_col, 1)
        self.combo_op = QComboBox()
        self.combo_op.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_op.setMinimumContentsLength(12)
        fila.addWidget(self.combo_op, 1)
        btn_quitar = QPushButton("X")
        btn_quitar.setFixedWidth(28)
        btn_quitar.setToolTip("Quitar esta condición")
        btn_quitar.clicked.connect(lambda: self.quitar_pedido.emit(self))
        fila.addWidget(btn_quitar)
        lay.addLayout(fila)

        self.lista = QListWidget()
        self.lista.setMaximumHeight(110)
        lay.addWidget(self.lista)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("número")
        lay.addWidget(self.edit)
        self.lbl_nota = QLabel("")
        self.lbl_nota.setObjectName("muted")
        self.lbl_nota.setWordWrap(True)
        lay.addWidget(self.lbl_nota)

        if cond and cond.get("columna") in self._cols:
            self.combo_col.setCurrentText(cond["columna"])
        op_inicial = cond.get("operador", "es") if cond else "es"
        vals = [str(v) for v in (cond.get("valores") or [])] if cond else []
        self._llenar_operadores(op_inicial)
        self._llenar_valores(marcados=vals if op_inicial in _CON_LISTA else None)
        if op_inicial in _NUMERICOS and vals:
            self.edit.setText(vals[0])
        self._mostrar_segun_operador()

        self.combo_col.currentTextChanged.connect(self._al_cambiar_columna)
        self.combo_op.currentIndexChanged.connect(lambda _i: self._mostrar_segun_operador())

    def _es_numerica(self, col):
        return (self._df is not None and col in self._df.columns
                and pd.api.types.is_numeric_dtype(self._df[col]))

    def _llenar_operadores(self, preferido=None):
        codigos = list(OPERADORES_FILTRO) if self._es_numerica(self.combo_col.currentText()) \
            else ["es", "no_es", "vacio", "no_vacio"]
        self.combo_op.blockSignals(True)
        self.combo_op.clear()
        for c in codigos:
            self.combo_op.addItem(OPERADORES_FILTRO[c], c)
        idx = self.combo_op.findData(preferido) if preferido else 0
        self.combo_op.setCurrentIndex(max(idx, 0))
        self.combo_op.blockSignals(False)

    def _llenar_valores(self, marcados=None):
        """Lista los valores de la columna elegida, los más frecuentes primero
        (igual que en Alarma)."""
        self.lista.clear()
        self.lbl_nota.setText("")
        col = self.combo_col.currentText()
        if self._df is None or col not in self._df.columns:
            return
        conteo = self._df[col].astype(str).value_counts()
        marcados = set(str(v) for v in (marcados or []))
        for valor in conteo.index[:MAX_VALORES_FILTRO]:
            item = QListWidgetItem(str(valor))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if valor in marcados else Qt.Unchecked)
            self.lista.addItem(item)
        if len(conteo) > MAX_VALORES_FILTRO:
            self.lbl_nota.setText(
                f"Se muestran los {MAX_VALORES_FILTRO} valores más frecuentes (hay {len(conteo)}).")

    def _al_cambiar_columna(self, _texto=None):
        self._llenar_operadores()
        self._llenar_valores()
        self.edit.clear()
        self._mostrar_segun_operador()

    def _mostrar_segun_operador(self):
        op = self.combo_op.currentData()
        self.lista.setVisible(op in _CON_LISTA)
        self.lbl_nota.setVisible(op in _CON_LISTA and bool(self.lbl_nota.text()))
        self.edit.setVisible(op in _NUMERICOS)

    def get_cond(self):
        op = self.combo_op.currentData()
        if op in _CON_LISTA:
            valores = [self.lista.item(i).text() for i in range(self.lista.count())
                       if self.lista.item(i).checkState() == Qt.Checked]
        elif op in _NUMERICOS:
            valores = [self.edit.text().strip()]
        else:
            valores = []
        return {"columna": self.combo_col.currentText(), "operador": op, "valores": valores}

    def error(self):
        c = self.get_cond()
        if c["operador"] in _CON_LISTA and not c["valores"]:
            return f"En «{c['columna']}» marca al menos un valor (o quita la condición)."
        if c["operador"] in _NUMERICOS and _numero(c["valores"][0]) is None:
            return f"En «{c['columna']}» escribe un número para comparar."
        return None


class FiltroFilasWidget(QWidget):
    """Editor del filtro: hasta 3 condiciones. Sin condiciones = todas las filas."""

    def __init__(self, df, filtro=None, parent=None):
        super().__init__(parent)
        self._df = df
        self._cols = [str(c) for c in df.columns] if df is not None else []
        self._condiciones = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.caja = QVBoxLayout()
        lay.addLayout(self.caja)
        self.lbl_todas = QLabel("Se usan todas las filas.")
        self.lbl_todas.setObjectName("muted")
        lay.addWidget(self.lbl_todas)
        self.btn_agregar = QPushButton("+ Acotar a ciertas filas")
        self.btn_agregar.clicked.connect(lambda: self._agregar())
        lay.addWidget(self.btn_agregar, alignment=Qt.AlignLeft)

        for c in filtro or []:
            self._agregar(c)
        self._refrescar()

    def _agregar(self, cond=None):
        if len(self._condiciones) >= MAX_CONDICIONES or not self._cols:
            return
        w = _Condicion(self._df, self._cols, cond)
        w.quitar_pedido.connect(self._quitar)
        self.caja.addWidget(w)
        self._condiciones.append(w)
        self._refrescar()

    def _quitar(self, w):
        if w in self._condiciones:
            self._condiciones.remove(w)
        w.setParent(None)
        w.deleteLater()
        self._refrescar()

    def _refrescar(self):
        self.lbl_todas.setVisible(not self._condiciones)
        self.btn_agregar.setText("+ Y además…" if self._condiciones else "+ Acotar a ciertas filas")
        self.btn_agregar.setVisible(bool(self._cols) and len(self._condiciones) < MAX_CONDICIONES)

    def get_filtro(self):
        return [w.get_cond() for w in self._condiciones] or None

    def error(self):
        for w in self._condiciones:
            e = w.error()
            if e:
                return e
        return None


class DialogoFiltroFilas(QDialog):
    """Elegir con qué filas se calcula algo (una métrica, un gráfico...)."""

    def __init__(self, df, filtro=None, titulo="Filtrar filas", parent=None):
        super().__init__(parent)
        self.setWindowTitle(titulo)
        self.setMinimumWidth(440)
        self.resize(460, 420)
        raiz = QVBoxLayout(self)
        raiz.addWidget(QLabel("Calcular solo con las filas donde:"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.filtro_widget = FiltroFilasWidget(df, filtro)
        scroll.setWidget(self.filtro_widget)
        raiz.addWidget(scroll, 1)
        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self._aceptar)
        botones.rejected.connect(self.reject)
        raiz.addWidget(botones)

    def _aceptar(self):
        error = self.filtro_widget.error()
        if error:
            QMessageBox.information(self, "Revisa el filtro", error)
            return
        self.accept()

    def get_filtro(self):
        return self.filtro_widget.get_filtro()