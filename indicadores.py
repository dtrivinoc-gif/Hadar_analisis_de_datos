"""
Indicadores (KPIs) configurables: la clase Indicador (cálculo + formato),
sus tarjetas de UI (IndicadorCard) y el diálogo para crear/editar uno
(DialogoIndicador).

Un indicador se arma con clics: una operación sobre una columna (o la razón
entre dos), opcionalmente acotada a ciertas filas (filtro_filas.py) y/o a los
últimos N días, y con el formato elegido de una lista.
"""
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QTextEdit, QDialog, QDialogButtonBox, QScrollArea, QWidget,
    QCheckBox, QSpinBox, QMessageBox,
)

from .formulas import evaluar_formula_segura, extraer_columnas_formula, FormulaSeguridadError
from .filtro_filas import (
    aplicar_filtro, aplicar_ventana, columnas_del_filtro, describir_filtro,
    describir_ventana, FiltroFilasWidget,
)

INDICADOR_OPERACIONES = {
    "Suma": "suma",
    "Promedio": "promedio",
    "Mediana": "mediana",
    "Mínimo": "minimo",
    "Máximo": "maximo",
    "Conteo (no nulos)": "conteo",
    "Conteo de valores únicos": "conteo_unico",
    "Razón entre dos columnas (A ÷ B)": "razon",
    "Fórmula personalizada": "formula",
}
INDICADOR_OPERACIONES_INV = {v: k for k, v in INDICADOR_OPERACIONES.items()}

# Formato elegido de una lista. "personalizado" es el de antes (un texto tipo
# {value:,.0f}): los indicadores guardados antes de existir esta lista se
# siguen viendo exactamente igual.
FORMATOS = {
    "Número entero": "entero",
    "Número con 2 decimales": "decimales",
    "Porcentaje (0,25 → 25 %)": "porcentaje",
    "Personalizado (avanzado)": "personalizado",
}
FORMATOS_INV = {v: k for k, v in FORMATOS.items()}
SEPARADORES = {
    "Punto para miles (1.234,5)": "punto",
    "Coma para miles (1,234.5)": "coma",
}
SEPARADORES_INV = {v: k for k, v in SEPARADORES.items()}


class Indicador:
    """Un indicador que el propio usuario define: una operación básica sobre
    una columna, la razón entre dos columnas, o una fórmula simple combinando
    columnas con pandas. Puede acotarse a ciertas filas (`filtro`) y a los
    últimos N días (`ventana`)."""

    def __init__(self, nombre, operacion, columna=None, formula=None, formato="{value:,.0f}",
                 *, columna2=None, filtro=None, ventana=None,
                 formato_tipo="personalizado", prefijo="", sufijo="", separador="punto"):
        self.nombre = nombre
        self.operacion = operacion
        self.columna = columna
        self.columna2 = columna2          # solo para la razón A ÷ B
        self.formula = formula
        self.formato = formato or "{value:,.0f}"
        self.filtro = filtro or None      # lista de condiciones (ver filtro_filas.py)
        self.ventana = ventana or None    # {"columna": ..., "dias": N}
        self.formato_tipo = formato_tipo
        self.prefijo = prefijo or ""
        self.sufijo = sufijo or ""
        self.separador = separador

    def _acotar(self, df):
        """Las filas con las que se calcula: ya aplicado el filtro y la ventana."""
        return aplicar_ventana(aplicar_filtro(df, self.filtro), self.ventana)

    def calcular(self, df):
        if df is None or df.empty:
            return None
        df = self._acotar(df)
        if df is None or df.empty:
            return None
        try:
            if self.operacion == "formula":
                if not self.formula:
                    return None
                # Evaluador seguro basado en AST (lista blanca), no eval().
                # Ver _EvaluadorFormulaSegura para el detalle de qué se permite.
                return evaluar_formula_segura(self.formula, df)
            if not self.columna or self.columna not in df.columns:
                return None
            if self.operacion == "razon":
                if not self.columna2 or self.columna2 not in df.columns:
                    return None
                a = pd.to_numeric(df[self.columna], errors="coerce").sum()
                b = pd.to_numeric(df[self.columna2], errors="coerce").sum()
                return None if b == 0 else a / b
            serie = df[self.columna]
            if self.operacion == "suma":
                return pd.to_numeric(serie, errors="coerce").sum()
            if self.operacion == "promedio":
                return pd.to_numeric(serie, errors="coerce").mean()
            if self.operacion == "mediana":
                return pd.to_numeric(serie, errors="coerce").median()
            if self.operacion == "minimo":
                return pd.to_numeric(serie, errors="coerce").min()
            if self.operacion == "maximo":
                return pd.to_numeric(serie, errors="coerce").max()
            if self.operacion == "conteo":
                return serie.count()
            if self.operacion == "conteo_unico":
                return serie.nunique()
        except Exception:
            return None
        return None

    def columnas_de_las_que_depende(self):
        """De qué columnas del DataFrame depende el VALOR de este indicador --
        la base del linaje: sirve para saber, sin recalcular nada, si una
        columna con anomalías activas afecta a este indicador. Para una
        operación simple (Suma, Promedio, etc.) es solo `self.columna`; para
        una razón, las dos; para una fórmula personalizada, se obtiene
        recorriendo su AST (ver formulas.extraer_columnas_formula). Nunca
        lanza una excepción hacia afuera -- una fórmula con error de sintaxis
        simplemente no aporta columnas conocidas todavía, en vez de romper la
        tarjeta del indicador."""
        if self.operacion == "formula":
            if not self.formula:
                return set()
            try:
                return extraer_columnas_formula(self.formula)
            except FormulaSeguridadError:
                return set()
        columnas = {self.columna} if self.columna else set()
        if self.operacion == "razon" and self.columna2:
            columnas.add(self.columna2)
        return columnas

    def columnas_necesarias(self):
        """Todas las columnas que hacen falta para calcularlo: las del valor
        más las del filtro y la ventana de tiempo. El Panel lo usa para saber
        si el indicador se puede calcular con la tabla que está mostrando."""
        extra = set(columnas_del_filtro(self.filtro))
        if self.ventana and self.ventana.get("columna"):
            extra.add(self.ventana["columna"])
        return self.columnas_de_las_que_depende() | extra

    def describir_acotado(self):
        """'Solo donde máquina = A · últimos 30 días (fecha)' o '' si usa todo."""
        partes = []
        f = describir_filtro(self.filtro)
        if f:
            partes.append(f"Solo donde {f}")
        v = describir_ventana(self.ventana)
        if v:
            partes.append(v)
        return " · ".join(partes)

    def formatear(self, valor):
        if valor is None:
            return "-"
        if self.formato_tipo == "personalizado":
            try:
                return self.formato.format(value=valor)
            except Exception:
                try:
                    return f"{valor:,.2f}"
                except Exception:
                    return str(valor)
        try:
            v = float(valor)
        except (TypeError, ValueError):
            return str(valor)
        if not np.isfinite(v):
            return "-"
        decimales = 0
        if self.formato_tipo == "decimales":
            decimales = 2
        elif self.formato_tipo == "porcentaje":
            v *= 100
            decimales = 1
        texto = f"{v:,.{decimales}f}"
        if self.separador == "punto":
            texto = texto.translate({ord(","): ".", ord("."): ","})
        if self.formato_tipo == "porcentaje":
            texto += " %"
        return f"{self.prefijo}{texto}{self.sufijo}"

    def to_dict(self):
        return {
            "nombre": self.nombre,
            "operacion": self.operacion,
            "columna": self.columna,
            "columna2": self.columna2,
            "formula": self.formula,
            "formato": self.formato,
            "filtro": self.filtro,
            "ventana": self.ventana,
            "formato_tipo": self.formato_tipo,
            "prefijo": self.prefijo,
            "sufijo": self.sufijo,
            "separador": self.separador,
        }

    @classmethod
    def from_dict(cls, data):
        # Los indicadores guardados antes de estas opciones no traen las
        # claves nuevas: quedan sin filtro y con su formato de siempre.
        return cls(
            nombre=data.get("nombre", "Indicador"),
            operacion=data.get("operacion", "suma"),
            columna=data.get("columna"),
            formula=data.get("formula"),
            formato=data.get("formato", "{value:,.0f}"),
            columna2=data.get("columna2"),
            filtro=data.get("filtro"),
            ventana=data.get("ventana"),
            formato_tipo=data.get("formato_tipo", "personalizado"),
            prefijo=data.get("prefijo", ""),
            sufijo=data.get("sufijo", ""),
            separador=data.get("separador", "punto"),
        )


def sugerir_indicadores_por_defecto(df):
    """Propone algunos indicadores razonables a partir de las columnas del
    DataFrame, solo como punto de partida: el usuario los puede editar o
    borrar después."""
    indicadores = []
    if df is None or df.empty:
        return indicadores

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        col = num_cols[0]
        indicadores.append(Indicador(f"Total {col}", "suma", col, formato_tipo="entero"))
        indicadores.append(Indicador(f"Promedio {col}", "promedio", col, formato_tipo="decimales"))
        if len(num_cols) > 1:
            col2 = num_cols[1]
            indicadores.append(Indicador(
                f"Razón {col} ÷ {col2}", "razon", col, columna2=col2, formato_tipo="porcentaje",
            ))

    texto_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in texto_cols[:2]:
        if df[col].nunique() < 50:
            indicadores.append(Indicador(f"Categorías únicas en {col}", "conteo_unico", col,
                                          formato_tipo="entero"))

    return indicadores


class IndicadorCard(QFrame):
    """Tarjeta de un indicador: nombre, valor calculado, y botones para
    editarlo o eliminarlo. Si el indicador tiene una regla de Alarma
    asociada y está activada, se marca con marco rojo y un mensaje."""

    def __init__(self, indicador, valor, on_editar, on_eliminar, parent=None,
                 en_alarma=False, mensaje_alarma=None, columnas_con_anomalias=None,
                 on_panel=None):
        super().__init__(parent)
        self.indicador = indicador
        self.setObjectName("statCard")
        self.setProperty("alarma", "true" if en_alarma else "false")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)

        lbl_titulo = QLabel(indicador.nombre)
        lbl_titulo.setObjectName("statTitle")
        lbl_titulo.setWordWrap(True)
        layout.addWidget(lbl_titulo)

        lbl_valor = QLabel(indicador.formatear(valor))
        lbl_valor.setObjectName("statValue")
        layout.addWidget(lbl_valor)

        acotado = indicador.describir_acotado()
        if acotado:
            lbl_acotado = QLabel(acotado)
            lbl_acotado.setWordWrap(True)
            lbl_acotado.setStyleSheet("color: #9CA3AF; font-size: 11px;")
            layout.addWidget(lbl_acotado)

        columnas_dependencia = sorted(indicador.columnas_de_las_que_depende())
        if columnas_dependencia:
            lbl_depende = QLabel("Depende de: " + ", ".join(columnas_dependencia))
            lbl_depende.setWordWrap(True)
            lbl_depende.setStyleSheet("color: #9CA3AF; font-size: 11px;")
            layout.addWidget(lbl_depende)

        # No es una Alarma (esa la configura el usuario a mano sobre el
        # indicador) -- esto es más simple: si Narrativa ya detectó una
        # anomalía en alguna columna de la que depende este indicador, se
        # avisa acá también, sin tener que ir a revisar la otra pestaña.
        columnas_afectadas = sorted(set(columnas_dependencia) & (columnas_con_anomalias or set()))
        if columnas_afectadas:
            lbl_dependencia_riesgo = QLabel(
                "⚠ " + ", ".join(f"'{c}'" for c in columnas_afectadas)
                + (" tiene" if len(columnas_afectadas) == 1 else " tienen")
                + " una anomalía activa (ver Narrativa)"
            )
            lbl_dependencia_riesgo.setWordWrap(True)
            lbl_dependencia_riesgo.setStyleSheet("color: #b45309; font-size: 11px; font-weight: bold;")
            layout.addWidget(lbl_dependencia_riesgo)

        if en_alarma and mensaje_alarma:
            lbl_alarma = QLabel(f"⚠ {mensaje_alarma}")
            lbl_alarma.setWordWrap(True)
            lbl_alarma.setStyleSheet("color: #EF4444; font-weight: bold;")
            layout.addWidget(lbl_alarma)

        fila_botones = QHBoxLayout()
        fila_botones.addStretch()
        if on_panel is not None:
            btn_panel = QPushButton("Al Panel")
            btn_panel.setToolTip("Muestra este indicador también en la pestaña Panel.")
            btn_panel.clicked.connect(lambda: on_panel(indicador))
            fila_botones.addWidget(btn_panel)
        btn_editar = QPushButton("Editar")
        btn_editar.clicked.connect(lambda: on_editar(indicador))
        fila_botones.addWidget(btn_editar)
        btn_eliminar = QPushButton("X")
        btn_eliminar.setFixedWidth(28)
        btn_eliminar.clicked.connect(lambda: on_eliminar(indicador))
        fila_botones.addWidget(btn_eliminar)
        layout.addLayout(fila_botones)


class DialogoIndicador(QDialog):
    """Crear o editar un indicador. Si se entrega `indicador`, se abre en
    modo edición pre-llenado con sus valores actuales. `df` se usa solo para
    listar los valores posibles del filtro de filas."""

    def __init__(self, columnas, indicador=None, parent=None, df=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Indicador" if indicador else "Nuevo Indicador")
        self.setMinimumWidth(440)
        self.resize(470, 660)
        columnas = [str(c) for c in columnas]
        raiz = QVBoxLayout(self)

        # Todo dentro de un área con scroll: con el filtro y el formato la
        # ventana ya no cabe en pantallas chicas.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        interior = QWidget()
        layout = QVBoxLayout(interior)
        scroll.setWidget(interior)
        raiz.addWidget(scroll, 1)

        layout.addWidget(QLabel("Nombre:"))
        self.nombre_edit = QLineEdit(indicador.nombre if indicador else "")
        layout.addWidget(self.nombre_edit)

        layout.addWidget(QLabel("Operación:"))
        self.operacion_combo = QComboBox()
        self.operacion_combo.addItems(list(INDICADOR_OPERACIONES.keys()))
        if indicador:
            self.operacion_combo.setCurrentText(
                INDICADOR_OPERACIONES_INV.get(indicador.operacion, "Suma")
            )
        layout.addWidget(self.operacion_combo)

        self.lbl_columna = QLabel("Columna:")
        layout.addWidget(self.lbl_columna)
        self.columna_combo = QComboBox()
        self.columna_combo.addItems(columnas)
        if indicador and indicador.columna in columnas:
            self.columna_combo.setCurrentText(str(indicador.columna))
        layout.addWidget(self.columna_combo)

        self.lbl_columna2 = QLabel("Columna B (la que divide):")
        layout.addWidget(self.lbl_columna2)
        self.columna2_combo = QComboBox()
        self.columna2_combo.addItems(columnas)
        if indicador and indicador.columna2 in columnas:
            self.columna2_combo.setCurrentText(str(indicador.columna2))
        elif len(columnas) > 1:
            self.columna2_combo.setCurrentIndex(1)
        layout.addWidget(self.columna2_combo)

        self.lbl_formula = QLabel(
            "Fórmula (usa df para tus datos, ej: df['Ventas'].sum() / df['Costos'].sum() * 100):"
        )
        layout.addWidget(self.lbl_formula)
        self.formula_edit = QTextEdit(indicador.formula if (indicador and indicador.formula) else "")
        self.formula_edit.setMaximumHeight(70)
        layout.addWidget(self.formula_edit)

        # -- Acotar el cálculo ------------------------------------------------
        layout.addSpacing(6)
        layout.addWidget(QLabel("Calcular solo con las filas donde (opcional):"))
        self.filtro_widget = FiltroFilasWidget(df, indicador.filtro if indicador else None)
        layout.addWidget(self.filtro_widget)

        self.chk_ventana = QCheckBox("Solo los últimos")
        fila_v = QHBoxLayout()
        fila_v.addWidget(self.chk_ventana)
        self.dias_spin = QSpinBox()
        self.dias_spin.setRange(1, 3650)
        self.dias_spin.setValue(30)
        fila_v.addWidget(self.dias_spin)
        fila_v.addWidget(QLabel("días, según la columna:"))
        self.fecha_combo = QComboBox()
        self.fecha_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.fecha_combo.setMinimumContentsLength(10)
        self.fecha_combo.addItems(columnas)
        fila_v.addWidget(self.fecha_combo, 1)
        layout.addLayout(fila_v)
        nota_v = QLabel("Se cuenta hacia atrás desde la fecha más reciente que haya en los datos.")
        nota_v.setObjectName("muted")
        nota_v.setWordWrap(True)
        layout.addWidget(nota_v)
        if indicador and indicador.ventana and indicador.ventana.get("columna") in columnas:
            self.chk_ventana.setChecked(True)
            self.dias_spin.setValue(int(indicador.ventana.get("dias") or 30))
            self.fecha_combo.setCurrentText(indicador.ventana["columna"])
        self.chk_ventana.toggled.connect(self._actualizar_ventana)
        self._actualizar_ventana()

        # -- Formato ----------------------------------------------------------
        layout.addSpacing(6)
        layout.addWidget(QLabel("Cómo se muestra el número:"))
        self.formato_combo = QComboBox()
        self.formato_combo.addItems(list(FORMATOS.keys()))
        tipo_actual = indicador.formato_tipo if indicador else "entero"
        self.formato_combo.setCurrentText(FORMATOS_INV.get(tipo_actual, "Número entero"))
        layout.addWidget(self.formato_combo)

        self.fila_prefijo = QWidget()
        lf = QHBoxLayout(self.fila_prefijo)
        lf.setContentsMargins(0, 0, 0, 0)
        lf.addWidget(QLabel("Antes:"))
        self.prefijo_edit = QLineEdit(indicador.prefijo if indicador else "")
        self.prefijo_edit.setPlaceholderText("ej: $")
        lf.addWidget(self.prefijo_edit)
        lf.addWidget(QLabel("Después:"))
        self.sufijo_edit = QLineEdit(indicador.sufijo if indicador else "")
        self.sufijo_edit.setPlaceholderText("ej: °C, días, kg")
        lf.addWidget(self.sufijo_edit)
        layout.addWidget(self.fila_prefijo)

        self.separador_combo = QComboBox()
        self.separador_combo.addItems(list(SEPARADORES.keys()))
        self.separador_combo.setCurrentText(
            SEPARADORES_INV.get(indicador.separador if indicador else "punto", "Punto para miles (1.234,5)"))
        layout.addWidget(self.separador_combo)

        self.formato_edit = QLineEdit(indicador.formato if indicador else "{value:,.0f}")
        self.formato_edit.setToolTip("Ejemplos: {value:,.0f}  ·  ${value:,.2f}  ·  {value:.1f}%")
        layout.addWidget(self.formato_edit)

        self.operacion_combo.currentTextChanged.connect(self._actualizar_campos_visibles)
        self.formato_combo.currentTextChanged.connect(self._actualizar_formato_visible)
        self._actualizar_campos_visibles(self.operacion_combo.currentText())
        self._actualizar_formato_visible(self.formato_combo.currentText())
        layout.addStretch(1)

        botones = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        botones.accepted.connect(self._validar_y_aceptar)
        botones.rejected.connect(self.reject)
        raiz.addWidget(botones)

    def _actualizar_campos_visibles(self, operacion_label):
        es_formula = operacion_label == "Fórmula personalizada"
        es_razon = operacion_label == "Razón entre dos columnas (A ÷ B)"
        self.lbl_columna.setVisible(not es_formula)
        self.columna_combo.setVisible(not es_formula)
        self.lbl_columna.setText("Columna A (la que se divide):" if es_razon else "Columna:")
        self.lbl_columna2.setVisible(es_razon)
        self.columna2_combo.setVisible(es_razon)
        self.lbl_formula.setVisible(es_formula)
        self.formula_edit.setVisible(es_formula)

    def _actualizar_ventana(self, _marcado=None):
        activo = self.chk_ventana.isChecked()
        self.dias_spin.setEnabled(activo)
        self.fecha_combo.setEnabled(activo)

    def _actualizar_formato_visible(self, etiqueta):
        personalizado = FORMATOS.get(etiqueta) == "personalizado"
        self.formato_edit.setVisible(personalizado)
        self.fila_prefijo.setVisible(not personalizado)
        self.separador_combo.setVisible(not personalizado)

    def _validar_y_aceptar(self):
        error = self.filtro_widget.error()
        if error:
            QMessageBox.information(self, "Revisa el filtro", error)
            return
        self.accept()

    def get_indicador(self):
        operacion_label = self.operacion_combo.currentText()
        operacion = INDICADOR_OPERACIONES.get(operacion_label, "suma")
        ventana = None
        if self.chk_ventana.isChecked():
            ventana = {"columna": self.fecha_combo.currentText(), "dias": int(self.dias_spin.value())}
        return Indicador(
            nombre=self.nombre_edit.text().strip() or "Indicador",
            operacion=operacion,
            columna=self.columna_combo.currentText() if operacion != "formula" else None,
            formula=self.formula_edit.toPlainText().strip() if operacion == "formula" else None,
            formato=self.formato_edit.text().strip() or "{value:,.0f}",
            columna2=self.columna2_combo.currentText() if operacion == "razon" else None,
            filtro=self.filtro_widget.get_filtro(),
            ventana=ventana,
            formato_tipo=FORMATOS.get(self.formato_combo.currentText(), "entero"),
            prefijo=self.prefijo_edit.text(),
            sufijo=self.sufijo_edit.text(),
            separador=SEPARADORES.get(self.separador_combo.currentText(), "punto"),
        )


# ----------------------------------------------------------------------------
# Transformación estilo Excel: crear una columna nueva combinando columnas
# existentes con una fórmula simple, aplicada a todas las filas de una sola
# vez -- el equivalente a escribir una fórmula en la primera celda de Excel y
# arrastrarla hacia abajo. Con esto, cualquier cálculo (el que antes solo se
# veía como una tarjeta en Indicadores o un recuadro de +Cálculo) puede
# convertirse en una columna real, disponible para Gráficos, Métricas,
# Frecuencias, Narrativa y Reporte -- igual que cualquier columna cargada desde
# archivo. Filosofía Hadar: que se arme haciendo clic, sin sintaxis que
# aprender.
# ----------------------------------------------------------------------------