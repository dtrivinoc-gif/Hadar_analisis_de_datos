"""
Función "Alarma": el usuario define un umbral sobre un Indicador, una
Métrica (media/mediana/moda/desviación estándar de una columna numérica),
una Frecuencia (conteo de la categoría más repetida de una columna) o una
Columna (se revisa FILA POR FILA: si alguna fila cumple la condición, la
alarma se activa), y Hadar avisa visualmente cuando se cumple.

Las alarmas de tipo Columna pueden acotarse a ciertas filas (ej. solo la
máquina A y C), porque en bases planas hay productos o máquinas con
niveles normales muy distintos entre sí.

Este módulo tiene tres partes:
- El motor puro (AlarmaHadar, calcular_valor_actual, evaluar_regla): no
  sabe nada de Qt, solo guarda reglas en SQLite y las evalúa contra un
  DataFrame. Mismo patrón de persistencia que memoria.py.
- AlarmaCard: la tarjeta de una regla en la pestaña "Alarma".
- DialogoAlarma: crear/editar una regla.

La MARCA visual (marco rojo) sobre las tarjetas de Indicadores/Métricas/
Frecuencias la aplica main_window.py directamente sobre esos widgets --
este módulo solo dice "esta regla está en alarma sí/no".
"""
import json
import operator
import os
import sqlite3
from contextlib import contextmanager

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QComboBox, QPushButton, QFrame, QMessageBox, QCheckBox,
    QListWidget, QListWidgetItem, QScrollArea, QApplication, QWidget,
)
from PySide6.QtCore import Qt

from .memoria import ruta_directorio_datos_usuario

OPERADORES = {
    "Mayor que (>)": operator.gt,
    "Mayor o igual (≥)": operator.ge,
    "Menor que (<)": operator.lt,
    "Menor o igual (≤)": operator.le,
    "Igual a (=)": operator.eq,
}

ESTADISTICOS_METRICA = {
    "Media": lambda serie: serie.mean(),
    "Mediana": lambda serie: serie.median(),
    "Moda": lambda serie: serie.mode().iloc[0] if not serie.mode().empty else None,
    "Desviación Std": lambda serie: serie.std(),
}

TIPOS_ALARMA = ["Indicador", "Métrica", "Frecuencia", "Columna"]

# Máximo de valores que se listan al acotar una alarma de Columna (se
# muestran los más frecuentes si la columna tiene más).
MAX_VALORES_FILTRO = 200

# Cuántas filas se listan (en la tarjeta y en el correo) cuando una alarma de
# Columna se activa; si hay más, se dice cuántas faltan.
MAX_FILAS_LISTADAS = 5


def ruta_alarmas_db():
    return os.path.join(ruta_directorio_datos_usuario(), "hadar_alarmas.db")


@contextmanager
def _conexion(ruta_db):
    con = sqlite3.connect(ruta_db)
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ----------------------------------------------------------------------------
# Motor: guardar y evaluar reglas
# ----------------------------------------------------------------------------

class AlarmaHadar:
    """Guarda y evalúa las reglas de alarma configuradas por el usuario.
    Persisten en SQLite (misma carpeta que la memoria de anomalías) para
    que sobrevivan a cerrar y volver a abrir Hadar."""

    def __init__(self, ruta_db=None):
        self.ruta_db = ruta_db or ruta_alarmas_db()
        self._crear_tabla()

    def _crear_tabla(self):
        with _conexion(self.ruta_db) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS reglas_alarma (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre TEXT NOT NULL,
                    tipo TEXT NOT NULL,
                    objetivo TEXT NOT NULL,
                    estadistico TEXT,
                    operador TEXT NOT NULL,
                    umbral REAL NOT NULL,
                    mensaje TEXT,
                    activa INTEGER NOT NULL DEFAULT 1,
                    notificar_correo INTEGER NOT NULL DEFAULT 0,
                    correo_destino TEXT,
                    filtro_columna TEXT,
                    filtro_valores TEXT,
                    columna_id TEXT
                )
            """)
            # Migración suave: si la tabla ya existía de antes de agregar
            # el aviso por correo, le sumamos las columnas que falten sin
            # perder las reglas ya guardadas.
            columnas_existentes = {
                fila[1] for fila in con.execute("PRAGMA table_info(reglas_alarma)")
            }
            if "notificar_correo" not in columnas_existentes:
                con.execute(
                    "ALTER TABLE reglas_alarma ADD COLUMN notificar_correo INTEGER NOT NULL DEFAULT 0"
                )
            if "correo_destino" not in columnas_existentes:
                con.execute("ALTER TABLE reglas_alarma ADD COLUMN correo_destino TEXT")
            if "filtro_columna" not in columnas_existentes:
                con.execute("ALTER TABLE reglas_alarma ADD COLUMN filtro_columna TEXT")
            if "filtro_valores" not in columnas_existentes:
                con.execute("ALTER TABLE reglas_alarma ADD COLUMN filtro_valores TEXT")
            if "columna_id" not in columnas_existentes:
                con.execute("ALTER TABLE reglas_alarma ADD COLUMN columna_id TEXT")

    def listar_reglas(self):
        columnas = ["id", "nombre", "tipo", "objetivo", "estadistico",
                    "operador", "umbral", "mensaje", "activa",
                    "notificar_correo", "correo_destino",
                    "filtro_columna", "filtro_valores", "columna_id"]
        with _conexion(self.ruta_db) as con:
            filas = con.execute(
                f"SELECT {', '.join(columnas)} FROM reglas_alarma ORDER BY id"
            ).fetchall()
        reglas = [dict(zip(columnas, f)) for f in filas]
        for regla in reglas:
            regla["filtro_valores"] = _leer_valores(regla["filtro_valores"])
        return reglas

    def agregar_regla(self, nombre, tipo, objetivo, operador, umbral,
                       estadistico=None, mensaje="", notificar_correo=0, correo_destino=None,
                       filtro_columna=None, filtro_valores=None, columna_id=None):
        with _conexion(self.ruta_db) as con:
            cur = con.execute(
                "INSERT INTO reglas_alarma "
                "(nombre, tipo, objetivo, estadistico, operador, umbral, mensaje, activa, "
                "notificar_correo, correo_destino, filtro_columna, filtro_valores, columna_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
                (nombre, tipo, objetivo, estadistico, operador, umbral, mensaje,
                 notificar_correo, correo_destino, filtro_columna,
                 _guardar_valores(filtro_valores), columna_id),
            )
            return cur.lastrowid

    def actualizar_regla(self, regla_id, **campos):
        if not campos:
            return
        if "filtro_valores" in campos:
            campos["filtro_valores"] = _guardar_valores(campos["filtro_valores"])
        sets = ", ".join(f"{k} = ?" for k in campos)
        valores = list(campos.values()) + [regla_id]
        with _conexion(self.ruta_db) as con:
            con.execute(f"UPDATE reglas_alarma SET {sets} WHERE id = ?", valores)

    def eliminar_regla(self, regla_id):
        with _conexion(self.ruta_db) as con:
            con.execute("DELETE FROM reglas_alarma WHERE id = ?", (regla_id,))


def _guardar_valores(valores):
    """Lista de valores del filtro -> texto JSON para SQLite (None si no hay)."""
    if not valores:
        return None
    return json.dumps([str(v) for v in valores], ensure_ascii=False)


def _leer_valores(texto):
    if not texto:
        return []
    try:
        valores = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(v) for v in valores] if isinstance(valores, list) else []


def evaluar_columna(regla, df):
    """Revisa fila por fila una alarma de tipo Columna.

    Devuelve None si no se puede revisar ahora (la columna vigilada o la
    del filtro ya no están en los datos cargados). Si se puede, devuelve:
      revisadas: cuántas filas se miraron (ya aplicado el filtro)
      cumplen:   cuántas de ellas cumplen la condición
      extremo:   el valor más alto (si la condición es > o ≥), el más bajo
                 (si es < o ≤) o el primero que coincide (si es =); None
                 si ninguna cumple
      filas:     hasta MAX_FILAS_LISTADAS filas que la activaron, las de valor
                 más extremo primero (en el orden del archivo si la condición
                 es =). Cada una: {"fila": número como se ve en la pestaña
                 Datos, "valor": ..., "id": texto de la columna que
                 identifica la fila, o None}
      mas:       cuántas filas que cumplen quedaron sin listar
    Las celdas vacías o con texto en la columna vigilada no se revisan."""
    if df is None or df.empty:
        return None
    columna = regla["objetivo"]
    if columna not in df.columns:
        return None
    funcion_operador = OPERADORES.get(regla["operador"])
    if funcion_operador is None:
        return None

    filtro_columna = regla.get("filtro_columna")
    valores_filtro = regla.get("filtro_valores") or []
    if filtro_columna and valores_filtro:
        if filtro_columna not in df.columns:
            return None
        df = df[df[filtro_columna].astype(str).isin([str(v) for v in valores_filtro])]

    numeros = pd.to_numeric(df[columna], errors="coerce")
    validos = numeros.notna()
    resultado = {"revisadas": int(validos.sum()), "cumplen": 0, "extremo": None,
                 "filas": [], "mas": 0}
    if not validos.any():
        return resultado
    try:
        cumple = (funcion_operador(numeros, regla["umbral"]) & validos).to_numpy()
    except TypeError:
        return None
    resultado["cumplen"] = int(cumple.sum())
    if not resultado["cumplen"]:
        return resultado

    sub = df[cumple]
    valores = numeros[cumple].to_numpy(dtype=float)
    if regla["operador"] in ("Mayor que (>)", "Mayor o igual (≥)"):
        orden = np.argsort(-valores, kind="stable")
        resultado["extremo"] = float(valores.max())
    elif regla["operador"] in ("Menor que (<)", "Menor o igual (≤)"):
        orden = np.argsort(valores, kind="stable")
        resultado["extremo"] = float(valores.min())
    else:
        orden = np.arange(len(valores))
        resultado["extremo"] = float(valores[0])

    columna_id = regla.get("columna_id")
    ids = sub[columna_id] if columna_id and columna_id in sub.columns else None
    for k in orden[:MAX_FILAS_LISTADAS]:
        # Mismo número que muestra la pestaña Datos (índice real + 1).
        try:
            numero_fila = int(sub.index[k]) + 1
        except (TypeError, ValueError):
            numero_fila = int(k) + 1
        id_texto = None
        if ids is not None and pd.notna(ids.iloc[k]):
            id_texto = str(ids.iloc[k])
        resultado["filas"].append({"fila": numero_fila, "valor": float(valores[k]), "id": id_texto})
    resultado["mas"] = resultado["cumplen"] - len(resultado["filas"])
    return resultado


def _num(valor):
    """Número con punto de miles y coma decimal solo si hace falta."""
    if float(valor).is_integer():
        return f"{int(valor):,}".replace(",", ".")
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def describir_filtro(regla):
    """'máquina = A, C' o '' si la alarma mira todas las filas."""
    if not regla.get("filtro_columna") or not regla.get("filtro_valores"):
        return ""
    valores = regla["filtro_valores"]
    texto = ", ".join(valores) if len(valores) <= 3 else f"{len(valores)} valores"
    return f"{regla['filtro_columna']} = {texto}"


def estado_en_palabras(regla, valor_actual, resumen=None):
    """Una frase corta y simple con cómo va la alarma (la usan la pestaña
    Alarma y el Panel)."""
    if valor_actual is None:
        return "Sin datos para evaluar"
    if regla["tipo"] == "Columna":
        if resumen is None or resumen["revisadas"] == 0:
            return "No hay filas que revisar con ese filtro"
        n, total = resumen["cumplen"], resumen["revisadas"]
        if n == 0:
            return f"Ninguna de las {_num(total)} filas la cumple" if total != 1 else "La fila revisada no la cumple"
        if n == 1:
            return f"1 de {_num(total)} filas cumple la condición"
        return f"{_num(n)} de {_num(total)} filas cumplen la condición"
    return f"Valor actual: {valor_actual:,.2f}".replace(",", ".")


def lineas_filas(regla, resumen):
    """Líneas de texto con las filas que activaron una alarma de Columna:
    'Fila 842 · maquina: A · consumo: 120', y 'y 12 más' si quedaron sin
    listar. Lista vacía si no hay nada que mostrar."""
    if resumen is None or not resumen.get("filas"):
        return []
    lineas = []
    for f in resumen["filas"]:
        partes = [f"Fila {_num(f['fila'])}"]
        if f.get("id") is not None:
            partes.append(f"{regla['columna_id']}: {f['id']}")
        partes.append(f"{regla['objetivo']}: {_num(f['valor'])}")
        lineas.append(" · ".join(partes))
    if resumen.get("mas", 0) > 0:
        lineas.append(f"y {_num(resumen['mas'])} más")
    return lineas


def texto_detalle_columna(regla, resumen):
    """Texto para el correo de una alarma de Columna."""
    lineas = [f"Columna vigilada: {regla['objetivo']}",
              f"Condición: {regla['operador']} {_num(regla['umbral'])}"]
    filtro = describir_filtro(regla)
    lineas.append(f"Solo en las filas donde: {filtro}" if filtro else "Se revisaron todas las filas")
    if resumen is not None and resumen["revisadas"]:
        lineas.append(f"Filas que cumplen: {_num(resumen['cumplen'])} de {_num(resumen['revisadas'])} revisadas")
        if resumen["extremo"] is not None:
            etiqueta = {"Mayor que (>)": "Valor más alto", "Mayor o igual (≥)": "Valor más alto",
                        "Menor que (<)": "Valor más bajo", "Menor o igual (≤)": "Valor más bajo"}
            lineas.append(f"{etiqueta.get(regla['operador'], 'Valor encontrado')}: {_num(resumen['extremo'])}")
    filas = lineas_filas(regla, resumen)
    if filas:
        orden = {"Mayor que (>)": "de mayor a menor", "Mayor o igual (≥)": "de mayor a menor",
                 "Menor que (<)": "de menor a mayor", "Menor o igual (≤)": "de menor a mayor"}
        cuando = orden.get(regla["operador"], "en el orden del archivo")
        lineas.append("")
        lineas.append(f"Filas que la activaron ({cuando}; el número es el de la pestaña Datos):")
        lineas.extend(f"  {l}" for l in filas)
    return "\n".join(lineas) + "\n"


def calcular_valor_actual(regla, df, indicadores):
    """Valor actual de lo que la regla vigila, o None si no se puede
    calcular ahora mismo (ej. la columna ya no existe en los datos
    cargados, o el indicador fue borrado)."""
    if df is None or df.empty:
        return None

    if regla["tipo"] == "Columna":
        # Aquí el "valor actual" es cuántas filas cumplen la condición.
        resumen = evaluar_columna(regla, df)
        return None if resumen is None else resumen["cumplen"]

    if regla["tipo"] == "Indicador":
        indicador = next((i for i in indicadores if i.nombre == regla["objetivo"]), None)
        if indicador is None:
            return None
        try:
            return indicador.calcular(df)
        except Exception:
            return None

    if regla["tipo"] == "Métrica":
        columna = regla["objetivo"]
        if columna not in df.columns:
            return None
        serie = pd.to_numeric(df[columna], errors="coerce").dropna()
        if serie.empty:
            return None
        funcion = ESTADISTICOS_METRICA.get(regla["estadistico"])
        if funcion is None:
            return None
        try:
            valor = funcion(serie)
            return float(valor) if valor is not None else None
        except (TypeError, ValueError):
            return None

    if regla["tipo"] == "Frecuencia":
        columna = regla["objetivo"]
        if columna not in df.columns:
            return None
        serie = df[columna].dropna().astype(str)
        if serie.empty:
            return None
        return int(serie.value_counts().max())

    return None


def evaluar_regla(regla, valor_actual):
    """True si el valor actual dispara la alarma."""
    if valor_actual is None or not regla.get("activa", True):
        return False
    if regla["tipo"] == "Columna":
        # valor_actual = cuántas filas cumplen; basta con que haya una.
        return valor_actual > 0
    funcion_operador = OPERADORES.get(regla["operador"])
    if funcion_operador is None:
        return False
    try:
        return bool(funcion_operador(valor_actual, regla["umbral"]))
    except TypeError:
        return False


# ----------------------------------------------------------------------------
# UI: tarjeta de una regla + diálogo para crearla/editarla
# ----------------------------------------------------------------------------

class AlarmaCard(QFrame):
    """Tarjeta de una regla de alarma: nombre, condición en palabras,
    estado actual (🟢/🔴), valor actual, y botones editar/eliminar."""

    def __init__(self, regla, valor_actual, en_alarma, on_editar, on_eliminar, parent=None,
                 resumen=None):
        super().__init__(parent)
        self.regla = regla
        self.setObjectName("statCard")
        self.setProperty("alarma", "true" if en_alarma else "false")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)

        fila_titulo = QHBoxLayout()
        icono = QLabel("🔴" if en_alarma else "🟢")
        fila_titulo.addWidget(icono)
        lbl_nombre = QLabel(f"<b>{regla['nombre']}</b>")
        lbl_nombre.setWordWrap(True)
        fila_titulo.addWidget(lbl_nombre, stretch=1)
        layout.addLayout(fila_titulo)

        condicion = _describir_condicion(regla)
        lbl_condicion = QLabel(condicion)
        lbl_condicion.setObjectName("muted")
        lbl_condicion.setWordWrap(True)
        layout.addWidget(lbl_condicion)

        if regla["tipo"] == "Columna":
            texto_valor = estado_en_palabras(regla, valor_actual, resumen)
            if resumen is not None and resumen.get("extremo") is not None:
                etiqueta = "Valor más bajo" if regla["operador"] in ("Menor que (<)", "Menor o igual (≤)") \
                    else "Valor más alto" if regla["operador"] in ("Mayor que (>)", "Mayor o igual (≥)") \
                    else "Valor encontrado"
                texto_valor += f"\n{etiqueta}: {_num(resumen['extremo'])}"
            filas = lineas_filas(regla, resumen)
            if filas and en_alarma:
                texto_valor += "\n\nFilas que la activaron:\n" + "\n".join(filas)
        else:
            texto_valor = "Sin datos" if valor_actual is None else f"Valor actual: {valor_actual:,.2f}".replace(",", ".")
        lbl_valor = QLabel(texto_valor)
        lbl_valor.setWordWrap(True)
        layout.addWidget(lbl_valor)

        texto_correo = (
            f"📧 Avisa por correo a {regla['correo_destino']}" if regla.get("notificar_correo") and regla.get("correo_destino")
            else "📧 Avisa por correo (al predeterminado)" if regla.get("notificar_correo")
            else "📧 Sin aviso por correo"
        )
        lbl_correo = QLabel(texto_correo)
        lbl_correo.setObjectName("muted")
        layout.addWidget(lbl_correo)

        if en_alarma and regla.get("mensaje"):
            lbl_mensaje = QLabel(f"⚠ {regla['mensaje']}")
            lbl_mensaje.setWordWrap(True)
            lbl_mensaje.setStyleSheet("color: #EF4444; font-weight: bold;")
            layout.addWidget(lbl_mensaje)

        fila_botones = QHBoxLayout()
        fila_botones.addStretch()
        btn_editar = QPushButton("Editar")
        btn_editar.clicked.connect(lambda: on_editar(regla))
        fila_botones.addWidget(btn_editar)
        btn_eliminar = QPushButton("X")
        btn_eliminar.setFixedWidth(28)
        btn_eliminar.clicked.connect(lambda: on_eliminar(regla))
        fila_botones.addWidget(btn_eliminar)
        layout.addLayout(fila_botones)


def _describir_condicion(regla):
    if regla["tipo"] == "Indicador":
        sujeto = f"Indicador «{regla['objetivo']}»"
    elif regla["tipo"] == "Métrica":
        sujeto = f"{regla['estadistico']} de «{regla['objetivo']}»"
    elif regla["tipo"] == "Columna":
        sujeto = f"Alguna fila de «{regla['objetivo']}»"
    else:
        sujeto = f"Categoría más frecuente de «{regla['objetivo']}»"
    texto = f"{sujeto} {regla['operador']} {regla['umbral']:,.2f}".replace(",", ".")
    if regla["tipo"] == "Columna":
        filtro = describir_filtro(regla)
        texto += f"\nSolo donde {filtro}" if filtro else "\nEn todas las filas"
    return texto


class DialogoAlarma(QDialog):
    """Crear o editar una regla de alarma. Si se entrega `regla`, se abre
    en modo edición pre-llenado con sus valores actuales."""

    def __init__(self, nombres_indicadores, columnas_numericas, columnas_todas,
                 regla=None, parent=None, df=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Alarma" if regla else "Nueva Alarma")
        self.setMinimumWidth(380)
        self._columnas_numericas = columnas_numericas
        self._columnas_todas = columnas_todas
        self._nombres_indicadores = nombres_indicadores
        # `df` solo se usa para listar los valores posibles del filtro
        # ("solo en las filas donde la máquina sea...").
        self._df = df

        # El formulario es largo: va dentro de un área con scroll para que
        # nunca se salga de la pantalla, y los botones Aceptar/Cancelar quedan
        # siempre visibles abajo, fuera del scroll.
        raiz = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._contenido = QWidget()
        layout = QVBoxLayout(self._contenido)
        scroll.setWidget(self._contenido)
        raiz.addWidget(scroll, 1)

        layout.addWidget(QLabel("Nombre (opcional):"))
        self.nombre_edit = QLineEdit(regla["nombre"] if regla else "")
        self.nombre_edit.setPlaceholderText("Se genera solo si lo dejas vacío")
        layout.addWidget(self.nombre_edit)

        layout.addWidget(QLabel("Tipo:"))
        self.tipo_combo = QComboBox()
        self.tipo_combo.addItems(TIPOS_ALARMA)
        self.tipo_combo.currentTextChanged.connect(self._refrescar_objetivo)
        layout.addWidget(self.tipo_combo)

        layout.addWidget(QLabel("Objetivo:"))
        self.objetivo_combo = QComboBox()
        layout.addWidget(self.objetivo_combo)

        self.lbl_estadistico = QLabel("Estadístico:")
        self.estadistico_combo = QComboBox()
        self.estadistico_combo.addItems(list(ESTADISTICOS_METRICA.keys()))
        layout.addWidget(self.lbl_estadistico)
        layout.addWidget(self.estadistico_combo)

        fila_condicion = QHBoxLayout()
        self.operador_combo = QComboBox()
        self.operador_combo.addItems(list(OPERADORES.keys()))
        fila_condicion.addWidget(self.operador_combo)
        self.umbral_edit = QLineEdit(str(regla["umbral"]) if regla else "")
        self.umbral_edit.setPlaceholderText("Número umbral")
        fila_condicion.addWidget(self.umbral_edit)
        layout.addLayout(fila_condicion)

        # Acotar la alarma a ciertas filas (solo para tipo Columna)
        self.lbl_filtro = QLabel("Solo en las filas donde (opcional):")
        layout.addWidget(self.lbl_filtro)
        self.filtro_col_combo = QComboBox()
        self.filtro_col_combo.addItem("(todas las filas)")
        self.filtro_col_combo.addItems(self._columnas_todas)
        self.filtro_col_combo.currentTextChanged.connect(self._refrescar_valores_filtro)
        layout.addWidget(self.filtro_col_combo)
        self.lbl_filtro_valores = QLabel("…sea alguno de estos valores:")
        layout.addWidget(self.lbl_filtro_valores)
        self.filtro_valores_lista = QListWidget()
        self.filtro_valores_lista.setMaximumHeight(130)
        layout.addWidget(self.filtro_valores_lista)
        self.lbl_filtro_nota = QLabel("")
        self.lbl_filtro_nota.setObjectName("muted")
        self.lbl_filtro_nota.setWordWrap(True)
        layout.addWidget(self.lbl_filtro_nota)

        # Qué columna identifica cada fila en el aviso (solo tipo Columna)
        self.lbl_id = QLabel("Columna que identifica cada fila en el aviso (opcional):")
        layout.addWidget(self.lbl_id)
        self.id_col_combo = QComboBox()
        self.id_col_combo.addItem("(solo el número de fila)")
        self.id_col_combo.addItems(self._columnas_todas)
        layout.addWidget(self.id_col_combo)

        layout.addWidget(QLabel("Mensaje de aviso (opcional):"))
        self.mensaje_edit = QLineEdit(regla.get("mensaje", "") if regla else "")
        self.mensaje_edit.setPlaceholderText("Ej: revisar con el proveedor")
        layout.addWidget(self.mensaje_edit)

        self.checkbox_correo = QCheckBox("Avisar también por correo cuando se active")
        self.checkbox_correo.setChecked(bool(regla.get("notificar_correo")) if regla else False)
        layout.addWidget(self.checkbox_correo)

        layout.addWidget(QLabel("Correo destino (opcional -- si lo dejas vacío, usa el predeterminado):"))
        self.correo_destino_edit = QLineEdit(regla.get("correo_destino") or "" if regla else "")
        self.correo_destino_edit.setPlaceholderText("tu-correo@ejemplo.com")
        layout.addWidget(self.correo_destino_edit)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self._validar_y_aceptar)
        botones.rejected.connect(self.reject)
        self._botones = botones
        raiz.addWidget(botones)

        if regla:
            self.tipo_combo.setCurrentText(regla["tipo"])
            self._refrescar_objetivo(regla["tipo"])
            self.objetivo_combo.setCurrentText(regla["objetivo"])
            if regla.get("estadistico"):
                self.estadistico_combo.setCurrentText(regla["estadistico"])
            self.operador_combo.setCurrentText(regla["operador"])
            if regla["tipo"] == "Columna" and regla.get("columna_id") in self._columnas_todas:
                self.id_col_combo.setCurrentText(regla["columna_id"])
            if regla["tipo"] == "Columna" and regla.get("filtro_columna") in self._columnas_todas:
                self.filtro_col_combo.setCurrentText(regla["filtro_columna"])
                self._refrescar_valores_filtro(regla["filtro_columna"], marcados=regla.get("filtro_valores"))
        else:
            self._refrescar_objetivo(self.tipo_combo.currentText())

    def _alto_maximo(self):
        pantalla = self.screen() or QApplication.primaryScreen()
        return int(pantalla.availableGeometry().height() * 0.85) if pantalla else 600

    def _ajustar_tamano(self):
        """Alto justo para el contenido, sin pasarse del alto de la pantalla
        (si no cabe, aparece la barra de desplazamiento)."""
        alto = self._contenido.sizeHint().height() + self._botones.sizeHint().height() + 30
        self.resize(max(self.width(), 400), min(alto, self._alto_maximo()))

    def _mantener_en_pantalla(self):
        pantalla = self.screen() or QApplication.primaryScreen()
        if pantalla is None:
            return
        area = pantalla.availableGeometry()
        marco = self.frameGeometry()
        x = min(max(marco.x(), area.left()), max(area.left(), area.right() - marco.width()))
        y = min(max(marco.y(), area.top()), max(area.top(), area.bottom() - marco.height()))
        if (x, y) != (marco.x(), marco.y()):
            self.move(x, y)

    def showEvent(self, event):
        super().showEvent(event)
        self._ajustar_tamano()
        self._mantener_en_pantalla()

    def _refrescar_objetivo(self, tipo):
        self.objetivo_combo.clear()
        if tipo == "Indicador":
            self.objetivo_combo.addItems(self._nombres_indicadores)
            self.lbl_estadistico.setVisible(False)
            self.estadistico_combo.setVisible(False)
        elif tipo in ("Métrica", "Columna"):
            self.objetivo_combo.addItems(self._columnas_numericas)
            self.lbl_estadistico.setVisible(tipo == "Métrica")
            self.estadistico_combo.setVisible(tipo == "Métrica")
        else:  # Frecuencia
            self.objetivo_combo.addItems(self._columnas_todas)
            self.lbl_estadistico.setVisible(False)
            self.estadistico_combo.setVisible(False)
        es_columna = tipo == "Columna"
        for w in (self.lbl_filtro, self.filtro_col_combo, self.lbl_id, self.id_col_combo):
            w.setVisible(es_columna)
        self._refrescar_valores_filtro(self.filtro_col_combo.currentText() if es_columna else "")
        if self.isVisible():
            self._ajustar_tamano()
            self._mantener_en_pantalla()

    def _refrescar_valores_filtro(self, columna, marcados=None):
        """Llena la lista de valores del filtro con los de la columna
        elegida (los más frecuentes primero). `marcados` = valores que
        deben aparecer ya tildados (al editar una alarma)."""
        self.filtro_valores_lista.clear()
        self.lbl_filtro_nota.setText("")
        es_columna = self.tipo_combo.currentText() == "Columna"
        hay_filtro = es_columna and columna and columna != "(todas las filas)"
        self.lbl_filtro_valores.setVisible(bool(hay_filtro))
        self.filtro_valores_lista.setVisible(bool(hay_filtro))
        self.lbl_filtro_nota.setVisible(bool(hay_filtro))
        if self.isVisible():
            self._ajustar_tamano()
            self._mantener_en_pantalla()
        if not hay_filtro:
            return
        if self._df is None or columna not in self._df.columns:
            self.lbl_filtro_nota.setText("No se pudieron leer los valores de esa columna.")
            return
        conteo = self._df[columna].astype(str).value_counts()
        total_distintos = len(conteo)
        marcados = set(str(v) for v in (marcados or []))
        for valor in conteo.index[:MAX_VALORES_FILTRO]:
            item = QListWidgetItem(str(valor))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if valor in marcados else Qt.Unchecked)
            self.filtro_valores_lista.addItem(item)
        if total_distintos > MAX_VALORES_FILTRO:
            self.lbl_filtro_nota.setText(
                f"Se muestran los {MAX_VALORES_FILTRO} valores más frecuentes (hay {total_distintos})."
            )

    def _valores_filtro_marcados(self):
        if not self.filtro_valores_lista.isVisible():
            return []
        return [
            self.filtro_valores_lista.item(i).text()
            for i in range(self.filtro_valores_lista.count())
            if self.filtro_valores_lista.item(i).checkState() == Qt.Checked
        ]

    def _validar_y_aceptar(self):
        if not self.objetivo_combo.currentText():
            _avisar_sin_objetivo(self)
            return
        try:
            float(self.umbral_edit.text().replace(",", "."))
        except ValueError:
            self.umbral_edit.setStyleSheet("border: 1px solid #EF4444;")
            return
        if (self.tipo_combo.currentText() == "Columna"
                and self.filtro_col_combo.currentText() != "(todas las filas)"
                and not self._valores_filtro_marcados()):
            QMessageBox.information(
                self, "Elige al menos un valor",
                "Marca de la lista los valores a los que se aplica la alarma, "
                "o elige «(todas las filas)» para no acotarla.",
            )
            return
        self.accept()

    def get_datos(self):
        tipo = self.tipo_combo.currentText()
        objetivo = self.objetivo_combo.currentText()
        umbral = float(self.umbral_edit.text().replace(",", "."))
        filtro_columna, filtro_valores = None, None
        if tipo == "Columna" and self.filtro_col_combo.currentText() != "(todas las filas)":
            filtro_columna = self.filtro_col_combo.currentText()
            filtro_valores = self._valores_filtro_marcados() or None
            if filtro_valores is None:
                filtro_columna = None
        nombre = self.nombre_edit.text().strip()
        if not nombre:
            nombre = f"{tipo}: {objetivo}"
            if filtro_columna:
                nombre += " · " + describir_filtro(
                    {"filtro_columna": filtro_columna, "filtro_valores": filtro_valores})
        return {
            "nombre": nombre,
            "tipo": tipo,
            "objetivo": objetivo,
            "estadistico": self.estadistico_combo.currentText() if tipo == "Métrica" else None,
            "operador": self.operador_combo.currentText(),
            "umbral": umbral,
            "mensaje": self.mensaje_edit.text().strip(),
            "notificar_correo": 1 if self.checkbox_correo.isChecked() else 0,
            "correo_destino": self.correo_destino_edit.text().strip() or None,
            "filtro_columna": filtro_columna,
            "filtro_valores": filtro_valores,
            "columna_id": (
                self.id_col_combo.currentText()
                if tipo == "Columna" and self.id_col_combo.currentIndex() > 0 else None
            ),
        }


def _avisar_sin_objetivo(parent):
    """Aviso corto cuando no hay ningún objetivo disponible para elegir
    (ej. no hay indicadores creados todavía)."""
    QMessageBox.information(
        parent, "Elige un objetivo",
        "No hay nada para elegir en 'Objetivo' -- si es un Indicador, créalo "
        "primero en la pestaña Indicadores.",
    )