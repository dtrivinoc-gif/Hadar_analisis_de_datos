"""
prediccion.py — Cálculo de la pestaña Predicción (sin Qt, igual que la
parte de arriba de panel.py: se puede probar sin abrir la ventana).

Idea central: UN MODELO POR PROYECTO. El modelo se entrena con una tabla de
ESTE proyecto y solo sirve para ese dataset; si abres otro proyecto, parte
desde cero (mismo criterio de aprendizaje_adaptativo.py: nada se comparte
entre proyectos).

Es generalista a propósito: no asume ningún dominio. Funciona con cualquier
tabla (ventas, Kaggle, lo que sea) siguiendo siempre los mismos pasos:

  1. Tú eliges qué columna predecir y en qué columnas debe fijarse el modelo
     (Hadar propone una selección automática y tú la corriges a mano --
     así no predice cosas inútiles como el siguiente id).
  2. Se prueba el modelo contra partes de los datos que NO vio al entrenar
     (validación cruzada: varias pruebas en vez de una sola, así se ve cuánto
     varía el resultado), y se compara contra dos referencias simples: una
     "respuesta tonta" (el promedio, o la clase más frecuente) y un modelo
     lineal básico. Así se nota si de verdad aporta algo.
  3. Se reentrena con TODOS los datos y se predice sobre una tabla de prueba.

Modelo: HistGradientBoosting de scikit-learn (la misma librería del
Isolation Forest, así que no hay nada nuevo que instalar). Es el mismo tipo
de modelo (árboles con boosting) que domina las competencias tabulares de
Kaggle, tolera datos faltantes y es rápido con millones de filas.

Decisiones de diseño:
- El modelo entrenado NO se guarda dentro del .hadarproy: ese archivo es un
  .zip común y los modelos de Python se guardan con "pickle", que puede
  ejecutar código al abrirlo. Se guarda solo la CONFIGURACIÓN (qué predecir,
  qué columnas usar) y se reentrena con un clic.
- Cada columna se convierte en números de forma automática: números tal
  cual; fechas -> año, mes, día, día de la semana...; categorías -> código
  (o frecuencia si hay demasiadas); todo lo vacío queda como "faltante".
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

try:
    from .avisos import ordenar_avisos, ETIQUETA as _ETIQUETA_AVISO
    from .formato import dec
except ImportError:  # pruebas sueltas fuera del paquete
    from avisos import ordenar_avisos, ETIQUETA as _ETIQUETA_AVISO
    from formato import dec

MAX_CATEGORIAS = 250          # HistGradientBoosting admite categorías nativas hasta ~255
MAX_CLASES = 100
MUESTRA_DIAGNOSTICO = 100_000
UMBRAL_CASI_UNICA = 0.95
UMBRAL_VACIA = 0.98
UMBRAL_FUGA = 0.999
FRACCION_VALIDACION = 0.2
MINIMO_FILAS = 30
MAX_FRECUENCIAS = 50_000
UMBRAL_DUPLICADAS = 0.02    # avisar si al menos el 2% de las filas están repetidas
REVISION_MAX = 50              # filas que se guardan para la lista de revisión
PUNTOS_GRAFICO = 1500           # puntos del gráfico «predicho vs real»
PLIEGUES_POR_DEFECTO = 5        # cuántas pruebas distintas se hacen (validación cruzada)
CV_MAX_FILAS = 150_000          # sobre este tamaño se hace una sola prueba (más rápido)
UMBRAL_DESBALANCE = 0.10        # una clase bajo este % de las filas recibe más peso
MINIMO_VALIDACION = 100         # bajo esto, el resultado se avisa como poco confiable
MAX_FILAS_LINEAL = 50_000       # filas con que se entrena el modelo lineal de referencia

_TOKENS_ID = {"id", "ids", "uuid", "guid", "index", "idx", "indice", "rut", "folio"}

TIPOS_LEGIBLES = {
    "numerica": "número",
    "booleana": "sí/no",
    "fecha": "fecha",
    "fecha_texto": "fecha (en texto)",
    "numerica_texto": "número (en texto)",
    "categorica": "categoría",
    "otra": "otra",
}


# ----------------------------------------------------------------------
# Configuración (lo único que se guarda en el proyecto)
# ----------------------------------------------------------------------
@dataclass
class ConfigPrediccion:
    tabla_entrenamiento: str | None = None
    tabla_prueba: str | None = None
    objetivo: str | None = None
    tipo: str = "auto"                 # "auto" | "clasificacion" | "regresion"
    cambios_columnas: dict = field(default_factory=dict)   # {columna: "usar"|"ignorar"} solo lo que el usuario cambió
    log_objetivo: bool = False
    validacion: str = "aleatoria"      # "aleatoria" | "temporal" | "grupos"
    columna_fecha: str | None = None
    columna_grupo: str | None = None   # solo para validacion = "grupos"
    pliegues: int = PLIEGUES_POR_DEFECTO   # cuántas pruebas distintas (1 = una sola)
    balancear: str = "auto"            # "auto" | "si" | "no": más peso a la clase poco frecuente
    salida: str = "clase"              # "clase" | "probabilidad"
    columna_id_salida: str | None = None   # None = automática
    max_iter: int = 200
    tasa_aprendizaje: float = 0.1
    profundidad_max: int = 0           # 0 = automática
    max_filas: int = 500_000
    semilla: int = 0
    # --- tablas relacionadas (ver relacional.py) ---
    usar_relaciones: bool = False
    profundidad_relaciones: int = 2     # saltos máximos entre tablas
    enlaces_manuales: list = field(default_factory=list)      # relaciones que el usuario agregó a mano
    enlaces_desactivados: list = field(default_factory=list)  # relaciones que el usuario desmarcó
    fecha_relaciones: str | None = None  # None = automática; "__ninguna__" = sin restringir por fecha

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d) -> "ConfigPrediccion":
        d = d or {}
        nombres = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in nombres})


@dataclass
class SugerenciaColumna:
    nombre: str
    tipo: str            # clave de TIPOS_LEGIBLES
    usar: bool
    motivo: str


# ----------------------------------------------------------------------
# Detección de tipos y sugerencia de columnas
# ----------------------------------------------------------------------
def _tokens(nombre) -> list:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(nombre))
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def es_columna_id(nombre) -> bool:
    """Compara por palabra completa ('customer_id', 'PassengerId' sí;
    'paid' o 'valid' no)."""
    if str(nombre).lower().startswith("unnamed"):
        return True
    return any(t in _TOKENS_ID for t in _tokens(nombre))


def normalizar_clave(s: pd.Series) -> pd.Series:
    """Deja una columna clave como texto comparable: 1, 1.0 y '1' pasan a ser
    '1' (así un código numérico en una tabla y en otra siempre coincide)."""
    if pd.api.types.is_float_dtype(s):
        v = s.dropna()
        if len(v) and bool((v % 1 == 0).all()):
            s = s.astype("Int64")
    return s.astype("string").str.strip()


def _es_texto(s: pd.Series) -> bool:
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype))


_RE_ISO = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
_RE_DMA = re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})")


def detectar_dayfirst(serie: pd.Series) -> bool:
    """¿Las fechas en texto vienen como DD/MM/AAAA (Chile y casi todo el mundo)
    o como MM/DD/AAAA (EE.UU.)? Se mira la columna completa: si algún primer
    número pasa de 12, es el día; si algún segundo número pasa de 12, el día va
    segundo. Si todas las fechas son ambiguas ("03/04/2024") se asume DD/MM,
    igual que el resto de Hadar. Las fechas que empiezan por el año
    (2024-03-04) no dependen de esto: se devuelve False para no confundirlas."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return False
    m = serie.dropna()
    if m.empty:
        return True
    if len(m) > 20000:
        m = m.sample(20000, random_state=0)
    texto = m.astype(str)
    if texto.str.match(_RE_ISO).mean() >= 0.5:
        return False
    partes = texto.str.extract(_RE_DMA).dropna()
    if partes.empty:
        return True
    a = pd.to_numeric(partes[0], errors="coerce")
    b = pd.to_numeric(partes[1], errors="coerce")
    if bool((b > 12).any()) and not bool((a > 12).any()):
        return False
    return True


def _a_datetime(serie: pd.Series, dayfirst: bool | None = None) -> pd.Series:
    """Convierte a fecha. `dayfirst=None` lo detecta mirando la columna; al
    entrenar se guarda lo detectado y se reutiliza al predecir."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        dt = serie
    else:
        if dayfirst is None:
            dayfirst = detectar_dayfirst(serie)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dt = pd.to_datetime(serie, errors="coerce", format="mixed", dayfirst=dayfirst)
            except (ValueError, TypeError):
                dt = pd.to_datetime(serie, errors="coerce", dayfirst=dayfirst)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)
    return dt


def _parece_numerica(m: pd.Series) -> bool:
    conv = pd.to_numeric(m.astype(str), errors="coerce")
    return bool(conv.notna().mean() >= 0.95)


def _parece_fecha(m: pd.Series) -> bool:
    muestra = m.astype(str)
    if len(muestra) > 300:
        muestra = muestra.sample(300, random_state=0)
    largo = muestra.str.len().mean()
    if not (6 <= largo <= 35):
        return False
    if muestra.str.contains(r"\d", regex=True).mean() < 0.95:
        return False
    conv = _a_datetime(muestra)
    return bool(conv.notna().mean() >= 0.95)


def tipo_columna(serie: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(serie):
        return "booleana"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "fecha"
    if pd.api.types.is_timedelta64_dtype(serie) or pd.api.types.is_numeric_dtype(serie):
        return "numerica"
    if _es_texto(serie):
        m = serie.dropna()
        if m.empty:
            return "categorica"
        if len(m) > 2000:
            m = m.iloc[:2000]
        if _parece_numerica(m):
            return "numerica_texto"
        if _parece_fecha(m):
            return "fecha_texto"
        return "categorica"
    return "otra"


def _es_entera(serie: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(serie)


def inferir_tipo_problema(serie: pd.Series, manual: str = "auto") -> str:
    """'clasificacion' (predecir una categoría) o 'regresion' (predecir un
    número). En 'auto': texto/booleanos/pocas categorías -> clasificación;
    números con muchos valores distintos -> regresión."""
    if manual in ("clasificacion", "regresion"):
        return manual
    t = tipo_columna(serie)
    if t in ("booleana", "categorica", "fecha", "fecha_texto", "otra"):
        return "clasificacion"
    s = pd.to_numeric(serie.dropna(), errors="coerce").dropna()
    if s.empty:
        return "clasificacion"
    nun = s.nunique()
    if nun <= 2:
        return "clasificacion"
    if nun <= 20:
        muestra = s.iloc[:5000]
        if np.allclose(muestra.to_numpy(float), np.round(muestra.to_numpy(float))):
            return "clasificacion"
    return "regresion"


def pct_txt_(x) -> str:
    """0.153 -> '15%' (en el formato de números elegido)."""
    return f"{dec(float(x) * 100, 0)}%" if x is not None and np.isfinite(x) else "—"


def columnas_fecha(df: pd.DataFrame, excluir=()) -> list:
    """Columnas de la tabla que parecen fechas (mira solo una muestra, así es barato)."""
    salida = []
    for c in df.columns:
        if c in excluir or es_columna_id(c):
            continue
        try:
            if tipo_columna(df[c].head(2000)) in ("fecha", "fecha_texto"):
                salida.append(str(c))
        except Exception:
            continue
    return salida


def sugerir_columnas(df: pd.DataFrame, objetivo: str | None = None,
                     df_prueba: pd.DataFrame | None = None) -> list:
    """Propone qué columnas usar. Es solo una sugerencia: el usuario puede
    cambiar cada una a mano. Devuelve una SugerenciaColumna por columna
    (sin incluir la columna objetivo)."""
    n = len(df)
    muestra = df if n <= MUESTRA_DIAGNOSTICO else df.sample(MUESTRA_DIAGNOSTICO, random_state=0)
    ym = muestra[objetivo] if objetivo in muestra.columns else None
    ym_num = None
    if ym is not None and (pd.api.types.is_numeric_dtype(ym) and not pd.api.types.is_bool_dtype(ym)):
        ym_num = ym

    resultado = []
    for col in df.columns:
        if col == objetivo:
            continue
        s = muestra[col]
        tipo = tipo_columna(s)
        usar, motivo = True, ""
        nn = int(s.notna().sum())

        if df_prueba is not None and col not in df_prueba.columns:
            usar, motivo = False, "No existe en la tabla de prueba"
        elif es_columna_id(col):
            usar, motivo = False, "Parece un identificador (no sirve para predecir)"
        elif nn == 0 or s.isna().mean() >= UMBRAL_VACIA:
            usar, motivo = False, "Casi toda la columna está vacía"
        else:
            nun = int(s.nunique(dropna=True))
            if nun <= 1:
                usar, motivo = False, "Todos los valores son iguales"
            elif tipo in ("categorica", "numerica_texto") and nn >= MINIMO_FILAS \
                    and nun / nn >= UMBRAL_CASI_UNICA:
                usar, motivo = False, "Casi todos los valores son distintos (parece un identificador o texto libre)"
            elif tipo == "numerica" and _es_entera(s) and nn >= MINIMO_FILAS and nun == nn:
                rango = float(s.max() - s.min()) + 1
                if rango <= nn * 1.05:
                    usar, motivo = False, "Es una secuencia 1, 2, 3… (parece un identificador)"
            if usar and tipo in ("categorica",) and nn >= MINIMO_FILAS and nun / nn > 0.5:
                usar, motivo = False, "Demasiados valores distintos para ser una categoría"

        if usar and ym_num is not None and tipo == "numerica":
            try:
                a = pd.to_numeric(s, errors="coerce")
                ok = a.notna() & ym_num.notna()
                if ok.sum() >= MINIMO_FILAS and a[ok].nunique() > 1 and ym_num[ok].nunique() > 1:
                    r = np.corrcoef(a[ok].to_numpy(float), ym_num[ok].to_numpy(float))[0, 1]
                    if np.isfinite(r) and abs(r) >= UMBRAL_FUGA:
                        usar, motivo = False, "Es casi idéntica a la columna a predecir (posible trampa)"
            except Exception:
                pass

        resultado.append(SugerenciaColumna(str(col), tipo, usar, motivo))
    return resultado


def columnas_a_usar(sugerencias: list, cambios: dict | None) -> list:
    """Aplica los cambios manuales del usuario sobre las sugerencias."""
    cambios = cambios or {}
    usadas = []
    for sug in sugerencias:
        c = cambios.get(sug.nombre)
        usar = sug.usar if c is None else (c == "usar")
        if usar:
            usadas.append(sug.nombre)
    return usadas


def elegir_columna_id(df: pd.DataFrame, preferida: str | None = None):
    """La columna que identifica cada fila en el archivo de entrega."""
    if preferida and preferida in df.columns:
        return preferida
    for c in df.columns:
        if es_columna_id(c):
            return c
    return None


# ----------------------------------------------------------------------
# Convertir columnas a números
# ----------------------------------------------------------------------
def _a_float32(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_timedelta64_dtype(s):
        arr = s.dt.total_seconds().to_numpy(dtype="float32", na_value=np.nan)
    else:
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float32", na_value=np.nan)
    arr[~np.isfinite(arr)] = np.nan
    return arr


class Preprocesador:
    """Aprende cómo convertir cada columna a números (con los datos de
    entrenamiento) y aplica exactamente lo mismo a datos nuevos."""

    def __init__(self, columnas):
        self.columnas = list(columnas)
        self.plan = {}
        self.nombres = []
        self.fuente = []
        self.mascara_categ = []

    def ajustar(self, df: pd.DataFrame, forzar_categoria=()) -> "Preprocesador":
        forzar = set(forzar_categoria or ())
        for col in self.columnas:
            s = df[col]
            tipo = "categorica" if col in forzar else tipo_columna(s)
            if tipo in ("numerica", "booleana"):
                self.plan[col] = {"tipo": "num"}
            elif tipo == "numerica_texto":
                self.plan[col] = {"tipo": "num"}
            elif tipo in ("fecha", "fecha_texto"):
                dia_primero = detectar_dayfirst(s)
                dt = _a_datetime(s, dia_primero)
                hora = bool((dt.dt.hour.fillna(0) != 0).any())
                self.plan[col] = {"tipo": "fecha", "hora": hora, "dayfirst": dia_primero}
            else:
                norm = col in forzar
                texto = (normalizar_clave(s) if norm else s).dropna().astype(str)
                cats = sorted(texto.unique())
                if len(cats) <= MAX_CATEGORIAS:
                    self.plan[col] = {"tipo": "cat", "categorias": cats, "norm": norm}
                else:
                    frec = texto.value_counts(normalize=True).head(MAX_FRECUENCIAS)
                    self.plan[col] = {"tipo": "freq", "frecuencias": frec.to_dict(), "norm": norm}
        self._construir_nombres()
        return self

    def _construir_nombres(self):
        self.nombres, self.fuente, self.mascara_categ = [], [], []
        for col in self.columnas:
            p = self.plan[col]
            if p["tipo"] == "fecha":
                partes = ["año", "mes", "día", "día de la semana", "día del año", "fecha como número"]
                if p["hora"]:
                    partes.append("hora")
                for parte in partes:
                    self.nombres.append(f"{col} ({parte})")
                    self.fuente.append(col)
                    self.mascara_categ.append(False)
            else:
                self.nombres.append(col)
                self.fuente.append(col)
                self.mascara_categ.append(p["tipo"] == "cat")

    def transformar(self, df: pd.DataFrame) -> np.ndarray:
        self.desconocidas = {}          # columna -> (filas con un valor que no se vio al entrenar, un ejemplo)
        columnas = []
        for col in self.columnas:
            p = self.plan[col]
            s = df[col]
            if p["tipo"] == "num":
                columnas.append(_a_float32(s))
            elif p["tipo"] == "fecha":
                dt = _a_datetime(s, p.get("dayfirst"))
                def f(x):
                    return x.to_numpy(dtype="float32", na_value=np.nan)
                columnas.extend([
                    f(dt.dt.year), f(dt.dt.month), f(dt.dt.day), f(dt.dt.dayofweek),
                    f(dt.dt.dayofyear),
                    f((dt - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0),
                ])
                if p["hora"]:
                    columnas.append(f(dt.dt.hour))
            else:
                if p.get("norm"):
                    s = normalizar_clave(s)
                faltante = s.isna().to_numpy()
                texto = np.where(faltante, None, s.astype(object).astype(str).to_numpy())
                if p["tipo"] == "cat":
                    # Categorías que no se vieron al entrenar pasan a «vacío» (NaN). Se hace con get_indexer: construir
                    # un Categorical con valores fuera de las categorías está deprecado y fallará en pandas 4.
                    codigos = pd.Index(p["categorias"]).get_indexer(texto).astype("float32")
                    nuevas = (codigos < 0) & ~faltante
                    codigos[codigos < 0] = np.nan
                    columnas.append(codigos)
                else:
                    frec = p["frecuencias"]
                    columnas.append(np.array(
                        [frec.get(t, 0.0) if t is not None else np.nan for t in texto],
                        dtype="float32"))
                    nuevas = np.array([t is not None and t not in frec for t in texto], dtype=bool)
                if nuevas.any():
                    self.desconocidas[col] = (int(nuevas.sum()), str(texto[np.argmax(nuevas)]))
        if not columnas:
            return np.empty((len(df), 0), dtype="float32")
        return np.column_stack(columnas).astype("float32", copy=False)


# ----------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------
def _fmt(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    a = abs(x)
    if a >= 1000:
        return f"{dec(x, 0, True)}"
    if a >= 1:
        return f"{dec(x, 2, True)}"
    return f"{dec(x, 4)}"


def _mejora_de(tipo, m, b) -> float:
    """Qué fracción del error de la «respuesta fácil» evita el modelo (0 a 1).
    En clasificación se usa la exactitud PAREJA entre clases (balanced accuracy),
    para que un modelo no parezca inútil solo porque una clase es muy frecuente."""
    if tipo == "regresion":
        return float(np.clip(1 - m["rmse"] / b["rmse"], 0, 1)) if b.get("rmse") else 0.0
    bal, base = m.get("bal_acc"), b.get("bal_acc")
    if bal is None or base is None or base >= 1:
        return 0.0
    return float(np.clip((bal - base) / (1 - base), 0, 1))


@dataclass
class ResultadoEntrenamiento:
    tipo: str
    objetivo: str
    columnas: list
    clases: list | None
    metricas: dict
    baseline: dict
    importancias: list            # [(columna, porcentaje)] de más a menos importante
    n_entrenamiento: int
    n_validacion: int
    n_iteraciones: int
    advertencias: list
    log_objetivo: bool
    salida: str
    modelo: object = field(default=None, repr=False)
    preprocesador: object = field(default=None, repr=False)
    validacion: tuple | None = field(default=None, repr=False)   # (reales, predichos) de lo que no vio
    revision: dict | None = field(default=None, repr=False)      # filas que conviene mirar primero
    metricas_sd: dict = field(default_factory=dict)              # variación entre las pruebas
    mejoras_pliegue: list = field(default_factory=list)
    n_pliegues: int = 1
    modo_validacion: str = "aleatoria"
    balanceado: bool = False       # se dio más peso a la clase poco frecuente
    clase_minoritaria: object = None   # solo en problemas de dos clases
    tipo_salida: str = ""              # valor | probabilidad | ranking (qué significa el número que se entrega)
    calibracion: dict | None = field(default=None, repr=False)   # medición de calibración (dos clases)
    calibrador: object = field(default=None, repr=False)         # corrección en memoria, o None
    semilla: int | None = None         # semilla fija con la que se entrenó (mismo dato + misma semilla = mismo resultado)

    # ---- lectura simple del resultado (para la pantalla) --------------
    def mejora(self) -> float:
        return _mejora_de(self.tipo, self.metricas, self.baseline)

    def mejora_sd(self) -> float:
        return float(np.std(self.mejoras_pliegue)) if len(self.mejoras_pliegue) > 1 else 0.0

    def desbalanceado(self) -> bool:
        return self.tipo == "clasificacion" and self.baseline.get("accuracy", 0) >= 0.85

    def _caso_raro(self) -> bool:
        """Dos clases, una muy poco frecuente: ahí detectar importa más que acertar."""
        m = self.metricas
        return bool(self.desbalanceado() and self.clase_minoritaria is not None
                    and m.get("recall_min") is not None)

    def veredicto(self):
        """(nivel, titulo, frase) con palabras simples. Niveles: muy_bueno, bueno,
        regular, no_aporta, sospechoso."""
        m, b = self.metricas, self.baseline
        mej = self.mejora()
        top = self.importancias[0] if self.importancias else None
        casi_perfecto = (m.get("r2", 0) >= 0.99) if self.tipo == "regresion" else \
            (m.get("bal_acc", 0) >= 0.99 and m.get("accuracy", 0) >= 0.99)
        domina = top is not None and top[1] >= 85 and mej >= 0.8
        if casi_perfecto or domina:
            if top is not None and top[1] >= 60:
                frase = (f"Una sola columna («{top[0]}») concentra el {dec(top[1], 0)}% de las decisiones. Suele pasar "
                         f"cuando esa columna es casi lo mismo que lo que predices (por ejemplo, un total calculado "
                         f"con los mismos datos). Desmárcala en el Paso 2 y vuelve a entrenar para ver el resultado real.")
            else:
                frase = ("Un resultado casi perfecto rara vez es real. Revisa si alguna columna ya contiene la "
                         "respuesta (un total, una copia, algo que solo se sabe después de que ocurre).")
            return "sospechoso", "Revisa: resultado sospechosamente bueno", frase
        if self.tipo == "regresion":
            detalle = f"Reduce el error en {dec(mej * 100, 0)}% frente a responder siempre el promedio."
        elif self._caso_raro():
            detalle = (f"Detecta el {dec(m['recall_min'] * 100, 0)}% de los casos «{self.clase_minoritaria}» y, cuando "
                       f"los marca, acierta el {dec(m['precision_min'] * 100, 0)}% de las veces. Como son pocos casos, "
                       f"acertar en general no sirve de medida: responder siempre lo más frecuente acertaría el "
                       f"{dec(b['accuracy'] * 100, 0)}%.")
        else:
            detalle = (f"Acierta el {dec(m['accuracy'] * 100, 0)}% de las veces; responder siempre la clase más "
                       f"frecuente acertaría el {dec(b['accuracy'] * 100, 0)}%.")
        if self.n_pliegues > 1 and self.mejora_sd() >= 0.15:
            detalle += " El resultado cambia bastante de una prueba a otra: tómalo como aproximado."
        if mej >= 0.8:
            return "muy_bueno", "Muy bueno", detalle
        if mej >= 0.5:
            return "bueno", "Bueno", detalle
        if mej >= 0.2:
            return "regular", "Regular", detalle + " Hay margen: prueba con otras columnas."
        return "no_aporta", "No aporta", detalle + " Con estas columnas el modelo casi no sirve."

    def _pm(self, clave, escala=100.0, decimales=0) -> str:
        """' ± 4' entre pruebas, solo si hubo varias pruebas."""
        if self.n_pliegues <= 1:
            return ""
        sd = self.metricas_sd.get(clave, 0.0) * escala
        return f" (± {sd:.{decimales}f})" if sd >= 0.5 * 10 ** -decimales else ""

    def texto_salida(self) -> str:
        """Qué significa el número que entrega el modelo, en palabras simples."""
        if self.tipo_salida == "valor":
            return f"El número es un valor pronosticado de «{self.objetivo}» (una estimación, no una probabilidad)."
        c = self.calibracion or {}
        if self.tipo_salida == "probabilidad":
            base = ("Es una probabilidad: en datos que no vio, de los casos con un X% estimado ocurrió aproximadamente "
                    "ese X% de las veces")
            if c.get("calibrada") and c.get("metodo"):
                base += (f" (Hadar corrigió el número con lo observado: método {'isotónico' if c['metodo'] == 'isotonica' else 'sigmoide'})")
            return base + "."
        if self.tipo_salida == "ranking":
            razon = c.get("motivo") or ("se dio más peso a la clase poco frecuente, lo que distorsiona el número"
                                        if self.balanceado else "no se pudo comprobar que coincida con lo que ocurre")
            return ("El número es un puntaje para ORDENAR (de más a menos probable), no un porcentaje real: " + razon + ".")
        return ""

    def tarjetas(self) -> list:
        """Tres cifras clave, cada una con su explicación en palabras: [(valor, titulo, explicacion)]."""
        m, b = self.metricas, self.baseline
        if self.tipo == "regresion":
            return [
                (f"{dec(self.mejora() * 100, 0)}%", "Menos error que adivinar",
                 "frente a responder siempre el promedio"
                 + (f" (± {dec(self.mejora_sd() * 100, 0)} entre pruebas)" if self.n_pliegues > 1 else "")),
                (_fmt(m["mae"]), "Error promedio", f"cuánto se equivoca en promedio, en las unidades de «{self.objetivo}»"),
                (f"{dec(max(m['r2'], 0) * 100, 0)}%", "Variación explicada", "qué parte de las diferencias entre filas logra explicar (R²)"),
            ]
        tercera = ((dec(m['auc'], 2), "Separa las clases (AUC)", f"{dec(0.5, 1)} es azar y 1 es perfecto")
                   if m.get("auc") is not None else
                   (f"{dec(m['f1'] * 100, 0)}%", "Equilibrio entre clases", "qué tan parejo acierta en todas (F1)"))
        if self._caso_raro():
            return [
                (f"{dec(m['recall_min'] * 100, 0)}%", "Casos detectados",
                 f"de los casos «{self.clase_minoritaria}» que había, en datos que no vio al entrenar"),
                (f"{dec(m['precision_min'] * 100, 0)}%", "Alarmas correctas",
                 "cuando marca un caso, acierta así de seguido"),
                tercera,
            ]
        return [
            (f"{dec(m['accuracy'] * 100, 0)}%", "Aciertos", "en datos que no vio al entrenar"
             + (f" (± {dec(self.metricas_sd.get('accuracy', 0) * 100, 0)} entre pruebas)" if self.n_pliegues > 1 else "")),
            (f"{dec(b['accuracy'] * 100, 0)}%", "Respuesta fácil", "acertando siempre la clase más frecuente"),
            tercera,
        ]

    def detalle_experto(self) -> dict:
        """Detalles técnicos para el modo experto: cómo se probó, con cuánto, cuánto se equivocó y contra qué se comparó."""
        m, b, sd = self.metricas, self.baseline, self.metricas_sd or {}
        modos = {"aleatoria": "filas mezcladas al azar", "temporal": "por fecha (se prueba con lo más nuevo)",
                 "grupos": "por grupos (un grupo nunca queda en entrenamiento y prueba a la vez)"}
        filas = [("Tipo de problema", "regresión (un número)" if self.tipo == "regresion" else "clasificación (una categoría)"),
                 ("Cómo se probó", modos.get(self.modo_validacion, self.modo_validacion)),
                 ("Pruebas (pliegues)", dec(self.n_pliegues, 0)),
                 ("Filas de entrenamiento", dec(self.n_entrenamiento, 0, True)),
                 ("Filas de prueba (en total)", dec(self.n_validacion, 0, True)),
                 ("Iteraciones del modelo", dec(self.n_iteraciones, 0)),
                 ("Semilla", "—" if self.semilla is None else str(self.semilla)),
                 ("Mejora sobre la respuesta fácil", f"{pct_txt_(self.mejora())} (± {pct_txt_(self.mejora_sd())} entre pruebas)")]
        metodos = []
        if self.tipo == "regresion":
            metodos = [("Este modelo (RMSE)", m.get("rmse")), ("Responder siempre el promedio (RMSE)", b.get("rmse")),
                       ("Modelo lineal simple (RMSE)", b.get("lineal_rmse"))]
            titulo_m = "Error típico en datos que no vio (menor es mejor)"
        else:
            e = lambda v: None if v is None else 1.0 - float(v)
            metodos = [("Este modelo (error pareja entre clases)", e(m.get("bal_acc"))),
                       ("Responder siempre la clase más frecuente", e(b.get("bal_acc"))),
                       ("Modelo lineal simple", e(b.get("lineal_bal_acc")))]
            titulo_m = "Error en datos que no vio, parejo entre clases (menor es mejor)"
        metodos = [(k, float(v)) for k, v in metodos if v is not None and np.isfinite(v)]
        c = self.calibracion or {}
        if c.get("verificable"):
            filas += [("Calibración: error por tramos", f"{dec(c['ece'], 3)} (tolerancia {dec(c.get('tolerancia', 0), 3)})"),
                      ("Brier (menor es mejor)", f"{dec(c['brier'], 4)} frente a {dec(c['brier_base'], 4)} de decir siempre la tasa base"),
                      ("Corrección de probabilidades", {"isotonica": "isotónica", "platt": "sigmoide (Platt)"}.get(c.get("metodo"), "no hizo falta" if c.get("calibrada") else "no se aplicó"))]
        elif c.get("motivo"):
            filas.append(("Calibración", "no comprobada: " + c["motivo"]))
        if self.tipo_salida:
            filas.append(("El número que se entrega es", {"valor": "un valor pronosticado", "probabilidad": "una probabilidad",
                                                           "ranking": "un puntaje para ordenar"}.get(self.tipo_salida, self.tipo_salida)))
        residuos = None
        if self.tipo == "regresion" and self.validacion is not None:
            try:
                reales, pred = (np.asarray(v, dtype="float64") for v in self.validacion)
                r = (pred - reales)[np.isfinite(pred - reales)]
                if len(r) > 400:
                    r = r[np.linspace(0, len(r) - 1, 400).astype(int)]
                residuos = [float(x) for x in r]
            except Exception:
                residuos = None
        return {"titulo": "Detalles técnicos", "filas": filas, "metodos": metodos, "metodos_titulo": titulo_m,
                "residuos": residuos, "residuos_titulo": "Cuánto se equivocó el modelo en datos que no vio (el centro es acertar)"
                if residuos else ""}

    def texto_validacion(self) -> str:
        modos = {"aleatoria": "con las filas mezcladas al azar", "temporal": "respetando el orden de las fechas "
                 "(siempre se prueba con filas más nuevas que las de entrenamiento)",
                 "grupos": "sin repartir un mismo grupo entre entrenamiento y prueba"}
        modo = modos.get(self.modo_validacion, "")
        if self.n_pliegues > 1:
            return (f"Se probó {self.n_pliegues} veces {modo}: en cada una el modelo entrenó con unas "
                    f"{dec(self.n_entrenamiento, 0, True)} filas y se evaluó con otras que no había visto "
                    f"({dec(self.n_validacion, 0, True)} en total). Las cifras son el promedio de las pruebas; "
                    f"luego el modelo se reentrenó con todas las filas.")
        return (f"Se entrenó con {dec(self.n_entrenamiento, 0, True)} filas y se probó con {dec(self.n_validacion, 0, True)} "
                f"que el modelo no había visto ({modo}); luego se reentrenó con todas.")

    def texto_lineal(self) -> str | None:
        """Comparación con un modelo lineal básico (otra referencia simple)."""
        m, b = self.metricas, self.baseline
        if self.tipo == "regresion" and b.get("lineal_rmse"):
            return (f"Referencia: un modelo lineal simple tendría un error típico de {_fmt(b['lineal_rmse'])} "
                    f"(este modelo: {_fmt(m['rmse'])}).")
        if self.tipo == "clasificacion" and b.get("lineal_bal_acc") is not None:
            return (f"Referencia: un modelo lineal simple acierta de forma pareja entre clases el "
                    f"{dec(b['lineal_bal_acc'] * 100, 0)}% (este modelo: {dec(m['bal_acc'] * 100, 0)}%).")
        return None

    def lineas_resumen(self) -> list:
        m, b = self.metricas, self.baseline
        lineas = []
        if self.tipo == "regresion":
            lineas.append(
                f"Error típico al predecir «{self.objetivo}»: {_fmt(m['rmse'])} "
                f"(error promedio absoluto: {_fmt(m['mae'])})."
            )
            if b.get("rmse"):
                mejora = (1 - m["rmse"] / b["rmse"]) * 100 if b["rmse"] else 0
                lineas.append(
                    f"Si siempre respondieras el promedio, el error típico sería {_fmt(b['rmse'])}: "
                    f"el modelo lo reduce en {dec(mejora, 0)}%."
                )
            lineas.append(f"Explica el {dec(max(m['r2'], 0) * 100, 0)}% de la variación (R² = {dec(m['r2'], 3)}).")
        else:
            lineas.append(f"Acierta «{self.objetivo}» el {dec(m['accuracy'] * 100, 1)}% de las veces en datos que no vio al entrenar.")
            lineas.append(
                f"Responder siempre la clase más frecuente acertaría el {dec(b['accuracy'] * 100, 1)}%."
            )
            if self._caso_raro():
                lineas.append(f"Detecta el {dec(m['recall_min'] * 100, 0)}% de los casos «{self.clase_minoritaria}» "
                              f"y acierta el {dec(m['precision_min'] * 100, 0)}% de las veces que los marca.")
            if m.get("auc") is not None:
                lineas.append(f"AUC: {dec(m['auc'], 3)} (1 es perfecto, {dec(0.5, 1)} es azar).")
        extra = self.texto_lineal()
        if extra:
            lineas.append(extra)
        lineas.append(self.texto_validacion())
        ts = self.texto_salida()
        if ts:
            lineas.append(ts)
        if self.importancias:
            lineas.append("Las columnas «que más pesan» resumen el modelo en general, no explican un caso puntual; y si "
                          "dos columnas dicen casi lo mismo, el peso se reparte entre ellas.")
        return lineas


# ----------------------------------------------------------------------
# Calibración: ¿el «30%» que dice el modelo ocurre de verdad ~30% de las veces?
# ----------------------------------------------------------------------
MIN_CASOS_CALIBRACION = 200       # menos casos que esto: no se puede comprobar
MIN_CLASE_CALIBRACION = 20        # y al menos 20 de cada clase
MIN_CASOS_ISOTONICA = 1000        # con menos casos se usa un ajuste suave (Platt) en vez de la isotónica


class Calibrador:
    """Corrige las probabilidades del modelo con lo que pasó de verdad en datos que no vio. Vive solo en memoria
    (no se guarda: igual que el modelo, se reentrena con un clic)."""

    def __init__(self, metodo: str):
        self.metodo = metodo
        self._m = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def ajustar(self, p, y) -> "Calibrador":
        if self.metodo == "isotonica":
            from sklearn.isotonic import IsotonicRegression
            self._m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, dtype="float64"), y)
        else:
            from sklearn.linear_model import LogisticRegression
            self._m = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        return self

    def transformar(self, p) -> np.ndarray:
        if self.metodo == "isotonica":
            return np.asarray(self._m.predict(np.asarray(p, dtype="float64")), dtype="float64")
        return self._m.predict_proba(self._logit(p))[:, 1]


def _ece_y_tabla(p, y):
    """Error de calibración (promedio ponderado de |lo que dijo − lo que pasó|) y la tabla por tramos de igual tamaño."""
    n = len(y)
    k = int(max(2, min(10, n // 50)))
    orden = np.argsort(p, kind="stable")
    tabla, ece = [], 0.0
    for grupo in np.array_split(orden, k):
        if len(grupo) == 0:
            continue
        pm, fr = float(np.mean(p[grupo])), float(np.mean(y[grupo]))
        ece += len(grupo) / n * abs(pm - fr)
        tabla.append({"desde": float(p[grupo].min()), "hasta": float(p[grupo].max()), "estimada": pm, "real": fr,
                      "casos": int(len(grupo))})
    return float(ece), tabla


def evaluar_calibracion(p, y, semilla: int = 0) -> dict:
    """Mide la calibración con predicciones FUERA DE MUESTRA (p = probabilidad estimada de la clase 1, y = 0/1).
    Calibra solo si hay casos suficientes y la calibración MEJORA en datos separados (validación cruzada de 2 partes)."""
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    y = np.asarray(y).astype(int)
    n, pos = len(y), int(y.sum())
    base = float(y.mean()) if n else 0.0
    out = {"n": n, "positivos": pos, "brier": float(np.mean((p - y) ** 2)) if n else None,
           "brier_base": float(np.mean((base - y) ** 2)) if n else None,
           "verificable": False, "calibrada": False, "metodo": None, "calibrador": None,
           "ece": None, "ece_calibrado": None, "tabla": [], "motivo": ""}
    if n < MIN_CASOS_CALIBRACION or pos < MIN_CLASE_CALIBRACION or (n - pos) < MIN_CLASE_CALIBRACION:
        out["motivo"] = (f"hay pocos casos para comprobarlo ({n} casos, {pos} de la clase que se predice; "
                         f"se necesitan al menos {MIN_CASOS_CALIBRACION} y {MIN_CLASE_CALIBRACION} de cada clase)")
        return out
    ece, tabla = _ece_y_tabla(p, y)
    tol = 0.05 + 0.5 / np.sqrt(n)               # tolerancia: con pocos casos el azar mueve la tabla
    out.update(verificable=True, ece=ece, tabla=tabla, tolerancia=float(tol))
    if ece <= tol:
        out["calibrada"] = True                  # ya decía la verdad: no hace falta corregir
        return out
    metodo = "isotonica" if (n >= MIN_CASOS_ISOTONICA and pos >= 50 and n - pos >= 50) else "platt"
    idx = np.random.default_rng(semilla).permutation(n)
    mitades = (idx[: n // 2], idx[n // 2:])
    antes, despues = [], []
    try:
        for a, b in ((0, 1), (1, 0)):
            cal = Calibrador(metodo).ajustar(p[mitades[a]], y[mitades[a]])
            despues.append(_ece_y_tabla(cal.transformar(p[mitades[b]]), y[mitades[b]])[0])
            antes.append(_ece_y_tabla(p[mitades[b]], y[mitades[b]])[0])
        ece_cal, ece_antes = float(np.mean(despues)), float(np.mean(antes))
        if ece_cal <= tol and ece_cal < 0.8 * ece_antes:
            out["calibrador"] = Calibrador(metodo).ajustar(p, y)
            out.update(calibrada=True, metodo=metodo, ece_calibrado=ece_cal)
        else:
            out["motivo"] = "corregirlas con lo observado no las dejó confiables al probar con datos separados"
    except Exception as e:                       # no se oculta: se anota el motivo
        out["motivo"] = f"no se pudo calibrar ({type(e).__name__})"
    return out


# ----------------------------------------------------------------------
# Entrenar
# ----------------------------------------------------------------------
def diagnosticar_variables(df: pd.DataFrame, objetivo: str, columnas: list, tipo: str) -> list:
    """Busca columnas que podrían ser «trampa» (contener ya la respuesta).

    Solo AVISA: nunca quita columnas, el usuario decide. Devuelve [(columna, motivo)].
    Es barato a propósito (trabaja con una muestra y mira solo categorías con
    muchas filas cada una) para no pesar en archivos grandes.
      - Número casi idéntico al objetivo numérico (correlación >= UMBRAL_FUGA).
      - Categoría/sí-no cuyos valores casi determinan el objetivo, y que además
        acierta bastante más que simplemente responder lo más frecuente."""
    muestra = df if len(df) <= MUESTRA_DIAGNOSTICO else df.sample(MUESTRA_DIAGNOSTICO, random_state=0)
    y = muestra[objetivo]
    yn = pd.to_numeric(y, errors="coerce") if tipo == "regresion" else None
    y_txt = y.astype(str) if tipo == "clasificacion" else None
    salida = []
    for c in columnas:
        if c not in muestra.columns or c == objetivo:
            continue
        s = muestra[c]
        try:
            t = tipo_columna(s)
            if tipo == "regresion" and t in ("numerica", "numerica_texto"):
                xn = pd.to_numeric(s, errors="coerce")
                ok = xn.notna() & yn.notna()
                if ok.sum() >= MINIMO_FILAS and xn[ok].nunique() > 1 and yn[ok].nunique() > 1:
                    r = abs(float(np.corrcoef(xn[ok].to_numpy(float), yn[ok].to_numpy(float))[0, 1]))
                    if np.isfinite(r) and r >= UMBRAL_FUGA:
                        salida.append((str(c), f"es casi una copia de «{objetivo}» (correlación {dec(r, 3)})"))
            elif tipo == "clasificacion" and t in ("numerica", "numerica_texto") and y_txt.nunique() == 2:
                # número que casi copia una respuesta de dos clases (por ejemplo 0/1)
                xn = pd.to_numeric(s, errors="coerce")
                cod = pd.Series(pd.factorize(y_txt)[0], index=y_txt.index).astype(float)
                ok = xn.notna()
                if ok.sum() >= MINIMO_FILAS and xn[ok].nunique() > 1:
                    r = abs(float(np.corrcoef(xn[ok].to_numpy(float), cod[ok].to_numpy(float))[0, 1]))
                    if np.isfinite(r) and r >= UMBRAL_FUGA:
                        salida.append((str(c), f"es casi una copia de «{objetivo}» (correlación {dec(r, 3)})"))
            elif tipo == "clasificacion" and t in ("categorica", "booleana"):
                nn = int(s.notna().sum())
                nun = int(s.nunique(dropna=True))
                # pocas categorías y con bastantes filas cada una (si no, acertar es trivial)
                if nn < MINIMO_FILAS or not (2 <= nun <= 50) or nn / nun < 20:
                    continue
                par = pd.DataFrame({"x": s.astype(str), "y": y_txt})[s.notna().to_numpy()]
                por_cat = par.groupby("x")["y"].agg(lambda v: v.value_counts().iloc[0])
                pureza = float(por_cat.sum() / len(par))
                frecuente = float(par["y"].value_counts(normalize=True).iloc[0])
                if pureza >= 0.995 and pureza - frecuente >= 0.02:
                    salida.append((str(c), f"sus valores casi determinan «{objetivo}» ({dec(pureza * 100, 1)}% de las filas)"))
        except Exception:
            continue
    return salida


def _crear_modelo(tipo, config, mascara, n_filas, n_iter=None, parar_antes=True):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    clase = HistGradientBoostingRegressor if tipo == "regresion" else HistGradientBoostingClassifier
    kw = dict(
        max_iter=int(n_iter or config.max_iter),
        learning_rate=float(config.tasa_aprendizaje),
        max_depth=int(config.profundidad_max) if config.profundidad_max else None,
        random_state=int(config.semilla),
    )
    if parar_antes and n_filas > 2000:
        kw.update(early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)
    else:
        kw.update(early_stopping=False)
    if any(mascara):
        kw["categorical_features"] = list(mascara)
    return clase(**kw)


def _usar_pesos(config, yt) -> bool:
    """¿Se le da más peso a la clase poco frecuente? (solo clasificación)"""
    if config.balancear == "no":
        return False
    if config.balancear == "si":
        return True
    cuentas = np.bincount(yt)
    cuentas = cuentas[cuentas > 0]
    return bool(len(cuentas) > 1 and cuentas.min() / cuentas.sum() < UMBRAL_DESBALANCE)


def _pesos(y):
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def _referencia_lineal(tipo, X_tr, y_tr, X_val, mascara, semilla, pesos=None):
    """Predicciones de un modelo lineal básico (otra «respuesta simple» contra la
    que comparar). Devuelve None si no se pudo armar: es solo una referencia."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        mascara = np.asarray(mascara, dtype=bool)
        idx_cat, idx_num = np.flatnonzero(mascara), np.flatnonzero(~mascara)
        if len(X_tr) > MAX_FILAS_LINEAL:
            sel = np.sort(np.random.default_rng(semilla).choice(len(X_tr), MAX_FILAS_LINEAL, replace=False))
            X_tr, y_tr = X_tr[sel], y_tr[sel]
            pesos = pesos[sel] if pesos is not None else None
        trans = []
        if len(idx_num):
            trans.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), idx_num))
        if len(idx_cat):
            trans.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value=-1),
                                               OneHotEncoder(handle_unknown="ignore")), idx_cat))
        if not trans:
            return None
        est = Ridge(alpha=1.0) if tipo == "regresion" else LogisticRegression(max_iter=300)
        modelo = Pipeline([("pre", ColumnTransformer(trans)), ("est", est)])
        modelo.fit(X_tr, y_tr, **({"est__sample_weight": pesos} if pesos is not None else {}))
        return modelo.predict(X_val)
    except Exception:
        return None


def _promediar(lista):
    claves = set().union(*[d.keys() for d in lista]) if lista else set()
    medias, sds = {}, {}
    for c in claves:
        v = [d[c] for d in lista if d.get(c) is not None and np.isfinite(d[c])]
        medias[c] = float(np.mean(v)) if v else None
        sds[c] = float(np.std(v)) if len(v) > 1 else 0.0
    return medias, sds


def _separar(tipo, config, yt, n, fechas=None, grupos=None):
    """Reparte las filas en pruebas: lista de (indices_entrenamiento, indices_validacion).
    Con K pruebas cada fila se usa para evaluar una vez (aleatoria y grupos), o se
    prueba siempre con filas más nuevas que las de entrenamiento (fecha)."""
    from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold,
                                         StratifiedKFold, TimeSeriesSplit, train_test_split)
    k = max(int(config.pliegues or 1), 1)
    if n > CV_MAX_FILAS:
        k = 1
    k = int(min(k, max(n // 10, 1)))
    idx = np.arange(n)
    semilla = int(config.semilla)
    clasif = tipo == "clasificacion"
    modo = config.validacion

    if modo == "temporal":
        orden = np.argsort(fechas, kind="stable")
        k = int(min(k, max(n // 20 - 1, 1)))
        if k <= 1:
            n_val = max(int(n * FRACCION_VALIDACION), 1)
            return [(orden[:-n_val], orden[-n_val:])]
        return [(orden[a], orden[b]) for a, b in TimeSeriesSplit(n_splits=k).split(orden)]

    if modo == "grupos":
        n_g = len(np.unique(grupos))
        if n_g < 5:
            raise ValueError("La columna de grupos tiene muy pocos valores distintos para separar una "
                             "parte de prueba. Elige otra columna o cambia a validación aleatoria.")
        k = int(min(k, n_g // 2))
        if k <= 1:
            gss = GroupShuffleSplit(n_splits=1, test_size=FRACCION_VALIDACION, random_state=semilla)
            return [next(gss.split(idx, groups=grupos))]
        if clasif:
            try:
                return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
                            .split(idx, yt, grupos))
            except ValueError:
                pass
        return list(GroupKFold(n_splits=k).split(idx, groups=grupos))

    if k <= 1:
        estratificar = None
        if clasif:
            cuentas = np.bincount(yt)
            if cuentas[cuentas > 0].min() >= 2:
                estratificar = yt
        a, b = train_test_split(idx, test_size=FRACCION_VALIDACION, random_state=semilla, stratify=estratificar)
        return [(a, b)]
    if clasif and np.bincount(yt)[np.bincount(yt) > 0].min() >= k:
        return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla).split(idx, yt))
    return list(KFold(n_splits=k, shuffle=True, random_state=semilla).split(idx))


def entrenar(df: pd.DataFrame, config: ConfigPrediccion, columnas: list,
             progreso=None, forzar_categoria=None) -> ResultadoEntrenamiento:
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 f1_score, log_loss, mean_absolute_error, mean_squared_error,
                                 precision_score, r2_score, recall_score, roc_auc_score)

    prog = progreso or (lambda texto: None)
    obj = config.objetivo
    if not obj or obj not in df.columns:
        raise ValueError("Elige la columna que quieres predecir.")
    columnas = [c for c in columnas if c in df.columns and c != obj]
    if not columnas:
        raise ValueError("No hay ninguna columna marcada para que el modelo se fije en ella.")
    advertencias = []

    prog("Preparando los datos…")
    pos = np.flatnonzero(df[obj].notna().to_numpy())
    if len(pos) < MINIMO_FILAS:
        raise ValueError(f"Hay muy pocas filas con valor en «{obj}» ({len(pos)}); se necesitan al menos {MINIMO_FILAS}.")
    if len(pos) > config.max_filas:
        rng = np.random.default_rng(config.semilla)
        pos = np.sort(rng.choice(pos, size=int(config.max_filas), replace=False))
        advertencias.append(
            f"La tabla es grande: se usó una muestra de {dec(len(pos), 0, True)} filas "
            f"(ajustable en Opciones avanzadas)."
        )
    modo = config.validacion if config.validacion in ("aleatoria", "temporal", "grupos") else "aleatoria"
    if modo == "temporal" and config.columna_fecha not in df.columns:
        raise ValueError("Para probar «por fecha» elige una columna de fecha, o cambia a validación aleatoria.")
    if modo == "grupos" and config.columna_grupo not in df.columns:
        raise ValueError("Para probar «por grupos» elige la columna que identifica el grupo "
                         "(por ejemplo, la persona o cosa que se repite en varias filas).")
    if modo == "aleatoria":
        con_fecha = columnas_fecha(df, excluir=[obj])
        if con_fecha:
            advertencias.append(
                f"La tabla tiene una columna de fecha («{con_fecha[0]}»), pero la prueba mezcló las filas al azar. "
                "Si lo que predices cambia con el tiempo, la nota sale más optimista que la realidad: "
                "para saber cómo le iría con casos nuevos, prueba «por fecha».")
    cols_necesarias = list(dict.fromkeys(
        columnas + [obj]
        + ([config.columna_fecha] if modo == "temporal" else [])
        + ([config.columna_grupo] if modo == "grupos" else [])))
    datos = df[cols_necesarias].iloc[pos].reset_index(drop=True)
    y_raw = datos[obj]
    # Filas idénticas: con prueba aleatoria una copia puede caer en el entrenamiento y otra en la prueba, y la nota sale
    # inflada. Solo se AVISA (nunca se borra nada). Si todas las columnas tienen pocos valores distintos, repetirse es
    # normal y no se avisa.
    cols_dup = [c for c in dict.fromkeys(columnas + [obj])]
    n_dup = int(datos[cols_dup].duplicated().sum())
    if n_dup and n_dup / len(datos) >= UMBRAL_DUPLICADAS and any(
            datos[c].nunique(dropna=True) >= 50 for c in columnas if c in datos.columns):
        advertencias.append(
            f"{dec(n_dup, 0, True)} filas ({dec(n_dup / len(datos) * 100, 0)}%) son idénticas a otra en las columnas que usa el modelo. "
            "Si una copia cae en el entrenamiento y otra en la prueba, la nota sale más alta que la real. "
            "Hadar no las borró: revisa si son repeticiones por error" + (" o prueba «por fecha»." if modo == "aleatoria" else "."))

    tipo = inferir_tipo_problema(y_raw, config.tipo)
    clases = None
    if tipo == "regresion":
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype="float64")
        if np.isnan(y).any():
            raise ValueError(f"«{obj}» tiene valores que no son números; elige tipo «Categoría» o revisa la columna.")
        if config.log_objetivo:
            if (y < 0).any():
                raise ValueError("La escala logarítmica solo sirve si todos los valores son 0 o más.")
            yt = np.log1p(y)
        else:
            yt = y
    else:
        valores = list(pd.unique(y_raw.astype(object)))
        try:
            clases = sorted(valores)
        except TypeError:
            clases = sorted(valores, key=str)
        if len(clases) < 2:
            raise ValueError(f"«{obj}» tiene un solo valor distinto; no hay nada que predecir.")
        if len(clases) > MAX_CLASES:
            raise ValueError(
                f"«{obj}» tiene {len(clases)} valores distintos. Si es un número, elige tipo «Número»; "
                f"si es un identificador, no sirve para predecir."
            )
        yt = pd.Categorical(y_raw.astype(object), categories=clases).codes.astype("int64")
        y = yt

    n = len(datos)
    fechas = grupos = None
    if modo == "temporal":
        fechas = _a_datetime(datos[config.columna_fecha]).fillna(pd.Timestamp("1900-01-01")).to_numpy()
    elif modo == "grupos":
        grupos = pd.factorize(normalizar_clave(datos[config.columna_grupo]))[0]

    # --- avisos útiles sobre cómo probar -------------------------------
    if modo == "aleatoria":
        repetidas = []
        for c in [c for c in df.columns if c != obj and es_columna_id(c)][:20]:
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= len(df) * 0.5:
                repetidas.append(str(c))
        if repetidas:
            advertencias.append(
                f"La columna «{repetidas[0]}» se repite en varias filas (la misma persona o cosa aparece más de una vez). "
                f"Si vas a predecir para casos nuevos, prueba «Por grupos»: así una misma persona no queda a la vez "
                f"en entrenamiento y en prueba, lo que haría el resultado más optimista de lo real.")
        elif config.columna_fecha in df.columns:
            advertencias.append(
                "Hay una columna de fecha. Si vas a predecir el futuro, prueba «Por fecha»: es más exigente y se "
                "parece más a lo que pasará en la realidad.")

    # --- columnas que podrían contener ya la respuesta (solo aviso) ----
    sospechosas = diagnosticar_variables(datos, obj, columnas, tipo)
    if sospechosas:
        lista = "; ".join(f"«{c}» {motivo}" for c, motivo in sospechosas[:4])
        advertencias.append(
            "Ojo, estas columnas podrían ser una trampa (ya contienen la respuesta o algo que solo se sabe "
            "después): " + lista + ". Si no las tendrías al momento de predecir, desmárcalas en el Paso 2.")

    # --- partir en pruebas --------------------------------------------
    partes = _separar(tipo, config, yt, n, fechas, grupos)
    if any(len(a) < 10 or len(b) < 5 for a, b in partes):
        raise ValueError("Hay muy pocas filas para separar una parte de prueba.")
    usar_pesos = tipo == "clasificacion" and _usar_pesos(config, yt)
    clase_min = None
    if tipo == "clasificacion" and len(clases) == 2:
        clase_min_cod = int(np.argmin(np.bincount(yt, minlength=2)))
        clase_min = clases[clase_min_cod]

    por_pliegue, base_pliegue, mejoras = [], [], []
    importancia_total = {}
    fallos_importancia = []
    n_iters, tam_tr, tam_val = [], [], []
    oof = {"pos": [], "real": [], "pred": [], "conf": [], "pbin": [], "ybin": []}
    pos_todas = []
    for i, (idx_tr, idx_val) in enumerate(partes):
        prog(f"Probando el modelo ({i + 1} de {len(partes)})…" if len(partes) > 1
             else "Entrenando y comprobando el modelo…")
        pre = Preprocesador(columnas).ajustar(datos.iloc[idx_tr][columnas], forzar_categoria)
        X_tr = pre.transformar(datos.iloc[idx_tr][columnas])
        X_val = pre.transformar(datos.iloc[idx_val][columnas])
        pesos_tr = _pesos(yt[idx_tr]) if usar_pesos else None
        modelo = _crear_modelo(tipo, config, pre.mascara_categ, len(idx_tr))
        try:
            modelo.fit(X_tr, yt[idx_tr], sample_weight=pesos_tr)
        except ValueError as e:
            raise ValueError(f"No se pudo entrenar el modelo: {e}")
        n_iters.append(int(getattr(modelo, "n_iter_", config.max_iter)))
        tam_tr.append(len(idx_tr))
        tam_val.append(len(idx_val))
        pos_val = pos[idx_val]
        met, bas = {}, {}
        lineal = _referencia_lineal(tipo, X_tr, yt[idx_tr], X_val, pre.mascara_categ, config.semilla, pesos_tr)

        if tipo == "regresion":
            pred = modelo.predict(X_val)
            if config.log_objetivo:
                pred = np.clip(np.expm1(pred), 0, None)
            real = y[idx_val]
            met["rmse"] = float(np.sqrt(mean_squared_error(real, pred)))
            met["mae"] = float(mean_absolute_error(real, pred))
            met["r2"] = float(r2_score(real, pred))
            base = float(np.mean(y[idx_tr]))
            bas["rmse"] = float(np.sqrt(np.mean((real - base) ** 2)))
            if lineal is not None:
                if config.log_objetivo:
                    lineal = np.clip(np.expm1(lineal), 0, None)
                bas["lineal_rmse"] = float(np.sqrt(mean_squared_error(real, lineal)))
            oof["pos"].append(pos_val)
            oof["real"].append(real)
            oof["pred"].append(pred)
        else:
            pred = modelo.predict(X_val)
            real = yt[idx_val]
            met["accuracy"] = float(accuracy_score(real, pred))
            met["f1"] = float(f1_score(real, pred, average="macro"))
            met["bal_acc"] = float(balanced_accuracy_score(real, pred))
            frecuente = np.bincount(yt[idx_tr]).argmax()
            bas["accuracy"] = float(np.mean(real == frecuente))
            bas["bal_acc"] = 1.0 / max(len(np.unique(real)), 1)
            if lineal is not None:
                bas["lineal_bal_acc"] = float(balanced_accuracy_score(real, lineal))
            met["auc"] = None
            proba = None
            try:
                proba = modelo.predict_proba(X_val)
                if len(clases) == 2 and len(np.unique(real)) == 2:
                    met["auc"] = float(roc_auc_score(real, proba[:, 1]))
                elif len(clases) > 2 and set(np.unique(real)) <= set(modelo.classes_):
                    met["auc"] = float(roc_auc_score(
                        real, proba, multi_class="ovr", labels=list(modelo.classes_)))
                met["logloss"] = float(log_loss(real, proba, labels=list(modelo.classes_)))
            except Exception:
                pass
            if clase_min is not None and proba is not None and clase_min_cod in list(modelo.classes_):
                es_min = real == clase_min_cod
                if es_min.any():
                    j = list(modelo.classes_).index(clase_min_cod)
                    met["recall_min"] = float(recall_score(es_min, pred == clase_min_cod, zero_division=0))
                    met["precision_min"] = float(precision_score(es_min, pred == clase_min_cod, zero_division=0))
                    try:
                        met["ap"] = float(average_precision_score(es_min, proba[:, j]))
                        bas["ap"] = float(es_min.mean())
                    except Exception:
                        pass
            if proba is not None:
                oof["pos"].append(pos_val)
                oof["real"].append(real)
                oof["pred"].append(pred)
                oof["conf"].append(proba.max(axis=1))
                if len(clases) == 2 and 1 in list(modelo.classes_):
                    oof["pbin"].append(proba[:, list(modelo.classes_).index(1)])
                    oof["ybin"].append((real == 1).astype(int))
        por_pliegue.append(met)
        base_pliegue.append(bas)
        mejoras.append(_mejora_de(tipo, met, bas))

        # --- en qué columnas se fijó (en cada prueba, sobre lo que no vio) ---
        try:
            from sklearn.inspection import permutation_importance
            k = min(2000, len(idx_val))
            sub = np.random.default_rng(config.semilla + i).choice(len(idx_val), size=k, replace=False)
            pi = permutation_importance(
                modelo, X_val[sub], yt[idx_val][sub], n_repeats=5, random_state=config.semilla, n_jobs=1,
                scoring="balanced_accuracy" if tipo == "clasificacion" else None)
            por_col = {}
            for fuente, valor in zip(pre.fuente, pi.importances_mean):
                por_col[fuente] = por_col.get(fuente, 0.0) + max(float(valor), 0.0)
            tot = sum(por_col.values())
            if tot > 0:
                for c, v in por_col.items():
                    importancia_total[c] = importancia_total.get(c, 0.0) + v / tot
        except Exception as e:      # no se oculta: el usuario debe saber que falta esta sección
            fallos_importancia.append(f"{type(e).__name__}: {str(e)[:80]}")

    if fallos_importancia:
        advertencias.append("No se pudo calcular qué columnas pesan más en el modelo (" + fallos_importancia[0]
                            + "). El resto de los resultados no se ve afectado.")

    metricas, metricas_sd = _promediar(por_pliegue)
    baseline, _ = _promediar(base_pliegue)
    n_val_total = int(sum(tam_val))

    # --- lo que no vio, junto, para la lista de revisión y el gráfico ---
    validacion, revision = None, None
    if oof["pos"]:
        p_all = np.concatenate(oof["pos"])
        r_all = np.concatenate(oof["real"])
        d_all = np.concatenate(oof["pred"])
        if tipo == "regresion":
            dif = r_all - d_all
            orden = np.argsort(-np.abs(dif))[:REVISION_MAX]
            revision = {"tipo": "regresion", "pos": p_all[orden], "real": r_all[orden],
                        "pred": d_all[orden], "extra": dif[orden]}
            kk = min(len(r_all), PUNTOS_GRAFICO)
            sel = np.random.default_rng(config.semilla).choice(len(r_all), size=kk, replace=False)
            validacion = (r_all[sel], d_all[sel])
        elif oof["conf"]:
            c_all = np.concatenate(oof["conf"])
            mal = np.flatnonzero(d_all != r_all)
            orden = mal[np.argsort(-c_all[mal])][:REVISION_MAX]
            arr = np.array(clases, dtype=object)
            revision = {"tipo": "clasificacion", "pos": p_all[orden], "real": arr[r_all[orden]],
                        "pred": arr[d_all[orden]], "extra": c_all[orden]}

    # --- avisos sobre el resultado ------------------------------------
    if tipo == "regresion":
        if baseline["rmse"] and metricas["rmse"] > baseline["rmse"] * 0.99:
            advertencias.append(
                "El modelo casi no mejora a «responder siempre el promedio». Revisa qué columnas "
                "marcaste como útiles, o puede que estos datos no alcancen para predecir eso.")
        lin = baseline.get("lineal_rmse")
        if lin and metricas["rmse"] >= lin * 0.98 and baseline["rmse"] and lin < baseline["rmse"] * 0.99:
            advertencias.append(
                "Un modelo lineal simple rinde igual o mejor que este: la relación parece simple, "
                "así que el modelo avanzado no está agregando nada.")
    else:
        if metricas["bal_acc"] <= baseline["bal_acc"] + 0.01:
            advertencias.append(
                "El modelo casi no mejora a «responder siempre la clase más frecuente». Revisa qué "
                "columnas marcaste como útiles.")
        lin = baseline.get("lineal_bal_acc")
        if lin is not None and metricas["bal_acc"] <= lin + 0.01 and lin > baseline["bal_acc"] + 0.05:
            advertencias.append(
                "Un modelo lineal simple rinde igual o mejor que este: la relación parece simple, "
                "así que el modelo avanzado no está agregando nada.")
        cuentas = np.bincount(yt)
        raras = [str(clases[i]) for i, c in enumerate(cuentas) if 0 < c < 5]
        if raras:
            advertencias.append("Hay clases con menos de 5 filas, el modelo casi no puede aprenderlas: " + ", ".join(raras[:5]) + ".")
    if n_val_total < MINIMO_VALIDACION:
        advertencias.append(
            f"Hay pocas filas de prueba ({n_val_total}): el resultado es poco confiable; "
            f"con más datos las cifras serían más estables.")

    importancias = []
    tot = sum(importancia_total.values())
    if tot > 0:
        importancias = sorted(((c, v / tot * 100) for c, v in importancia_total.items()),
                              key=lambda t: t[1], reverse=True)

    # --- modelo final con todos los datos -----------------------------
    prog("Entrenando el modelo final con todos los datos…")
    pre_final = Preprocesador(columnas).ajustar(datos[columnas], forzar_categoria)
    X_all = pre_final.transformar(datos[columnas])
    n_iter = max(int(np.median(n_iters)), 10)
    modelo_final = _crear_modelo(tipo, config, pre_final.mascara_categ, n, n_iter=n_iter, parar_antes=False)
    modelo_final.fit(X_all, yt, sample_weight=_pesos(yt) if usar_pesos else None)

    # --- ¿qué significa el número que se entrega? ---------------------------------------------------
    tipo_salida, calibracion, calibrador = "", None, None
    if tipo == "regresion":
        tipo_salida = "valor"
    elif config.salida == "probabilidad":
        if len(clases) == 2 and oof["pbin"]:
            calibracion = evaluar_calibracion(np.concatenate(oof["pbin"]), np.concatenate(oof["ybin"]), config.semilla)
            calibrador = calibracion.pop("calibrador", None)
            if calibracion["verificable"] and calibracion["calibrada"] and not (usar_pesos and calibrador is None):
                tipo_salida = "probabilidad"
            else:
                tipo_salida = "ranking"
                if usar_pesos and calibracion["verificable"] and calibracion["calibrada"]:
                    calibracion["motivo"] = "se dio más peso a la clase poco frecuente, lo que distorsiona el número"
            if tipo_salida == "ranking" and not calibracion["motivo"]:
                calibracion["motivo"] = "no se pudo comprobar que coincida con lo que ocurre"
        else:
            tipo_salida = "ranking"
            calibracion = {"motivo": "con más de dos categorías Hadar no comprueba la calibración", "verificable": False,
                           "calibrada": False}
    if usar_pesos:
        if tipo_salida == "probabilidad":
            advertencias.append("Hay una clase poco frecuente: se le dio más peso para que el modelo no la ignore, y después "
                                "se corrigieron las probabilidades con lo observado en datos que no vio.")
        elif config.salida == "probabilidad":
            advertencias.append("Hay una clase poco frecuente, así que se le dio más peso para que el modelo no la ignore. "
                                "Por eso las probabilidades que entrega sirven para ordenar casos (cuál es más probable), "
                                "no como porcentaje real.")
        else:
            advertencias.append("Hay una clase poco frecuente, así que se le dio más peso para que el modelo no la ignore.")
    if calibracion is not None and calibracion.get("verificable") and tipo_salida == "ranking" and not calibracion.get("calibrada"):
        advertencias.append("Las probabilidades que da el modelo no coinciden con lo que ocurre en datos que no vio "
                            "(dice, por ejemplo, 30% y ocurre otra proporción): úsalas para ordenar, no como porcentajes.")
    return ResultadoEntrenamiento(
        tipo=tipo, objetivo=obj, columnas=columnas, clases=clases,
        metricas=metricas, baseline=baseline, importancias=importancias,
        n_entrenamiento=int(np.mean(tam_tr)), n_validacion=n_val_total, n_iteraciones=n_iter,
        advertencias=advertencias, log_objetivo=bool(config.log_objetivo and tipo == "regresion"),
        salida=config.salida, modelo=modelo_final, preprocesador=pre_final,
        validacion=validacion, revision=revision,
        metricas_sd=metricas_sd, mejoras_pliegue=mejoras, n_pliegues=len(partes),
        modo_validacion=modo, balanceado=usar_pesos, clase_minoritaria=clase_min,
        tipo_salida=tipo_salida, calibracion=calibracion, calibrador=calibrador, semilla=config.semilla,
    )


# ----------------------------------------------------------------------
# Predecir
# ----------------------------------------------------------------------
def predecir(resultado: ResultadoEntrenamiento, df: pd.DataFrame,
             columna_id: str | None = None, salida: str | None = None) -> pd.DataFrame:
    """Devuelve la tabla lista para entregar: la columna id (si existe) y la
    predicción con el nombre de la columna objetivo."""
    faltan = [c for c in resultado.columnas if c not in df.columns]
    if faltan:
        raise ValueError("La tabla de prueba no tiene estas columnas: " + ", ".join(faltan[:8]))
    salida = salida or resultado.salida
    X = resultado.preprocesador.transformar(df[resultado.columnas])
    modelo = resultado.modelo
    avisos = []
    for col, (n_nuevas, ejemplo) in getattr(resultado.preprocesador, "desconocidas", {}).items():
        avisos.append(f"{dec(n_nuevas, 0, True)} filas ({dec(n_nuevas / max(len(df), 1) * 100, 1)}%) traen en «{col}» valores que el modelo "
                      f"nunca vio (por ejemplo «{ejemplo}»): para esas filas el modelo no tiene cómo usar esa columna "
                      f"y la predicción es menos confiable.")

    id_col = elegir_columna_id(df, columna_id)
    salida_df = pd.DataFrame({id_col: df[id_col].to_numpy()}) if id_col else \
        pd.DataFrame({"Id": np.arange(len(df))})

    if resultado.tipo == "regresion":
        pred = modelo.predict(X)
        if resultado.log_objetivo:
            pred = np.clip(np.expm1(pred), 0, None)
        salida_df[resultado.objetivo] = pred
        salida_df.attrs["avisos"] = avisos
        salida_df.attrs["tipo_salida"] = "valor"
        return salida_df

    clases = np.array(resultado.clases, dtype=object)
    presentes = clases[modelo.classes_]
    if salida == "probabilidad":
        proba = modelo.predict_proba(X)
        if len(presentes) == 2 and len(resultado.clases) == 2:
            p1 = proba[:, 1]
            if getattr(resultado, "calibrador", None) is not None:
                p1 = resultado.calibrador.transformar(p1)
            salida_df[resultado.objetivo] = p1
        else:
            for j, etiqueta in enumerate(presentes):
                salida_df[str(etiqueta)] = proba[:, j]
    else:
        salida_df[resultado.objetivo] = presentes[np.searchsorted(modelo.classes_, modelo.predict(X))]
    salida_df.attrs["avisos"] = avisos
    salida_df.attrs["tipo_salida"] = resultado.tipo_salida if salida == "probabilidad" else ""
    return salida_df


# ----------------------------------------------------------------------
# Demo / auto-test rápido
# ----------------------------------------------------------------------
def _demo():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({
        "Id": np.arange(n),
        "fecha": pd.date_range("2022-01-01", periods=n, freq="h").astype(str),
        "tienda": rng.choice(list("ABCDE"), n),
        "precio": rng.normal(100, 20, n),
        "constante": 1,
    })
    df["ventas"] = df["precio"] * 0.5 + (df["tienda"] == "A") * 30 + rng.normal(0, 5, n)
    cfg = ConfigPrediccion(tabla_entrenamiento="t", objetivo="ventas")
    sug = sugerir_columnas(df, "ventas")
    for s in sug:
        print(f"  {s.nombre:12s} {s.tipo:12s} usar={s.usar!s:5s} {s.motivo}")
    res = entrenar(df, cfg, columnas_a_usar(sug, {}), progreso=print)
    print("\n".join(res.lineas_resumen()))
    print(res.importancias[:5])
    print(predecir(res, df.head(3)))


if __name__ == "__main__":
    _demo()