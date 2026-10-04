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
  2. Se prueba el modelo contra una parte de los datos que NO vio al
     entrenar, y se compara contra una "respuesta tonta" (el promedio, o la
     clase más frecuente) para que se note si de verdad aporta algo.
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

MAX_CATEGORIAS = 250          # HistGradientBoosting admite categorías nativas hasta ~255
MAX_CLASES = 100
MUESTRA_DIAGNOSTICO = 100_000
UMBRAL_CASI_UNICA = 0.95
UMBRAL_VACIA = 0.98
UMBRAL_FUGA = 0.999
FRACCION_VALIDACION = 0.2
MINIMO_FILAS = 30
MAX_FRECUENCIAS = 50_000

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
    validacion: str = "aleatoria"      # "aleatoria" | "temporal"
    columna_fecha: str | None = None
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


def _a_datetime(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(serie):
        dt = serie
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dt = pd.to_datetime(serie, errors="coerce", format="mixed")
            except (ValueError, TypeError):
                dt = pd.to_datetime(serie, errors="coerce")
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            conv = pd.to_datetime(muestra, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            conv = pd.to_datetime(muestra, errors="coerce")
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
                dt = _a_datetime(s)
                hora = bool((dt.dt.hour.fillna(0) != 0).any())
                self.plan[col] = {"tipo": "fecha", "hora": hora}
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
        columnas = []
        for col in self.columnas:
            p = self.plan[col]
            s = df[col]
            if p["tipo"] == "num":
                columnas.append(_a_float32(s))
            elif p["tipo"] == "fecha":
                dt = _a_datetime(s)
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
                    codigos = pd.Categorical(texto, categories=p["categorias"]).codes.astype("float32")
                    codigos[codigos < 0] = np.nan
                    columnas.append(codigos)
                else:
                    frec = p["frecuencias"]
                    columnas.append(np.array(
                        [frec.get(t, 0.0) if t is not None else np.nan for t in texto],
                        dtype="float32"))
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
        return f"{x:,.0f}"
    if a >= 1:
        return f"{x:,.2f}"
    return f"{x:.4f}"


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
                    f"el modelo lo reduce en {mejora:.0f}%."
                )
            lineas.append(f"Explica el {max(m['r2'], 0) * 100:.0f}% de la variación (R² = {m['r2']:.3f}).")
        else:
            lineas.append(f"Acierta «{self.objetivo}» el {m['accuracy'] * 100:.1f}% de las veces en datos que no vio al entrenar.")
            lineas.append(
                f"Responder siempre la clase más frecuente acertaría el {b['accuracy'] * 100:.1f}%."
            )
            if m.get("auc") is not None:
                lineas.append(f"AUC: {m['auc']:.3f} (1 es perfecto, 0.5 es azar).")
        lineas.append(
            f"Se entrenó con {self.n_entrenamiento:,} filas y se probó con {self.n_validacion:,} "
            f"que el modelo no había visto; luego se reentrenó con todas."
        )
        return lineas


# ----------------------------------------------------------------------
# Entrenar
# ----------------------------------------------------------------------
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


def entrenar(df: pd.DataFrame, config: ConfigPrediccion, columnas: list,
             progreso=None, forzar_categoria=None) -> ResultadoEntrenamiento:
    from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                                 mean_squared_error, r2_score, roc_auc_score)
    from sklearn.model_selection import train_test_split

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
            f"La tabla es grande: se usó una muestra de {len(pos):,} filas "
            f"(ajustable en Opciones avanzadas)."
        )
    usar_fecha = config.validacion == "temporal" and config.columna_fecha in df.columns
    cols_necesarias = list(dict.fromkeys(
        columnas + [obj] + ([config.columna_fecha] if usar_fecha else [])))
    datos = df[cols_necesarias].iloc[pos].reset_index(drop=True)
    y_raw = datos[obj]

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

    # --- partir en entrenamiento / validación -------------------------
    n = len(datos)
    idx = np.arange(n)
    if usar_fecha:
        fechas = _a_datetime(datos[config.columna_fecha])
        fechas = fechas.fillna(pd.Timestamp("1900-01-01")).to_numpy()
        orden = np.argsort(fechas, kind="stable")
        n_val = max(int(n * FRACCION_VALIDACION), 1)
        idx_val, idx_tr = orden[-n_val:], orden[:-n_val]
    else:
        estratificar = None
        if tipo == "clasificacion":
            cuentas = np.bincount(yt)
            if cuentas[cuentas > 0].min() >= 2:
                estratificar = yt
        idx_tr, idx_val = train_test_split(
            idx, test_size=FRACCION_VALIDACION, random_state=config.semilla, stratify=estratificar)
    if len(idx_tr) < 10 or len(idx_val) < 5:
        raise ValueError("Hay muy pocas filas para separar una parte de prueba.")

    pre_val = Preprocesador(columnas).ajustar(datos.iloc[idx_tr][columnas], forzar_categoria)
    X_tr = pre_val.transformar(datos.iloc[idx_tr][columnas])
    X_val = pre_val.transformar(datos.iloc[idx_val][columnas])

    prog("Entrenando y comprobando el modelo…")
    modelo_a = _crear_modelo(tipo, config, pre_val.mascara_categ, len(idx_tr))
    try:
        modelo_a.fit(X_tr, yt[idx_tr])
    except ValueError as e:
        raise ValueError(f"No se pudo entrenar el modelo: {e}")

    # --- métricas ------------------------------------------------------
    metricas, baseline = {}, {}
    if tipo == "regresion":
        pred = modelo_a.predict(X_val)
        if config.log_objetivo:
            pred = np.clip(np.expm1(pred), 0, None)
        real = y[idx_val]
        metricas["rmse"] = float(np.sqrt(mean_squared_error(real, pred)))
        metricas["mae"] = float(mean_absolute_error(real, pred))
        metricas["r2"] = float(r2_score(real, pred))
        base = float(np.mean(y[idx_tr]))
        baseline["rmse"] = float(np.sqrt(np.mean((real - base) ** 2)))
        if baseline["rmse"] and metricas["rmse"] > baseline["rmse"] * 0.99:
            advertencias.append(
                "El modelo casi no mejora a «responder siempre el promedio». Revisa qué columnas "
                "marcaste como útiles, o puede que estos datos no alcancen para predecir eso."
            )
    else:
        pred = modelo_a.predict(X_val)
        real = yt[idx_val]
        metricas["accuracy"] = float(accuracy_score(real, pred))
        metricas["f1"] = float(f1_score(real, pred, average="macro"))
        frecuente = np.bincount(yt[idx_tr]).argmax()
        baseline["accuracy"] = float(np.mean(real == frecuente))
        metricas["auc"] = None
        try:
            proba = modelo_a.predict_proba(X_val)
            if len(clases) == 2 and len(np.unique(real)) == 2:
                metricas["auc"] = float(roc_auc_score(real, proba[:, 1]))
            elif len(clases) > 2 and set(np.unique(real)) <= set(modelo_a.classes_):
                metricas["auc"] = float(roc_auc_score(
                    real, proba, multi_class="ovr", labels=list(modelo_a.classes_)))
            metricas["logloss"] = float(log_loss(real, proba, labels=list(modelo_a.classes_)))
        except Exception:
            pass
        if metricas["accuracy"] <= baseline["accuracy"] + 0.005:
            advertencias.append(
                "El modelo casi no mejora a «responder siempre la clase más frecuente». Revisa qué "
                "columnas marcaste como útiles."
            )
        cuentas = np.bincount(yt)
        raras = [str(clases[i]) for i, c in enumerate(cuentas) if 0 < c < 5]
        if raras:
            advertencias.append("Hay clases con menos de 5 filas, el modelo casi no puede aprenderlas: " + ", ".join(raras[:5]) + ".")

    # --- qué columnas pesaron más -------------------------------------
    prog("Calculando en qué columnas se fijó el modelo…")
    importancias = []
    try:
        from sklearn.inspection import permutation_importance
        k = min(3000, len(idx_val))
        sub = np.random.default_rng(config.semilla).choice(len(idx_val), size=k, replace=False)
        pi = permutation_importance(modelo_a, X_val[sub], yt[idx_val][sub], n_repeats=3,
                                    random_state=config.semilla, n_jobs=1)
        por_columna = {}
        for fuente, valor in zip(pre_val.fuente, pi.importances_mean):
            por_columna[fuente] = por_columna.get(fuente, 0.0) + max(float(valor), 0.0)
        total = sum(por_columna.values())
        if total > 0:
            importancias = sorted(((c, v / total * 100) for c, v in por_columna.items()),
                                  key=lambda t: t[1], reverse=True)
    except Exception:
        importancias = []

    # --- modelo final con todos los datos -----------------------------
    prog("Entrenando el modelo final con todos los datos…")
    pre_final = Preprocesador(columnas).ajustar(datos[columnas], forzar_categoria)
    X_all = pre_final.transformar(datos[columnas])
    n_iter = max(int(getattr(modelo_a, "n_iter_", config.max_iter)), 10)
    modelo_final = _crear_modelo(tipo, config, pre_final.mascara_categ, n, n_iter=n_iter, parar_antes=False)
    modelo_final.fit(X_all, yt)

    return ResultadoEntrenamiento(
        tipo=tipo, objetivo=obj, columnas=columnas, clases=clases,
        metricas=metricas, baseline=baseline, importancias=importancias,
        n_entrenamiento=len(idx_tr), n_validacion=len(idx_val), n_iteraciones=n_iter,
        advertencias=advertencias, log_objetivo=bool(config.log_objetivo and tipo == "regresion"),
        salida=config.salida, modelo=modelo_final, preprocesador=pre_final,
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

    id_col = elegir_columna_id(df, columna_id)
    salida_df = pd.DataFrame({id_col: df[id_col].to_numpy()}) if id_col else \
        pd.DataFrame({"Id": np.arange(len(df))})

    if resultado.tipo == "regresion":
        pred = modelo.predict(X)
        if resultado.log_objetivo:
            pred = np.clip(np.expm1(pred), 0, None)
        salida_df[resultado.objetivo] = pred
        return salida_df

    clases = np.array(resultado.clases, dtype=object)
    presentes = clases[modelo.classes_]
    if salida == "probabilidad":
        proba = modelo.predict_proba(X)
        if len(presentes) == 2 and len(resultado.clases) == 2:
            salida_df[resultado.objetivo] = proba[:, 1]
        else:
            for j, etiqueta in enumerate(presentes):
                salida_df[str(etiqueta)] = proba[:, j]
    else:
        salida_df[resultado.objetivo] = presentes[np.searchsorted(modelo.classes_, modelo.predict(X))]
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