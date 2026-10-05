"""
relacional.py — Cruza las tablas relacionadas de un proyecto para que la
pestaña Predicción pueda usar información de TODA la base, no solo de una
tabla (sin Qt, igual que prediccion.py).

Usa las relaciones que detecta la ontología (ontologia.py) y también
relaciones que el usuario agregue a mano. Hay dos formas de traer datos de
otra tabla, y Hadar elige sola cuál corresponde mirando los datos:

  1. "Cada fila apunta a una fila" (ej. cada venta -> un producto): los datos
     del producto simplemente se agregan a la fila de la venta. Es seguro.
     Se puede encadenar: venta -> producto -> proveedor.

  2. "Una fila tiene muchas" (ej. cada cliente -> muchas compras): no se puede
     pegar todo, así que se RESUME (cantidad, suma, promedio, mínimo y máximo
     de las columnas numéricas y, con fechas, también hace cuánto fue lo último
     y cuánto ocurrió en los últimos días). Aquí hay un riesgo que Hadar cuida: si el
     resumen incluye compras POSTERIORES a la fila que se quiere predecir,
     el modelo "se entera del futuro" y el resultado sale falsamente bueno.
     Por eso, si la tabla base tiene una columna de fecha y la tabla hija
     también, cada fila se resume SOLO con lo ocurrido ANTES de su fecha
     (estrictamente antes: lo del mismo día no entra).

Reglas de seguridad:
- Nunca se duplican filas de la tabla base: sale con las mismas filas, en
  el mismo orden.
- No se recorren ciclos (A -> B -> A).
- Con fecha, los resúmenes de más de un salto no se hacen (no se puede
  garantizar "solo lo anterior"); los saltos "cada fila apunta a una fila"
  sí se encadenan siempre.
- Si no hay fechas, se resume todo el historial y se avisa del riesgo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from .prediccion import _a_datetime, es_columna_id, normalizar_clave, tipo_columna
except ImportError:  # pruebas sueltas fuera del paquete
    from prediccion import _a_datetime, es_columna_id, normalizar_clave, tipo_columna

MAX_COLS_AGG = 8            # columnas numéricas que se resumen por tabla hija
MAX_COLS_POR_TABLA = 60     # columnas que se traen de una tabla "una fila"
MAX_COLUMNAS_NUEVAS = 200
MAX_PROFUNDIDAD = 3
SIN_FECHA = "__ninguna__"


# ----------------------------------------------------------------------
# Relaciones entre tablas
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Enlace:
    tabla_a: str
    col_a: str
    tabla_b: str
    col_b: str
    manual: bool = False

    def clave(self) -> str:
        return "||".join(sorted([f"{self.tabla_a}.{self.col_a}", f"{self.tabla_b}.{self.col_b}"]))

    def texto(self) -> str:
        return f"{self.tabla_a}.{self.col_a} ↔ {self.tabla_b}.{self.col_b}"

    def a_dict(self) -> dict:
        return {"tabla_a": self.tabla_a, "col_a": self.col_a, "tabla_b": self.tabla_b,
                "col_b": self.col_b, "manual": self.manual}

    @staticmethod
    def de_dict(d: dict) -> "Enlace":
        return Enlace(str(d["tabla_a"]), str(d["col_a"]), str(d["tabla_b"]), str(d["col_b"]),
                      bool(d.get("manual", True)))

    def desde(self, tabla: str):
        """(columna propia, otra tabla, columna de la otra) mirado desde `tabla`."""
        if tabla == self.tabla_a and tabla != self.tabla_b:
            return self.col_a, self.tabla_b, self.col_b
        if tabla == self.tabla_b and tabla != self.tabla_a:
            return self.col_b, self.tabla_a, self.col_a
        return None


def enlaces_desde_ontologia(relaciones) -> list:
    """Convierte las RelacionSugerida de ontologia.py en Enlace."""
    salida = []
    for r in relaciones or []:
        if r.tabla_origen != r.tabla_destino:
            salida.append(Enlace(r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino))
    return _sin_repetidos(salida)


def _sin_repetidos(enlaces) -> list:
    vistos, salida = set(), []
    for e in enlaces:
        if e.clave() not in vistos:
            vistos.add(e.clave())
            salida.append(e)
    return salida


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def _fecha_ns(serie: pd.Series) -> pd.Series:
    dt = _a_datetime(serie).reset_index(drop=True)
    try:
        return dt.astype("datetime64[ns]")
    except Exception:
        return pd.to_datetime(dt, errors="coerce").astype("datetime64[ns]")


def _codigos(clave_base: pd.Series, clave_otra: pd.Series):
    """Convierte las dos columnas clave en códigos enteros comparables
    (-1 = vacío). Así 1, 1.0 y '1' coinciden y la unión es rápida."""
    kb = normalizar_clave(clave_base).astype(object).reset_index(drop=True)
    ko = normalizar_clave(clave_otra).astype(object).reset_index(drop=True)
    codigos, _ = pd.factorize(pd.concat([kb, ko], ignore_index=True))
    codigos = codigos.astype("int64")
    return codigos[:len(kb)], codigos[len(kb):]


def _es_unica(codigos_otra: np.ndarray) -> bool:
    validos = codigos_otra[codigos_otra >= 0]
    return len(validos) == len(np.unique(validos))


def _columnas_numericas(df: pd.DataFrame, excluir) -> list:
    salida = []
    for c in df.columns:
        if c in excluir or es_columna_id(c):
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            continue
        if s.nunique(dropna=True) > 1:
            salida.append(c)
        if len(salida) >= MAX_COLS_AGG:
            break
    return salida


def _fecha_de_tabla(df: pd.DataFrame, excluir=()):
    """Primera columna de fecha de una tabla (prefiere nombres tipo 'fecha'/'date')."""
    candidatas = [c for c in df.columns if c not in excluir
                  and tipo_columna(df[c]) in ("fecha", "fecha_texto")]
    if not candidatas:
        return None
    for c in candidatas:
        if any(p in str(c).lower() for p in ("fecha", "date", "dia", "día")):
            return c
    return candidatas[0]


# ----------------------------------------------------------------------
# Las dos formas de traer datos
# ----------------------------------------------------------------------
def _traer_una_fila(codigos_base, otra, codigos_otra, columnas, prefijo) -> pd.DataFrame:
    validos = np.flatnonzero(codigos_otra >= 0)
    sub = otra.iloc[validos][columnas].copy()
    sub.index = codigos_otra[validos]
    alineado = sub.reindex(codigos_base).reset_index(drop=True)
    alineado.columns = [f"{prefijo}{c}" for c in columnas]
    return alineado


def _resumen_total(codigos_base, otra, codigos_otra, nums, prefijo) -> pd.DataFrame:
    validos = np.flatnonzero(codigos_otra >= 0)
    h = otra.iloc[validos][nums].apply(pd.to_numeric, errors="coerce").astype("float64")
    k = codigos_otra[validos]
    g = h.groupby(k)
    cuenta = pd.Series(g.size(), name="cantidad")
    salida = pd.DataFrame({f"{prefijo} (cantidad)": cuenta})
    if nums:
        agg = g.agg(["sum", "mean", "min", "max"])
        nombres = {"sum": "suma", "mean": "promedio", "min": "mínimo", "max": "máximo"}
        for c in nums:
            for est, etiqueta in nombres.items():
                salida[f"{prefijo}.{c} ({etiqueta})"] = agg[(c, est)]
    res = salida.reindex(codigos_base).reset_index(drop=True)
    res[f"{prefijo} (cantidad)"] = res[f"{prefijo} (cantidad)"].fillna(0)
    return res


def _ventanas_de(span_dias: float) -> list:
    """Ventanas «lo ocurrido en los últimos…» según cuánto tiempo abarcan los
    datos de la tabla hija (así sirve igual para datos de días que de años)."""
    if span_dias >= 90:
        return [(7, "últimos 7 días"), (30, "últimos 30 días")]
    if span_dias >= 14:
        return [(2, "últimos 2 días"), (7, "últimos 7 días")]
    if span_dias >= 2:
        return [(0.5, "últimas 12 horas"), (1, "último día")]
    return []


def _acumulado_antes(izq, acum, cols, n, dias=None) -> pd.DataFrame:
    """Acumulado de `cols` justo ANTES de (fecha de cada fila base - dias).
    Con dias=None es «antes de la fecha de la fila»."""
    esq = izq[["__k", "__t", "__pos"]].copy()
    if dias:
        esq["__t"] = esq["__t"] - pd.Timedelta(days=dias)
    esq = esq.sort_values("__t", kind="stable")
    u = pd.merge_asof(esq, acum[["__k", "__t"] + cols].sort_values("__t", kind="stable"),
                      on="__t", by="__k", allow_exact_matches=False, direction="backward")
    return u.set_index("__pos").reindex(range(n))[cols].reset_index(drop=True)


def _resumen_previo(codigos_base, fechas_base, otra, codigos_otra, fechas_otra, nums, prefijo) -> pd.DataFrame:
    """Para cada fila base, resume SOLO las filas de la tabla hija con fecha
    estrictamente anterior (acumulados por clave + unión 'hasta esa fecha').
    Además de cantidad/suma/promedio/mínimo/máximo, agrega:
      - recencia: días desde el último evento anterior,
      - ventanas: cantidad (y suma) de lo ocurrido en los últimos N días,
        con N elegido según el rango de fechas de la tabla hija.
    Todo mira estrictamente hacia atrás, así que no se «ve el futuro»."""
    n = len(codigos_base)
    validos = np.flatnonzero((codigos_otra >= 0) & fechas_otra.notna().to_numpy())
    h = otra.iloc[validos][nums].apply(pd.to_numeric, errors="coerce").astype("float64").reset_index(drop=True)
    h["__k"] = codigos_otra[validos].astype("int64")
    h["__t"] = fechas_otra.to_numpy()[validos]
    h = h.sort_values("__t", kind="stable").reset_index(drop=True)
    g_k = h["__k"]
    acum = pd.DataFrame({"__k": g_k, "__t": h["__t"], "__tprev": h["__t"],
                         "__cant": h.groupby("__k", sort=False).cumcount() + 1})
    nombres_cols = {}
    nums_ventana = nums[:3]
    if nums:
        valores = h[nums]
        cuenta_ok = valores.notna().astype("float64").groupby(g_k).cumsum()
        suma = valores.fillna(0.0).groupby(g_k).cumsum()
        minimo = valores.groupby(g_k).cummin().groupby(g_k).ffill()
        maximo = valores.groupby(g_k).cummax().groupby(g_k).ffill()
        for c in nums:
            hay = cuenta_ok[c] > 0
            acum[f"{c}|suma"] = suma[c].where(hay)
            acum[f"{c}|prom"] = (suma[c] / cuenta_ok[c]).where(hay)
            acum[f"{c}|min"] = minimo[c]
            acum[f"{c}|max"] = maximo[c]
            nombres_cols[f"{c}|suma"] = f"{prefijo}.{c} (suma anterior)"
            nombres_cols[f"{c}|prom"] = f"{prefijo}.{c} (promedio anterior)"
            nombres_cols[f"{c}|min"] = f"{prefijo}.{c} (mínimo anterior)"
            nombres_cols[f"{c}|max"] = f"{prefijo}.{c} (máximo anterior)"
        for c in nums_ventana:
            acum[f"{c}|cs"] = suma[c]          # suma acumulada sin vacíos (para las ventanas)
    nombres_cols["__cant"] = f"{prefijo} (cantidad anterior)"
    nombres_cols["__tprev"] = "__tprev"

    ok = np.flatnonzero((codigos_base >= 0) & fechas_base.notna().to_numpy())
    izq = pd.DataFrame({
        "__k": codigos_base[ok].astype("int64"),
        "__t": fechas_base.to_numpy()[ok],
        "__pos": ok,
    }).sort_values("__t", kind="stable")
    union = pd.merge_asof(izq, acum.drop(columns=[c for c in acum.columns if c.endswith("|cs")])
                          .sort_values("__t", kind="stable"), on="__t", by="__k",
                          allow_exact_matches=False, direction="backward")
    union = union.set_index("__pos").reindex(range(n))
    res = union[[c for c in nombres_cols if c != "__tprev"]].rename(columns=nombres_cols).reset_index(drop=True)
    res[nombres_cols["__cant"]] = res[nombres_cols["__cant"]].fillna(0)

    # --- recencia: días desde el último evento anterior -----------------
    ult = pd.to_datetime(union["__tprev"]).reset_index(drop=True)
    t_fila = pd.Series(pd.to_datetime(fechas_base.to_numpy()), index=range(n))
    res[f"{prefijo} (días desde la última)"] = ((t_fila - ult).dt.total_seconds() / 86400.0).clip(lower=0)

    # --- ventanas: lo ocurrido en los últimos N días --------------------
    if len(acum):
        span = float((acum["__t"].max() - acum["__t"].min()).total_seconds() / 86400.0)
        ventanas = _ventanas_de(span)
        if ventanas:
            cols_cum = ["__cant"] + [f"{c}|cs" for c in nums_ventana]
            fin = _acumulado_antes(izq, acum, cols_cum, n)
            for i, (dias, etq) in enumerate(ventanas):
                ini = _acumulado_antes(izq, acum, cols_cum, n, dias)
                res[f"{prefijo} (cantidad, {etq})"] = (fin["__cant"].fillna(0) - ini["__cant"].fillna(0))
                if i == len(ventanas) - 1:
                    for c in nums_ventana:
                        res[f"{prefijo}.{c} (suma, {etq})"] = (
                            fin[f"{c}|cs"].fillna(0) - ini[f"{c}|cs"].fillna(0))
    return res


# ----------------------------------------------------------------------
# Ampliar una tabla (recursivo, con límite de saltos)
# ----------------------------------------------------------------------
class _Contexto:
    def __init__(self, tablas, enlaces, fecha_base, base):
        self.tablas = tablas
        self.enlaces = enlaces
        self.fecha_base = fecha_base
        self.base = base
        self.advertencias = []
        self.info = {}
        self.claves_base = set()

    def avisar(self, texto):
        if texto not in self.advertencias:
            self.advertencias.append(texto)


def _ampliar(nombre, prof, camino, ctx: _Contexto, fechas_base=None) -> pd.DataFrame:
    """Devuelve la tabla `nombre` (índice 0..n-1) con las columnas nuevas
    traídas de las tablas vecinas. `prof` = saltos que quedan."""
    df = ctx.tablas[nombre].reset_index(drop=True)
    es_base = (nombre == ctx.base and len(camino) == 0)
    if prof <= 0:
        return df
    nuevas = []
    prefijos_usados = set()

    for enlace in ctx.enlaces:
        lado = enlace.desde(nombre)
        if lado is None:
            continue
        col_propia, otra, col_otra = lado
        if otra in camino or otra == nombre or otra not in ctx.tablas:
            continue
        if col_propia not in df.columns or col_otra not in ctx.tablas[otra].columns:
            continue

        otra_df = _ampliar(otra, prof - 1, camino + [nombre], ctx) if prof > 1 else ctx.tablas[otra].reset_index(drop=True)
        cb, co = _codigos(df[col_propia], otra_df[col_otra])
        if es_base:
            ctx.claves_base.add(col_propia)
        unica = _es_unica(co)

        base_pref = otra if otra not in prefijos_usados else f"{otra}[{col_propia}]"
        prefijos_usados.add(otra)

        if unica:
            traer = [c for c in otra_df.columns if c != col_otra and not es_columna_id(c)][:MAX_COLS_POR_TABLA]
            if not traer:
                continue
            bloque = _traer_una_fila(cb, otra_df, co, traer, f"{base_pref}.")
            for c in bloque.columns:
                ctx.info[c] = f"Viene de «{otra}» (unida por {col_propia})"
            nuevas.append(bloque)
            continue

        # --- "una fila tiene muchas": hay que resumir ---------------------
        usa_fecha = ctx.fecha_base is not None
        if usa_fecha and not es_base:
            ctx.avisar(f"Se omitió resumir «{otra}» desde «{nombre}»: con fecha, solo se resumen "
                       f"tablas que cuelgan directamente de la tabla base (para no mezclar el futuro).")
            continue
        nums = _columnas_numericas(otra_df, {col_otra})
        fecha_otra = _fecha_de_tabla(otra_df, excluir={col_otra}) if usa_fecha else None
        if usa_fecha and fecha_otra is None:
            ctx.avisar(f"«{otra}» no tiene columna de fecha: se resumió todo su historial, que puede "
                       f"incluir datos posteriores a cada fila (el modelo podría 'ver el futuro').")
        if not usa_fecha:
            ctx.avisar("Sin columna de fecha, las tablas con muchas filas por fila se resumieron con todo "
                       "su historial: puede incluir información posterior a cada fila (riesgo de 'ver el futuro').")
        if not (cb >= 0).any() or not (co >= 0).any():
            continue
        if usa_fecha and fecha_otra is not None:
            bloque = _resumen_previo(cb, fechas_base, otra_df, co, _fecha_ns(otra_df[fecha_otra]), nums, base_pref)
            nota = f"Resumen de «{otra}» por {col_propia}, solo con lo anterior a la fecha de cada fila"
        else:
            bloque = _resumen_total(cb, otra_df, co, nums, base_pref)
            nota = f"Resumen de «{otra}» por {col_propia} (todo el historial)"
        if not _es_unica(cb) and not unica:
            ctx.avisar(f"La relación entre «{nombre}» y «{otra}» es de muchos a muchos: el resumen es aproximado.")
        for c in bloque.columns:
            ctx.info[c] = nota
        nuevas.append(bloque)

    if not nuevas:
        return df
    return pd.concat([df] + nuevas, axis=1)


@dataclass
class Ampliacion:
    df: pd.DataFrame
    info: dict                 # {columna nueva: de dónde viene}
    advertencias: list
    columnas_nuevas: list
    claves_base: set           # columnas de la tabla base que son claves de unión


def construir_tabla_ampliada(tablas: dict, base: str, enlaces: list, profundidad: int = 2,
                             fecha_base: str | None = None) -> Ampliacion:
    """La tabla `base` con las mismas filas y el mismo orden, más columnas
    traídas de las tablas relacionadas (ver explicación arriba)."""
    if base not in tablas:
        raise ValueError(f"No existe la tabla «{base}».")
    original = tablas[base]
    if fecha_base is not None and fecha_base not in original.columns:
        raise ValueError(f"La tabla «{base}» necesita la columna de fecha «{fecha_base}» "
                         f"para resumir las tablas relacionadas.")
    profundidad = int(min(max(profundidad, 1), MAX_PROFUNDIDAD))
    ctx = _Contexto(tablas, _sin_repetidos(enlaces), fecha_base, base)
    fechas = _fecha_ns(original[fecha_base]) if fecha_base else None
    amp = _ampliar(base, profundidad, [], ctx, fechas_base=fechas)

    if len(amp) != len(original):
        raise RuntimeError("Error interno: el cruce de tablas cambió la cantidad de filas.")
    nuevas = [c for c in amp.columns[len(original.columns):]]
    if len(nuevas) > MAX_COLUMNAS_NUEVAS:
        ctx.avisar(f"Se trajeron demasiadas columnas ({len(nuevas)}); se conservaron las primeras {MAX_COLUMNAS_NUEVAS}.")
        nuevas = nuevas[:MAX_COLUMNAS_NUEVAS]
        amp = amp[list(original.columns) + nuevas]
    return Ampliacion(df=amp, info={c: ctx.info.get(c, "") for c in nuevas},
                      advertencias=ctx.advertencias, columnas_nuevas=nuevas,
                      claves_base=set(ctx.claves_base))


# ----------------------------------------------------------------------
# Punto de entrada para la pestaña Predicción
# ----------------------------------------------------------------------
@dataclass
class PreparacionRelacional:
    df_entrenamiento: pd.DataFrame
    df_prueba: pd.DataFrame | None
    info: dict
    advertencias: list
    columnas_nuevas: list
    claves_base: set
    fecha_usada: str | None
    enlaces: list


def enlaces_disponibles(tablas: dict, relaciones_ontologia, config) -> list:
    """Todas las relaciones conocidas (ontología + manuales), sin las que
    involucran a la tabla de prueba (esa tabla no es una fuente de datos)."""
    lista = enlaces_desde_ontologia(relaciones_ontologia)
    for d in (config.enlaces_manuales or []):
        try:
            lista.append(Enlace.de_dict(d))
        except Exception:
            pass
    prueba = config.tabla_prueba
    return [e for e in _sin_repetidos(lista)
            if e.tabla_a in tablas and e.tabla_b in tablas
            and (prueba is None or (e.tabla_a != prueba and e.tabla_b != prueba))]


def elegir_fecha_base(df: pd.DataFrame, config):
    if config.fecha_relaciones == SIN_FECHA:
        return None
    if config.fecha_relaciones in df.columns:
        return config.fecha_relaciones
    if config.columna_fecha in df.columns and tipo_columna(df[config.columna_fecha]) in ("fecha", "fecha_texto"):
        return config.columna_fecha
    return _fecha_de_tabla(df)


def preparar(tablas: dict, config, relaciones_ontologia) -> PreparacionRelacional:
    """Arma la tabla de entrenamiento (y la de prueba, con el mismo cruce)
    ya ampliadas con las tablas relacionadas."""
    train = config.tabla_entrenamiento
    prueba = config.tabla_prueba
    if train not in tablas:
        raise ValueError("Elige la tabla de entrenamiento.")
    desactivados = set(config.enlaces_desactivados or [])
    enlaces = [e for e in enlaces_disponibles(tablas, relaciones_ontologia, config)
               if e.clave() not in desactivados]
    fecha = elegir_fecha_base(tablas[train], config)
    amp = construir_tabla_ampliada(tablas, train, enlaces, config.profundidad_relaciones, fecha)

    df_prueba, advertencias = None, list(amp.advertencias)
    if prueba and prueba in tablas:
        # Las relaciones de la tabla de entrenamiento se aplican igual a la de prueba.
        mapeados = []
        for e in enlaces:
            a = prueba if e.tabla_a == train else e.tabla_a
            b = prueba if e.tabla_b == train else e.tabla_b
            mapeados.append(Enlace(a, e.col_a, b, e.col_b, e.manual))
        # Si la tabla de prueba no trae una columna clave, esa relación no se puede aplicar.
        utiles = [e for e in _sin_repetidos(mapeados)
                  if not (e.tabla_a == prueba and e.col_a not in tablas[prueba].columns)
                  and not (e.tabla_b == prueba and e.col_b not in tablas[prueba].columns)]
        # La tabla de entrenamiento no es fuente de datos al ampliar la de prueba.
        tablas_prueba = {n: d for n, d in tablas.items() if n != train}
        utiles = [e for e in utiles if e.tabla_a in tablas_prueba and e.tabla_b in tablas_prueba]
        amp_p = construir_tabla_ampliada(tablas_prueba, prueba, utiles, config.profundidad_relaciones, fecha)
        df_prueba = amp_p.df
        for a in amp_p.advertencias:
            if a not in advertencias:
                advertencias.append(a)
        faltan = [c for c in amp.columnas_nuevas if c not in df_prueba.columns]
        if faltan:
            advertencias.append("La tabla de prueba no pudo recibir " + str(len(faltan)) +
                                " columna(s) traídas de otras tablas (le falta una columna clave); "
                                "se marcarán como no disponibles.")
            for c in faltan:
                df_prueba[c] = np.nan
    return PreparacionRelacional(
        df_entrenamiento=amp.df, df_prueba=df_prueba, info=amp.info, advertencias=advertencias,
        columnas_nuevas=amp.columnas_nuevas, claves_base=amp.claves_base,
        fecha_usada=fecha, enlaces=enlaces)