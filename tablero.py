"""
tablero.py — Lo que la pestaña Predicción deja a la vista de las demás.

Idea (la misma que usa Palantir con su ontología): una predicción no vive
encerrada en su pestaña, es una propiedad más de cada fila. Aquí se guarda,
por tabla, una "foto" de las predicciones por fila (la que sale de
`prediccion.predecir`) para que Datos y Alarmas la lean como una columna más.

Reglas que se respetan a propósito:
  * No toca los datos del usuario: las columnas predichas se agregan solo a
    una COPIA liviana de la tabla (`con_predicciones`), nunca a `self.df`.
  * Son estimaciones, no datos: cada una lleva su fecha y se marca como
    «solo lectura» en la pantalla.
  * No se guarda ningún modelo (sin pickle): solo los números que ya salieron.
  * Sin Qt: se puede probar sola.

Hoy guarda tres cosas: las predicciones por fila (Datos y Alarmas), los
resultados de las «Preguntas» (proyección, agotamiento…) que el usuario
decide llevar al Panel («Lo que se viene»), y los resultados de comparar lo
predicho con lo real (el «circuito de retorno», ver retorno.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

SUFIJO_SI_REPETIDA = " (Hadar)"
_COL_ID = "__id__"


@dataclass
class PrediccionesPublicadas:
    tabla: str                 # tabla a la que pertenecen las filas
    datos: pd.DataFrame        # mismo índice que la tabla; solo las columnas predichas
    objetivo: str              # qué se predijo
    tipo_salida: str           # "valor" | "probabilidad" | "ranking" | "" (clase)
    calculado_en: str          # fecha y hora, en texto
    # Para poder comparar después con lo real (retorno.py):
    modo: str = ""             # regresion | clase | probabilidad | probabilidades | ranking
    ids: "pd.Series | None" = None     # identificador de cada fila (la columna id de la tabla), si había
    id_nombre: str = ""
    clase_positiva: str = ""   # a qué clase se refiere la probabilidad (caso de 2 clases)
    etiquetas: dict = field(default_factory=dict)      # {columna: clase} cuando hay una probabilidad por clase
    referencia: dict = field(default_factory=dict)     # los números de validación con que se entrenó el modelo


@dataclass
class PronosticoPublicado:
    """Lo esencial de un resultado de «Preguntas» (ver pronostico_base.ResultadoPronostico),
    en texto simple: es lo que el Panel muestra. No guarda el modelo ni la serie."""
    clave: str                 # identifica el análisis (el mismo análisis, recalculado, reemplaza al anterior)
    titulo: str                # «Agotamiento», «Proyección»…
    palabra: str
    titular: str               # la respuesta en una frase, con su rango
    nivel: str = "info"        # info | atencion | alerta
    confianza: str = "media"   # alta | media | baja
    por_que_confianza: str = ""
    tabla: str = ""
    calculado_en: str = ""
    aviso_critico: str = ""    # solo si hubo un aviso crítico (si no, vacío)
    ventanas: list = field(default_factory=list)   # fechas futuras que predice (ver pronostico_base.ventana_fechas)

    def a_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def desde_dict(cls, d):
        nombres = set(cls.__dataclass_fields__)
        d = {k: v for k, v in (d or {}).items() if k in nombres}
        if not d.get("clave"):
            return None
        return cls(**d)


_ORDEN_NIVEL = {"alerta": 0, "atencion": 1, "info": 2}
MAX_PRONOSTICOS = 12


def _nombres_de_columnas(columnas, objetivo, es_probabilidad, clases):
    """{nombre que entrega predecir: nombre que ve el usuario}."""
    columnas = list(columnas)
    if not es_probabilidad:
        return {c: f"Predicción: {objetivo}" for c in columnas[:1]}
    if len(clases or []) == 2 and columnas == [objetivo]:
        return {objetivo: f"Probabilidad de {objetivo} = {clases[1]}"}
    return {c: f"Probabilidad de {objetivo} = {c}" for c in columnas}


def alinear_salida(salida: pd.DataFrame, df_tabla: pd.DataFrame, posicional_seguro: bool = False):
    """Ordena las filas de `salida` (lo que devuelve `predecir`) igual que la
    tabla original y le pone el mismo índice.

    La primera columna de `salida` es siempre la de identificación (la real de
    la tabla o, si no hay, un «Id» numérico). Se empareja por ese id cuando se
    puede; por posición solo si `posicional_seguro` (se predijo sobre la misma
    tabla, sin cruzarla con otras, así que el orden no cambió).
    """
    if salida is None or len(salida.columns) < 2:
        raise ValueError("No hay predicciones que mostrar.")
    id_col = salida.columns[0]
    pred_cols = list(salida.columns[1:])
    por_id = (id_col in df_tabla.columns and salida[id_col].is_unique
              and df_tabla[id_col].is_unique)
    if por_id:
        pos = pd.Index(salida[id_col]).get_indexer(df_tabla[id_col])
        if (pos < 0).any():
            raise ValueError(
                f"{int((pos < 0).sum())} filas de la tabla no aparecen en las predicciones "
                f"(por «{id_col}»). Vuelve a generar las predicciones sobre esta misma tabla.")
        out = salida[pred_cols].iloc[pos]
    elif len(salida) == len(df_tabla) and (posicional_seguro or id_col not in df_tabla.columns):
        out = salida[pred_cols]
    else:
        raise ValueError(
            "No pude emparejar cada predicción con su fila: la columna que identifica las filas "
            f"(«{id_col}») se repite o no coincide. Elige otra columna id en la pestaña Predicción.")
    out = out.copy()
    out.index = df_tabla.index
    return out


class Tablero:
    """Predicciones por fila, publicadas por tabla."""

    def __init__(self):
        self._pred: dict[str, PrediccionesPublicadas] = {}
        self._pron: dict[str, PronosticoPublicado] = {}
        self._ret: dict[str, dict] = {}      # {tabla: última comparación con lo real}

    # ------------------------------------------------------------------ publicar
    def publicar_predicciones(self, tabla, df_tabla, salida, objetivo, *, es_probabilidad=False,
                              clases=None, tipo_salida="", posicional_seguro=False,
                              calculado_en=None, referencia=None) -> PrediccionesPublicadas:
        """Guarda las predicciones de `tabla`. Lanza ValueError (con un mensaje
        en palabras simples) si no se pueden emparejar con sus filas."""
        alineada = alinear_salida(salida, df_tabla, posicional_seguro)
        nombres = _nombres_de_columnas(alineada.columns, objetivo, es_probabilidad, clases)
        originales = list(nombres)
        alineada = alineada[originales].rename(columns=nombres)
        existentes = {str(c) for c in df_tabla.columns}
        alineada.columns = [c + SUFIJO_SI_REPETIDA if c in existentes else c for c in alineada.columns]
        binaria = es_probabilidad and len(clases or []) == 2 and len(originales) == 1
        if tipo_salida == "valor":
            modo = "regresion"
        elif es_probabilidad:
            modo = ("ranking" if tipo_salida == "ranking" else "probabilidad") if binaria else "probabilidades"
        else:
            modo = "clase"
        id_col = salida.columns[0]
        pub = PrediccionesPublicadas(
            tabla=tabla, datos=alineada, objetivo=str(objetivo), tipo_salida=tipo_salida or "",
            calculado_en=calculado_en or datetime.now().strftime("%d/%m/%Y %H:%M"),
            modo=modo,
            ids=df_tabla[id_col].copy() if id_col in df_tabla.columns else None,
            id_nombre=str(id_col) if id_col in df_tabla.columns else "",
            clase_positiva=str(clases[1]) if binaria else "",
            etiquetas=({c: str(o) for c, o in zip(alineada.columns, originales)}
                       if es_probabilidad and not binaria else {}),
            referencia={str(k): float(v) for k, v in (referencia or {}).items() if v is not None})
        self._pred[tabla] = pub
        return pub

    # ------------------------------------------------------------------ leer
    def hay(self, tabla=None) -> bool:
        """¿Hay predicciones por fila (para `tabla`, o para cualquiera)?"""
        return bool(self._pred) if tabla is None else tabla in self._pred

    def tablas(self):
        return list(self._pred)

    def columnas(self, tabla) -> list:
        p = self._pred.get(tabla)
        return [] if p is None else list(p.datos.columns)

    def desactualizada(self, df, tabla) -> bool:
        """True si la tabla ya no tiene las mismas filas que cuando se predijo."""
        p = self._pred.get(tabla)
        return p is not None and df is not None and not df.index.equals(p.datos.index)

    def descripciones(self, df, tabla) -> dict:
        """{columna: texto para el tooltip del encabezado}."""
        p = self._pred.get(tabla)
        if p is None:
            return {}
        texto = (f"Estimación de Hadar, no es un dato real. Calculada el {p.calculado_en} con el "
                 f"modelo de la pestaña Predicción (qué se predijo: {p.objetivo}). Solo lectura.")
        if self.desactualizada(df, tabla):
            texto += "\n\nOjo: las filas de la tabla cambiaron desde entonces."
        return {c: texto for c in p.datos.columns}

    def con_predicciones(self, df, tabla):
        """`df` más las columnas predichas, en una copia liviana: `df` no se
        modifica. Si no hay predicciones para `tabla` (o no se pueden alinear)
        devuelve el mismo `df`."""
        p = self._pred.get(tabla)
        if p is None or df is None or p.datos.empty:
            return df
        extra = p.datos
        if df.index.equals(extra.index):
            valores = {c: extra[c].to_numpy() for c in extra.columns}
        elif df.index.is_unique and extra.index.is_unique:
            alin = extra.reindex(df.index)
            valores = {c: alin[c].to_numpy() for c in alin.columns}
        else:
            return df
        out = df.copy(deep=False)
        for c, v in valores.items():
            if c not in out.columns:
                out[c] = v
        return out

    # ------------------------------------------------------------------ pronósticos (Preguntas)
    def publicar_pronostico(self, clave, r, tabla="", calculado_en=None, aviso_critico="") -> PronosticoPublicado:
        """Guarda lo esencial del resultado `r` (un ResultadoPronostico). Si ya había uno
        con la misma `clave` (el mismo análisis, recalculado) lo reemplaza."""
        if not clave:
            raise ValueError("Falta el nombre del análisis.")
        if clave not in self._pron and len(self._pron) >= MAX_PRONOSTICOS:
            raise ValueError(f"Ya hay {MAX_PRONOSTICOS} pronósticos en el Panel. Quita alguno para agregar otro.")
        p = PronosticoPublicado(
            clave=str(clave), titulo=str(getattr(r, "titulo", "") or ""), palabra=str(getattr(r, "palabra", "") or ""),
            titular=str(getattr(r, "titular", "") or ""), nivel=str(getattr(r, "nivel", "info") or "info"),
            confianza=str(getattr(r, "confianza", "media") or "media"),
            por_que_confianza=str(getattr(r, "por_que_confianza", "") or ""), tabla=str(tabla or ""),
            calculado_en=calculado_en or datetime.now().strftime("%d/%m/%Y %H:%M"),
            aviso_critico=str(aviso_critico or ""),
            ventanas=[dict(v) for v in (getattr(r, "rango_fechas", None) or [])
                      if isinstance(v, dict) and v.get("desde")])
        self._pron[p.clave] = p
        return p

    def pronosticos(self, tabla=None) -> list:
        """Los publicados, los más urgentes primero (alerta, atención, info); a igual urgencia, en el
        orden en que se agregaron. Con `tabla`, solo los de esa tabla."""
        lista = [p for p in self._pron.values() if tabla is None or p.tabla == tabla]
        return sorted(lista, key=lambda p: _ORDEN_NIVEL.get(p.nivel, 3))

    def ventanas_futuras(self, tabla=None) -> list:
        """Las fechas futuras que predicen los pronósticos publicados (para dibujarlas, por ejemplo en la
        Línea de Tiempo): una por elemento, con el título, el nivel y la tabla del pronóstico al que pertenecen."""
        out = []
        for p in self.pronosticos(tabla):
            for v in p.ventanas:
                out.append({**v, "titulo": p.titulo or p.palabra, "nivel": p.nivel, "tabla": p.tabla, "clave": p.clave})
        return out

    def quitar_pronostico(self, clave):
        self._pron.pop(clave, None)

    # ------------------------------------------------------------------ retorno (predicho vs real)
    def publicacion(self, tabla):
        """Las predicciones por fila de `tabla` (o None)."""
        return self._pred.get(tabla)

    def registrar_retorno(self, tabla, resultado: dict):
        """Guarda la última comparación con lo real de `tabla` (la anterior se reemplaza)."""
        self._ret[tabla] = dict(resultado)

    def retorno(self, tabla):
        return self._ret.get(tabla)

    def retornos(self) -> list:
        """Las comparaciones guardadas, las peores primero (alerta, atención, info, bien)."""
        orden = {"alerta": 0, "atencion": 1, "info": 2, "bien": 3}
        return sorted(self._ret.values(), key=lambda d: orden.get(d.get("nivel"), 4))

    def quitar_retorno(self, tabla):
        self._ret.pop(tabla, None)

    def retornos_a_lista(self) -> list:
        return list(self._ret.values())

    def pronosticos_a_lista(self) -> list:
        return [p.a_dict() for p in self._pron.values()]

    def firma(self):
        """Algo liviano y estable que cambia cuando cambian las predicciones;
        sirve para saber si hay cambios sin guardar en el proyecto."""
        return ([(t, p.calculado_en, p.datos.shape) for t, p in sorted(self._pred.items())]
                + [(c, p.calculado_en) for c, p in sorted(self._pron.items())]
                + [("retorno", t, d.get("comparado_en")) for t, d in sorted(self._ret.items())])

    # ------------------------------------------------------------------ olvidar
    def quitar(self, tabla):
        """Olvida todo lo que se publicó sobre `tabla` (sus datos cambiaron)."""
        self._pred.pop(tabla, None)
        self._ret.pop(tabla, None)
        for c in [c for c, p in self._pron.items() if p.tabla == tabla]:
            del self._pron[c]

    def limpiar(self):
        self._pred.clear()
        self._pron.clear()
        self._ret.clear()

    # ------------------------------------------------------------------ proyecto
    def a_meta(self) -> dict:
        """Lo liviano (texto) para proyecto.json; los números van aparte, en parquet."""
        return {t: {"objetivo": p.objetivo, "tipo_salida": p.tipo_salida,
                    "calculado_en": p.calculado_en, "columnas": list(p.datos.columns),
                    "modo": p.modo, "id_nombre": p.id_nombre, "clase_positiva": p.clase_positiva,
                    "etiquetas": dict(p.etiquetas), "referencia": dict(p.referencia)}
                for t, p in self._pred.items()}

    def frames(self) -> dict:
        """Los números predichos por tabla. Si había una columna que identifica cada fila, va aparte
        como «__id__» (se separa de nuevo al abrir el proyecto)."""
        out = {}
        for t, p in self._pred.items():
            d = p.datos
            if p.ids is not None:
                d = d.copy()
                d[_COL_ID] = p.ids.to_numpy()
            out[t] = d
        return out

    @classmethod
    def desde_proyecto(cls, meta, frames, pronosticos=None, retornos=None):
        """Rearma el tablero de un proyecto guardado. Los proyectos antiguos no
        traen nada y quedan con el tablero vacío."""
        tb = cls()
        for d in (pronosticos or []):
            p = PronosticoPublicado.desde_dict(d) if isinstance(d, dict) else None
            if p is not None:
                tb._pron[p.clave] = p
        for d in (retornos or []):
            if isinstance(d, dict) and d.get("tabla"):
                tb._ret[d["tabla"]] = d
        for tabla, m in (meta or {}).items():
            datos = (frames or {}).get(tabla)
            if datos is None:
                continue
            ids = None
            if _COL_ID in datos.columns:
                datos = datos.copy()
                ids = datos.pop(_COL_ID)
            tb._pred[tabla] = PrediccionesPublicadas(
                tabla=tabla, datos=datos, objetivo=str(m.get("objetivo", "")),
                tipo_salida=m.get("tipo_salida", ""), calculado_en=m.get("calculado_en", ""),
                modo=m.get("modo", ""), ids=ids, id_nombre=m.get("id_nombre", ""),
                clase_positiva=m.get("clase_positiva", ""), etiquetas=dict(m.get("etiquetas") or {}),
                referencia=dict(m.get("referencia") or {}))
        return tb