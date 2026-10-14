"""
historial.py — Comparación entre ejecuciones de la pestaña Predicción (sin Qt).

NO hay un modelo desplegado que se degrade: cada vez que se calcula, se vuelve a entrenar. Aun así es útil saber si
«lo mismo» hoy rinde distinto que la última vez (por ejemplo, porque llegaron datos nuevos que cambiaron de
comportamiento). Por cada análisis se guarda en el proyecto un resumen CHICO (fecha, qué se preguntó, cuánto mejoró
frente a la respuesta fácil, con cuántas filas). No se guardan modelos ni datos.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

try:
    from .pronostico_base import fecha_txt, pct_txt
except ImportError:
    from pronostico_base import fecha_txt, pct_txt

MAX_POR_ANALISIS = 10          # resúmenes que se guardan de cada análisis
MAX_TOTAL = 200                # y en total en el proyecto
CAMBIO_MEJORA_AVISAR = 0.10    # avisar si la mejora cambia en 10 puntos o más
_ORDEN_CONF = {"baja": 0, "media": 1, "alta": 2}
# Qué define que dos ejecuciones sean «el mismo análisis» (no el horizonte ni la fecha de corte: eso cambia a propósito)
CAMPOS_CLAVE = ("palabra", "tabla", "columna_fecha", "columna_valor", "columna_objetivo", "columna_evento", "valor_evento",
                "columna_entidad", "columna_inicio", "columna_fin", "columna_riesgo", "columna_impacto", "columna_driver",
                "columna_stock", "columna_grupo", "metodo_evento", "tabla_entrenamiento", "objetivo", "tipo")


def _como_dict(cfg) -> dict:
    return cfg if isinstance(cfg, dict) else cfg.to_dict()


def config_clave(cfg) -> dict:
    d = _como_dict(cfg)
    return {k: d[k] for k in CAMPOS_CLAVE if d.get(k) not in (None, "", [])}


def clave_analisis(cfg) -> str:
    return hashlib.sha1(json.dumps(config_clave(cfg), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]


def _mejora(res):
    f = getattr(res, "mejora", None)
    if callable(f):                                   # resultado de un modelo (Entrenar un modelo)
        try:
            v = float(f())
            return v if v == v else None
        except Exception:
            return None
    m = getattr(res, "metricas", None) or {}
    v = m.get("mejora")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v else None          # descarta NaN


def resumen_de(cfg, res, n_filas=None, ahora=None) -> dict:
    ahora = pd.Timestamp(ahora) if ahora is not None else pd.Timestamp.now()
    return {"fecha": ahora.strftime("%Y-%m-%d %H:%M"), "clave": clave_analisis(cfg), "palabra": getattr(res, "palabra", "") or "modelo",
            "mejora": _mejora(res), "confianza": getattr(res, "confianza", ""), "n": int(n_filas) if n_filas else None,
            "tipo_salida": getattr(res, "tipo_salida", ""), "config": config_clave(cfg)}


def limpiar(historial) -> list:
    """Lee lo guardado en el proyecto; ignora entradas dañadas o de formatos que no se entienden."""
    salida = []
    for h in historial if isinstance(historial, list) else []:
        if isinstance(h, dict) and isinstance(h.get("clave"), str) and isinstance(h.get("fecha"), str):
            salida.append(h)
    return salida


def comparar(previo: dict, nuevo: dict) -> list:
    """Avisos al comparar con la ejecución anterior del MISMO análisis."""
    avisos = []
    cuando = previo.get("fecha", "")[:10]
    try:
        cuando = fecha_txt(pd.Timestamp(cuando))
    except Exception:
        pass
    a, b = previo.get("mejora"), nuevo.get("mejora")
    if a is not None and b is not None and abs(b - a) >= CAMBIO_MEJORA_AVISAR:
        avisos.append(f"La última vez ({cuando}) el método mejoraba {pct_txt(a)} frente a la respuesta fácil; hoy mejora "
                      f"{pct_txt(b)}." + (" Si llegaron datos nuevos, puede que el comportamiento haya cambiado: conviene revisar."
                                          if b < a else ""))
    ca, cb = _ORDEN_CONF.get(previo.get("confianza")), _ORDEN_CONF.get(nuevo.get("confianza"))
    if ca is not None and cb is not None and cb < ca:
        avisos.append(f"La última vez ({cuando}) la confianza era {previo['confianza']}; hoy es {nuevo['confianza']}.")
    return avisos


def registrar_y_comparar(historial: list, cfg, res, n_filas=None, ahora=None) -> list:
    """Agrega el resumen de esta ejecución a `historial` (se modifica en el lugar) y devuelve los avisos de comparación."""
    nuevo = resumen_de(cfg, res, n_filas, ahora)
    previos = [h for h in historial if h.get("clave") == nuevo["clave"]]
    avisos = comparar(previos[-1], nuevo) if previos else []
    historial.append(nuevo)
    # se recorta: los más recientes de cada análisis y, en total, los últimos MAX_TOTAL
    cuenta, conservar = {}, []
    for h in reversed(historial):
        cuenta[h["clave"]] = cuenta.get(h["clave"], 0) + 1
        if cuenta[h["clave"]] <= MAX_POR_ANALISIS:
            conservar.append(h)
    historial[:] = list(reversed(conservar))[-MAX_TOTAL:]
    return avisos