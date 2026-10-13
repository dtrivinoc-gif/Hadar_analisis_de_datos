"""
analisis_guardados.py — Análisis guardados con nombre en el proyecto (sin Qt).

Un análisis guardado es solo la CONFIGURACIÓN de una pregunta (ConfigPronostico.to_dict()) con un nombre elegido por la
persona: «Ventas por sucursal», «Fallas de bombas». No guarda resultados, datos ni modelos: al abrirlo se vuelve a
calcular con los datos de hoy. No es un constructor de flujos: no encadena ni programa nada.
"""
from __future__ import annotations

import pandas as pd

MAX_GUARDADOS = 30
MAX_LARGO_NOMBRE = 60


def _norm(nombre) -> str:
    return " ".join(str(nombre or "").split())


def limpiar(guardados) -> list:
    """Lee lo guardado en el proyecto; ignora entradas dañadas."""
    salida = []
    for g in guardados if isinstance(guardados, list) else []:
        if isinstance(g, dict) and isinstance(g.get("nombre"), str) and isinstance(g.get("config"), dict) and _norm(g["nombre"]):
            salida.append(g)
    return salida


def nombres(guardados) -> list:
    return [g["nombre"] for g in limpiar(guardados)]


def guardar(guardados: list, nombre, config: dict, ahora=None) -> str:
    """Guarda (o reemplaza si ya existe uno con ese nombre). Modifica `guardados` en el lugar y devuelve el nombre limpio."""
    n = _norm(nombre)
    if not n:
        raise ValueError("Escribe un nombre para el análisis.")
    if len(n) > MAX_LARGO_NOMBRE:
        raise ValueError(f"El nombre es muy largo (máximo {MAX_LARGO_NOMBRE} letras).")
    if not isinstance(config, dict):
        raise ValueError("No hay una configuración que guardar.")
    cfg = {k: v for k, v in config.items() if k not in ("historial", "guardados")}      # solo la pregunta, no lo demás
    reg = {"nombre": n, "config": cfg, "fecha": (pd.Timestamp(ahora) if ahora is not None else pd.Timestamp.now()).strftime("%Y-%m-%d %H:%M")}
    for i, g in enumerate(guardados):
        if isinstance(g, dict) and _norm(g.get("nombre")).casefold() == n.casefold():
            guardados[i] = reg
            return n
    if len(limpiar(guardados)) >= MAX_GUARDADOS:
        raise ValueError(f"Ya hay {MAX_GUARDADOS} análisis guardados: borra alguno para guardar otro.")
    guardados.append(reg)
    return n


def obtener(guardados, nombre):
    n = _norm(nombre).casefold()
    for g in limpiar(guardados):
        if _norm(g["nombre"]).casefold() == n:
            return dict(g["config"])
    return None


def borrar(guardados: list, nombre) -> bool:
    n = _norm(nombre).casefold()
    for i, g in enumerate(list(guardados)):
        if isinstance(g, dict) and _norm(g.get("nombre")).casefold() == n:
            del guardados[i]
            return True
    return False
