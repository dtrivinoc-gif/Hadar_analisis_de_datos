"""
formato.py — Cómo se escriben los números en Hadar (sin Qt, sin dependencias). UN solo lugar.

Dos formatos: «chileno» (1.234,5) y «estandar» (1,234.5). La pantalla lo fija al abrir un proyecto o al cambiarlo en
Ajustes (ver ajustes.py); todas las frases, tablas y PDF de Predicción lo respetan.
"""
from __future__ import annotations

FORMATO_NUMERICO = "chileno"


def set_formato_numerico(formato: str) -> None:
    global FORMATO_NUMERICO
    FORMATO_NUMERICO = "estandar" if formato == "estandar" else "chileno"


def get_formato_numerico() -> str:
    return FORMATO_NUMERICO


def loc(s: str) -> str:
    """Pasa un número escrito en formato estándar (1,234.5) al formato elegido."""
    if FORMATO_NUMERICO != "chileno":
        return s
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def dec(x, decimales: int = 1, miles: bool = False) -> str:
    """Número con `decimales` decimales (y separador de miles si `miles`), en el formato elegido.
    dec(1234.5, 1, True) -> '1.234,5' (chileno) · '1,234.5' (estándar)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v or v in (float("inf"), float("-inf")):
        return "—"
    return loc(f"{v:,.{decimales}f}" if miles else f"{v:.{decimales}f}")
