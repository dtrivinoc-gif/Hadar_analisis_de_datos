"""
pistas.py — Palabras que Hadar reconoce en los NOMBRES de columnas (español e inglés). Sin Qt, sin dependencias.

Un solo lugar para todas las pistas: cómo juntar un valor por período (sumar, promediar o tomar el pico),
qué columna suele ser «el valor» y cuál «el evento». Los nombres de columna se leen sin tildes
(«ocupación» = «ocupacion») y separando mayúsculas, guiones y guiones bajos.
"""
from __future__ import annotations

import re
import unicodedata


def _sin_tildes(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def tokens(nombre) -> list:
    """'Ocupación_Total' -> ['ocupacion', 'total'];  'cpuUsage' -> ['cpu', 'usage']."""
    s = _sin_tildes(nombre)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


# Lo que es un NIVEL en cada instante (se promedia): temperatura, saldo, precio…
NIVEL = {"stock", "nivel", "saldo", "temperatura", "temp", "precio", "promedio", "tasa", "porcentaje",
         "humedad", "presion", "voltaje", "saturacion",
         "temperature", "balance", "price", "average", "avg", "mean", "rate", "percent", "percentage",
         "humidity", "pressure", "voltage", "level", "inventory", "score", "rating"}
# Lo que importa por su PICO (se toma el máximo): carga, uso, ocupación…
PICO = {"uso", "carga", "trafico", "conexiones", "usuarios", "cpu", "memoria", "ancho", "banda", "aforo", "cola",
        "usage", "load", "traffic", "connections", "users", "memory", "bandwidth", "capacity", "queue", "peak"}
# Pueden leerse de las dos maneras: decide para qué se usa (saturación -> pico; el resto -> promedio).
AMBIGUAS = {"ocupacion", "latencia", "occupancy", "latency"}
# Lo que se ACUMULA (se suma): ventas, consumo…
SUMA = {"venta", "ventas", "monto", "total", "cantidad", "consumo", "ingreso", "ingresos", "ganancia", "ganancias",
        "costo", "costos", "gasto", "gastos", "unidades", "pedidos", "produccion", "compras",
        "sale", "sales", "revenue", "amount", "quantity", "qty", "units", "consumption", "income", "profit",
        "cost", "costs", "expense", "expenses", "orders", "production", "purchases", "count"}

# Columnas que suelen ser un PUNTAJE u orden (sirve para ordenar, no es una probabilidad calibrada)
RANKING = {"score", "puntaje", "puntuacion", "ranking", "rank", "orden", "indice", "index", "rating", "prioridad", "priority"}

# Para sugerir «el valor a seguir» (puntaje más alto = más probable)
VALOR = SUMA | {"valor", "value", "precio", "price", "temperatura", "temperature", "uso", "usage", "carga", "load",
                "stock"}
EVENTO = {"falla", "fallo", "evento", "error", "incidente", "averia", "tipo", "estado", "status", "alarma", "alerta",
          "failure", "fault", "event", "incident", "type", "state", "alarm", "alert"}
INICIO = {"inicio", "apertura", "creacion", "ingreso", "solicitud", "start", "opened", "open", "created",
          "creation", "request"}
FIN = {"fin", "cierre", "termino", "entrega", "resolucion", "end", "closed", "close", "resolved", "resolution",
       "delivery", "finish"}


def agregacion_sugerida(col_valor, defecto: str = "suma"):
    """Devuelve (agregacion, reconocida). `reconocida` es False cuando el nombre no dio ninguna pista y se
    usó el valor por defecto (el que llama debería avisarlo)."""
    if col_valor is None:
        return "cuenta", True
    t = set(tokens(col_valor))
    if t & AMBIGUAS and not (t & (NIVEL | PICO | SUMA)):
        return ("max" if defecto == "max" else "promedio"), True
    if t & NIVEL:
        return "promedio", True
    if t & PICO:
        return "max", True
    if t & AMBIGUAS:
        return ("max" if defecto == "max" else "promedio"), True
    if t & SUMA:
        return "suma", True
    return defecto, False