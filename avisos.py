"""
avisos.py — Avisos con nivel de importancia (sin Qt, sin dependencias).

Tres niveles:  crítico (el resultado puede estar equivocado: p. ej. una columna que ya trae la respuesta),
               importante (conviene saberlo antes de confiar: calibración, duplicados, extrapolación…),
               informativo (contexto: qué se asumió, qué se descartó).
COMPATIBILIDAD: un aviso sigue pudiendo ser un string suelto, y un string suelto es INFORMATIVO. Para dar nivel se usa
`aviso(texto, nivel)` (un `str` con el atributo `.nivel`: todo lo que ya trataba avisos como texto sigue funcionando)
o se registra una frase en `NIVEL_POR_FRASE` (para los avisos que ya existían como texto).
Orden: críticos primero, luego importantes, luego informativos; dentro de cada nivel se conserva el orden original.
"""
from __future__ import annotations

CRITICO, IMPORTANTE, INFORMATIVO = "critico", "importante", "informativo"
_RANGO = {CRITICO: 0, IMPORTANTE: 1, INFORMATIVO: 2}
ETIQUETA = {CRITICO: "Crítico", IMPORTANTE: "Importante", INFORMATIVO: "Nota"}


class Aviso(str):
    """Un texto de aviso que además sabe su nivel. Se comporta como cualquier string."""
    nivel = INFORMATIVO

    def __new__(cls, texto, nivel: str = INFORMATIVO):
        obj = str.__new__(cls, texto)
        obj.nivel = nivel if nivel in _RANGO else INFORMATIVO
        return obj


def aviso(texto: str, nivel: str = INFORMATIVO) -> Aviso:
    return Aviso(texto, nivel)


# Frases de avisos que ya existían como texto suelto y que NO son simples notas (basta que el aviso CONTENGA la frase).
# Un string que no coincide con ninguna es informativo.
NIVEL_POR_FRASE = [
    # --- críticos: el resultado puede estar equivocado ---
    (CRITICO, "podrían ser una trampa"),
    (CRITICO, "es casi una copia"),
    # --- importantes: conviene saberlo antes de confiar ---
    (IMPORTANTE, "pero la prueba mezcló las filas al azar"),
    (IMPORTANTE, "son idénticas a otra"),
    (IMPORTANTE, "no coinciden con lo que ocurre"),
    (IMPORTANTE, "sirven para ordenar casos"),
    (IMPORTANTE, "casi no mejora a"),
    (IMPORTANTE, "Un modelo lineal simple rinde igual o mejor"),
    (IMPORTANTE, "Hay clases con menos de 5 filas"),
    (IMPORTANTE, "Hay pocas filas de prueba"),
    (IMPORTANTE, "se repite en varias filas"),
    (IMPORTANTE, "nunca vio (por ejemplo"),
    (IMPORTANTE, "No se pudo calcular qué columnas pesan"),
    (IMPORTANTE, "se cuentan desde ahí, no desde hoy"),
    (IMPORTANTE, "La última vez ("),
    (IMPORTANTE, "el rango es una extrapolación"),
    (IMPORTANTE, "queda fuera del rango que esa variable tuvo"),
    (IMPORTANTE, "El rango acertó solo en"),
    (IMPORTANTE, "se eligió y evaluó con los mismos casos"),
    (IMPORTANTE, "más lejos que la mitad del historial"),
    (IMPORTANTE, "porque está incompleto"),
    (IMPORTANTE, "bastante menos que lo habitual"),
    (IMPORTANTE, "no tenía datos y se rellenó"),
    (IMPORTANTE, "no es estable en el tiempo"),
    (IMPORTANTE, "no tienen registros en los últimos"),
    (IMPORTANTE, "no tienen stock conocido"),
    (IMPORTANTE, "no tienen una fecha de inicio válida"),
    (IMPORTANTE, "llevan más del doble del tiempo típico sin ningún registro"),
    (IMPORTANTE, "llega más lejos que el tiempo más largo visto"),
    (IMPORTANTE, "tienen pocos eventos"),
    (IMPORTANTE, "Ningún grupo tiene suficientes eventos"),
    (IMPORTANTE, "parece un puntaje u orden"),
    (IMPORTANTE, "Sin columna de impacto"),
    (IMPORTANTE, "no se usaron"),
    (IMPORTANTE, "tiene fechas posteriores a la fecha de la fila"),
    (IMPORTANTE, "no tiene dato en"),
    (IMPORTANTE, "ver el futuro"),
    (IMPORTANTE, "puede incluir información posterior"),
    (IMPORTANTE, "No hay fecha de inicio de cada elemento"),
]


def nivel_de(a) -> str:
    n = getattr(a, "nivel", None)
    if n in _RANGO:
        return n
    texto = str(a)
    for nivel, frase in NIVEL_POR_FRASE:
        if frase in texto:
            return nivel
    return INFORMATIVO


def ordenar_avisos(avisos) -> list:
    """Devuelve los avisos como `Aviso` (con su nivel), críticos primero. No modifica la lista original."""
    con_nivel = [a if isinstance(a, Aviso) and a.nivel in _RANGO else Aviso(a, nivel_de(a)) for a in (avisos or [])]
    return sorted(con_nivel, key=lambda a: _RANGO[a.nivel])           # sorted es estable: conserva el orden dentro del nivel