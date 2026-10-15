"""
ontologia.py — Motor de inferencia de relaciones entre tablas para Hadar.

Este módulo NO depende de PySide6 ni de nada de la interfaz. Su única
responsabilidad es: dado un diccionario de tablas (nombre -> DataFrame),
adivinar qué columnas de una tabla probablemente se relacionan con qué
columnas de otra tabla, y devolver esa lista con un nivel de confianza
y una explicación en español, pensada para mostrarse tal cual al usuario.

Idea general (en simple):
- Si dos columnas se llaman parecido (ej. "id_cliente" en ambas tablas),
  es una pista fuerte.
- Si además los VALORES se parecen (ej. los mismos números o códigos
  aparecen en las dos columnas), es una pista todavía más fuerte —
  de hecho es la prueba real de que hay una relación, no solo una
  coincidencia de nombre.
- Si los tipos de datos son incompatibles (una es texto libre tipo
  "comentarios" y la otra es un número), se descarta esa relación.
- Además, se intenta adivinar cuál de las dos columnas es la "tabla
  principal" (ej. Clientes.id_cliente, que no se repite) y cuál es la
  "tabla que la referencia" (ej. Pedidos.id_cliente, que sí se repite).

No hace falta que el usuario entienda nada de esto: el resultado es una
lista de relaciones sugeridas, ordenadas por confianza, listas para
dibujarse como líneas entre cajitas.

--- Cambios de esta versión (ver también ontologia_ui.py) ---
El diagrama estaba mostrando demasiadas líneas para pocas tablas (32
relaciones sugeridas para solo 3 tablas). Dos ajustes acá para que el
motor sea más exigente y deje pasar menos relaciones dudosas:

- UMBRAL_CONFIANZA_DEFECTO subió de 0.40 a 0.50: antes bastaba una
  coincidencia parcial de nombre + algo de solape de valores para
  sugerir una relación; ahora se pide más certeza.
- MAX_RELACIONES_POR_PAR_DE_TABLAS bajó de 5 a 3: por cada par de
  tablas, como máximo se muestran las 3 relaciones más confiables
  (antes eran 5, lo que sumaba rápido con varias tablas).

La otra mitad de la solución (agrupar todas las relaciones entre dos
tablas en UNA sola línea en el diagrama, en vez de una línea por cada
par de columnas) está en ontologia_ui.py — este archivo no necesita
saber nada de eso, solo sigue devolviendo la lista de relaciones.

--- Cambios de esta segunda ronda ---
Con esquemas más grandes (varias tablas de "personas": Empleados,
Clientes, Jugadores...) seguían apareciendo relaciones falsas de dos
tipos concretos, ambos corregidos acá:

1. Columnas de nombre genérico que coinciden por casualidad: dos
   columnas llamadas igual (ej. "tipo", "estado", "activo") en tablas
   que no tienen ninguna relación real entre sí. Antes, que el nombre
   fuera idéntico ya alcanzaba la confianza mínima por sí solo, sin
   necesitar que los VALORES tuvieran nada que ver. Ahora, si el
   nombre no tiene pinta de ser una llave (id, código, etc.), se exige
   además un mínimo de solape real de valores (UMBRAL_SOLAPE_MINIMO_
   NOMBRE_GENERICO).
2. Columnas de muy pocos valores distintos (booleanas tipo "activo",
   o con 2-3 categorías): CUALQUIER par de columnas así se va a
   "solapar" casi 100% entre cualquier par de tablas, sin que eso
   signifique nada. Ahora se ignoran como evidencia de relación
   (MINIMO_VALORES_DISTINTOS_PARA_SOLAPE).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd

# ----------------------------------------------------------------------
# Configuración de umbrales (fáciles de retocar más adelante)
# ----------------------------------------------------------------------

UMBRAL_CONFIANZA_DEFECTO = 0.50            # bajo esto, ni se sugiere (antes 0.40)
PESO_NOMBRE = 0.40                          # cuánto pesa que los nombres se parezcan
PESO_TIPO = 0.10                            # cuánto pesa que los tipos calcen
PESO_VALORES = 0.50                         # cuánto pesa que los valores se solapen
MAX_FILAS_MUESTRA_VALORES = 5000            # para no comparar sets gigantes
MAX_RELACIONES_POR_PAR_DE_TABLAS = 3        # evita saturar el diagrama (antes 5)
MINIMO_VALORES_DISTINTOS_PARA_SOLAPE = 5    # columnas con menos valores distintos que esto
                                             # (ej. sí/no, activo/inactivo) no cuentan como
                                             # evidencia: ese tipo de columna "se solapa" casi
                                             # siempre entre cualquier par de tablas, por azar
UMBRAL_SOLAPE_MINIMO_NOMBRE_GENERICO = 0.15 # para columnas con nombre genérico (no tipo "id"),
                                             # no basta con que el nombre coincida (ej. dos
                                             # columnas "tipo" o "estado" en tablas sin relación
                                             # real) -- se exige además este mínimo de solape real


# ----------------------------------------------------------------------
# Estructura de salida
# ----------------------------------------------------------------------

@dataclass
class RelacionSugerida:
    tabla_origen: str
    columna_origen: str
    tabla_destino: str
    columna_destino: str
    confianza: float                        # 0.0 a 1.0
    razon: str                              # explicación en español, para mostrar al usuario
    tabla_principal: str | None = None      # cuál de las dos tablas parece ser la "principal" (PK)
    columna_principal: str | None = None
    certeza_direccion: str = "sin_definir"  # "alta" | "sin_definir"
    # "alta": en una relación 1-a-N, el lado cuyos valores son (casi) todos únicos es
    #         el padre mientras el otro lado repite valores; o el usuario confirmó/corrigió
    #         la dirección a mano (ver direccion_confirmada).
    # "sin_definir": no hay padre/hijo. Ocurre en dos casos MUY distintos, que se
    #         distinguen con `cardinalidad`:
    #           - "uno_a_uno": no hay nada que asignar (ambas columnas son únicas). No es
    #             un problema; la interfaz NO debe pedir confirmación.
    #           - "muchos_a_muchos": ambas columnas repiten valores; el usuario puede
    #             indicar la principal a mano. tabla_principal queda en None.
    #         El motor de contagio no debe asumir dirección mientras sea None.
    direccion_confirmada: bool = False      # True si un humano eligió/corrigió tabla_principal
    # a mano en el diagrama (distinto de 'manual', que es toda la relación
    # agregada a mano). Sirve para que "Detectar automáticamente de nuevo"
    # no pise la corrección del usuario.
    manual: bool = False                    # True si el usuario la agregó a mano (no la detectó el motor)
    detalle: dict = field(default_factory=dict)  # datos extra por si la UI los quiere (opcional)
    cardinalidad: str = "sin_definir"       # "uno_a_uno", "uno_a_muchos", "muchos_a_muchos", "sin_definir"

    @property
    def direccion_pendiente(self) -> bool:
        """True solo cuando falta una decisión del usuario sobre la dirección.
        Una relación 1-a-1 NO queda pendiente: no existe padre/hijo que elegir."""
        return (
            not self.manual
            and self.certeza_direccion == "sin_definir"
            and self.cardinalidad != "uno_a_uno"
        )

    def como_texto(self) -> str:
        """Frase corta lista para mostrar en la interfaz."""
        return (
            f"{self.tabla_origen}.{self.columna_origen} ↔ "
            f"{self.tabla_destino}.{self.columna_destino} "
            f"({int(self.confianza * 100)}% de confianza) — {self.razon}"
        )


# ----------------------------------------------------------------------
# Utilidades internas
# ----------------------------------------------------------------------

def _normalizar_nombre(nombre: str) -> str:
    """
    'ID_Cliente', 'id cliente', 'Id-Cliente' -> 'idcliente'
    Saca tildes, mayúsculas, espacios, guiones y guiones bajos, para que
    la comparación de nombres no falle por detalles de formato.
    """
    texto = str(nombre).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[\s_\-\.]+", "", texto)
    return texto


def _similitud_nombres(col1: str, col2: str) -> float:
    """Similitud de texto entre 0 y 1 usando los nombres normalizados."""
    n1, n2 = _normalizar_nombre(col1), _normalizar_nombre(col2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    return SequenceMatcher(None, n1, n2).ratio()


def _tipos_compatibles(serie1: pd.Series, serie2: pd.Series) -> bool:
    """
    True si tiene sentido comparar los VALORES de estas dos columnas
    (ambas numéricas, o ambas texto/categoría). Evita, por ejemplo,
    comparar una columna de fechas con una de texto libre.
    """
    es_num1, es_num2 = pd.api.types.is_numeric_dtype(serie1), pd.api.types.is_numeric_dtype(serie2)
    es_fecha1, es_fecha2 = pd.api.types.is_datetime64_any_dtype(serie1), pd.api.types.is_datetime64_any_dtype(serie2)
    if es_fecha1 or es_fecha2:
        # las fechas casi nunca son claves de relación tabla-tabla en este contexto
        return es_fecha1 and es_fecha2
    if es_num1 != es_num2:
        return False
    return True


def _muestra_valores(serie: pd.Series) -> set:
    """Set de valores no nulos, recortado para no comparar columnas gigantes."""
    s = serie.dropna()
    if len(s) > MAX_FILAS_MUESTRA_VALORES:
        s = s.sample(MAX_FILAS_MUESTRA_VALORES, random_state=0)
    # normaliza texto para que "ABC" y "abc" no se consideren distintos
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
    return set(s.tolist())


def _solape_valores(serie1: pd.Series, serie2: pd.Series, permitir_pocos_valores: bool = False) -> float:
    """
    Qué tan solapados están los valores de dos columnas, de 0 a 1.
    Se mide contra el más chico de los dos conjuntos (para que una
    tabla "Clientes" con 50 ids y una "Pedidos" con 5000 filas, donde
    los 50 ids aparecen todos, dé un solape alto y correcto).
    """
    v1, v2 = _muestra_valores(serie1), _muestra_valores(serie2)
    if not v1 or not v2:
        return 0.0
    # Columnas con muy pocos valores distintos (activo, tipo, estado...) se
    # "solapan" casi 100% entre CUALQUIER par de tablas, sin que eso pruebe
    # nada. Solo se aceptan como evidencia si ambas columnas tienen nombre de
    # llave (ej. id_region con 4 regiones), que es el caso legítimo.
    if not permitir_pocos_valores and (
        len(v1) < MINIMO_VALORES_DISTINTOS_PARA_SOLAPE
        or len(v2) < MINIMO_VALORES_DISTINTOS_PARA_SOLAPE
    ):
        return 0.0
    interseccion = v1 & v2
    mas_chico = min(len(v1), len(v2))
    if mas_chico == 0:
        return 0.0
    return len(interseccion) / mas_chico


def _tokens_columna(nombre_columna: str) -> list[str]:
    """
    Separa un nombre de columna en 'palabras': 'id_cliente' -> ['id','cliente'],
    'IdCliente' -> ['id','cliente'], 'idcliente' -> ['idcliente'] (sin separador
    no se puede partir, se deja como una sola palabra).
    """
    texto = str(nombre_columna).strip()
    # separa por guiones/espacios/puntos y por cambios minúscula->mayúscula (camelCase)
    texto = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", texto)
    partes = re.split(r"[\s_\-\.]+", texto)
    return [_normalizar_nombre(p) for p in partes if p]


TOKENS_GENERICOS = {"id", "codigo", "cod", "rut", "folio", "numero", "key", "clave", "pk", "fk"}
UMBRAL_SIMILITUD_SEMANTICA = 0.6  # bajo esto, se asume que son entidades distintas


def _tokens_semanticos(nombre_columna: str) -> list[str]:
    """
    Los 'tokens con significado' de un nombre de columna, descartando
    sufijos/prefijos genéricos como 'id', 'codigo', 'folio', etc.
    'categoria_id' -> ['categoria'], 'id_cliente' -> ['cliente'],
    'cantidad' -> ['cantidad'] (no tiene sufijo genérico, es toda la palabra).
    """
    return [t for t in _tokens_columna(nombre_columna) if t not in TOKENS_GENERICOS]


def _similitud_semantica(col1: str, col2: str) -> float | None:
    """
    Compara solo la parte 'con nombre de entidad' de dos columnas, ignorando
    el sufijo genérico (_id, _codigo, etc). Es la pista que evita cruces
    como 'categoria_id' con 'cliente_id': el string completo se parece
    bastante (ambos terminan en 'id'), pero la entidad real -- categoria
    vs. cliente -- no tiene nada que ver.

    Devuelve un número de 0 a 1 (más alto = misma entidad), o None si
    alguna de las dos columnas es un identificador genérico puro sin parte
    semántica (ej. una columna llamada literalmente 'id') -- en ese caso
    no hay nada que comparar y no se aplica este filtro.
    """
    sem1, sem2 = _tokens_semanticos(col1), _tokens_semanticos(col2)
    if not sem1 or not sem2:
        return None
    mejor = 0.0
    for t1 in sem1:
        for t2 in sem2:
            mejor = max(mejor, SequenceMatcher(None, t1, t2).ratio())
    return mejor


def _parece_columna_clave(nombre_columna: str) -> bool:
    """
    Pistas típicas de que una columna es un identificador (id, código, etc.).
    Compara por PALABRA COMPLETA (tokens), no por subcadena, para que
    'cantidad' no se confunda con 'id' solo porque contiene esas letras.
    """
    patrones = {"id", "codigo", "cod", "rut", "folio", "numero", "key", "clave", "pk", "fk"}
    tokens = _tokens_columna(nombre_columna)
    if any(t in patrones for t in tokens):
        return True
    # CamelCase y nombres pegados: aceptar solo patrones con entidad detrás
    # del prefijo id, o una entidad antes del sufijo id; evitar grid/solid/ideal.
    texto = str(nombre_columna).strip()
    compacto = _normalizar_nombre(texto)
    if re.match(r"^[iI][dD][A-Z][a-zA-Z0-9]*$", texto):
        return True
    if re.match(r"^[A-Za-z][A-Za-z0-9]*ID$", texto):
        return True
    # Para nombres compactos en minúsculas, exigir longitud y token conocido
    # no es fiable; preferimos no clasificarlos como clave por subcadena.
    return compacto in {"id", "codigo", "cod", "rut", "folio", "numero", "key", "clave", "pk", "fk"}


UMBRAL_UNICIDAD_PK = 0.95    # "casi todos los valores únicos" = huella de PK
MIN_FILAS_PARA_UNICIDAD = 3  # ÚNICO mínimo de filas para hablar de unicidad (en todo el módulo)

# Valores que suelen significar "sin dato" y se repiten sin ser una relación real
# (ej. id_cliente = 0 para "sin cliente"). No cuentan al medir unicidad.
VALORES_CENTINELA = {0, -1, "", "0", "-1", "n/a", "na", "null", "none", "nan", "s/d", "s/i", "sin dato"}


def _valores_para_unicidad(serie: pd.Series) -> pd.Series:
    """Valores de la columna sin nulos, sin centinelas conocidos y sin el valor más
    frecuente cuando este se repite (un centinela desconocido, como 9999, suele ser
    justo ese). Así un "sin cliente" repetido no hace pasar una relación 1-a-N por
    muchos-a-muchos. Si la columna ya era única, no se quita nada."""
    v = serie.dropna()
    if v.empty:
        return v
    if v.dtype == object:
        v = v.astype(str).str.strip().str.lower()
    v = v[~v.isin(VALORES_CENTINELA)]
    if v.empty:
        return v
    conteos = v.value_counts()
    if conteos.iloc[0] > 1:
        v = v[v != conteos.index[0]]
    return v


def _es_columna_unica(serie: pd.Series) -> bool:
    v = _valores_para_unicidad(serie)
    n = len(v)
    return n >= MIN_FILAS_PARA_UNICIDAD and v.nunique() / n >= UMBRAL_UNICIDAD_PK


def _clasificar_cardinalidad(serie1: pd.Series, serie2: pd.Series):
    """Devuelve (cardinalidad, lado_padre) con lado_padre en {1, 2, None}.
    - un lado único y el otro repite -> "uno_a_muchos"; el lado único es el padre.
    - ambos únicos                   -> "uno_a_uno"; no hay padre/hijo.
    - ambos repiten                  -> "muchos_a_muchos"; sin padre (lo decide el usuario).
    Es el ÚNICO criterio de dirección del módulo."""
    u1, u2 = _es_columna_unica(serie1), _es_columna_unica(serie2)
    if u1 and u2:
        return "uno_a_uno", None
    if u1:
        return "uno_a_muchos", 1
    if u2:
        return "uno_a_muchos", 2
    return "muchos_a_muchos", None


# ----------------------------------------------------------------------
# Función principal
# ----------------------------------------------------------------------

def inferir_relaciones(
    tablas: dict[str, pd.DataFrame],
    umbral_confianza: float = UMBRAL_CONFIANZA_DEFECTO,
) -> list[RelacionSugerida]:
    """
    Punto de entrada del motor.

    Parámetros
    ----------
    tablas : dict {nombre_tabla: DataFrame}
        Ej: {"Clientes": df_clientes, "Pedidos": df_pedidos}
    umbral_confianza : float
        Relaciones por debajo de esto no se incluyen en el resultado.

    Devuelve
    --------
    Lista de RelacionSugerida, ordenada de mayor a menor confianza.
    Si solo hay 0 o 1 tabla, devuelve lista vacía (no hay nada que
    relacionar — así la pestaña Datos sabe que no debe mostrar el botón).
    """
    nombres_tablas = list(tablas.keys())
    if len(nombres_tablas) < 2:
        return []

    candidatas: list[RelacionSugerida] = []

    for i in range(len(nombres_tablas)):
        for j in range(i + 1, len(nombres_tablas)):
            t1, t2 = nombres_tablas[i], nombres_tablas[j]
            df1, df2 = tablas[t1], tablas[t2]

            relaciones_del_par: list[RelacionSugerida] = []

            for col1 in df1.columns:
                for col2 in df2.columns:
                    sim_nombre = _similitud_nombres(col1, col2)
                    col1_clave = _parece_columna_clave(col1)
                    col2_clave = _parece_columna_clave(col2)
                    ambas_son_clave = col1_clave and col2_clave
                    solo_una_es_clave = col1_clave != col2_clave

                    # Filtro de nombre. Ojo: cuando AMBAS columnas son de tipo
                    # 'id numérico' (ej. id_cliente vs id_pedido), exigimos más
                    # parecido de nombre que en el caso general — si no, dos
                    # columnas de ids chicos (1,2,3...) casi siempre se van a
                    # 'solapar' por pura casualidad y generan relaciones falsas.
                    # Si SOLO UNA parece llave (ej. "edad" vs "id_pedido"), el parecido de
                    # nombre debe ser mayor: con nombres cortos, SequenceMatcher da ~0.5 por
                    # puras letras en común, y una columna de rango chico (edad, mes, cantidad)
                    # "solapa" al 100% con cualquier secuencia de ids.
                    if ambas_son_clave:
                        umbral_nombre = 0.5
                    elif solo_una_es_clave:
                        umbral_nombre = 0.6
                    else:
                        umbral_nombre = 0.35
                    if sim_nombre < umbral_nombre:
                        continue

                    # Filtro semántico: aunque el nombre completo se parezca
                    # (ej. 'categoria_id' y 'cliente_id' comparten el sufijo
                    # '_id'), si la parte que nombra la ENTIDAD no tiene nada
                    # que ver ('categoria' vs. 'cliente'), se descarta antes
                    # de mirar los valores -- si no, columnas chicas de rango
                    # parecido (ids, cantidades) generan cruces sin sentido
                    # solo por casualidad numérica.
                    sim_semantica = _similitud_semantica(col1, col2)
                    if sim_semantica is not None and sim_semantica < UMBRAL_SIMILITUD_SEMANTICA:
                        continue

                    if not _tipos_compatibles(df1[col1], df2[col2]):
                        continue

                    solape = _solape_valores(df1[col1], df2[col2], permitir_pocos_valores=ambas_son_clave)

                    # Si el nombre no tiene pinta de ser una llave (id, código,
                    # rut, etc.), que se llamen igual no es prueba suficiente:
                    # columnas genéricas como "tipo" o "estado" se llaman igual
                    # todo el tiempo en tablas sin ninguna relación real. Se
                    # exige, además, algo de solape de valores de verdad.
                    if not ambas_son_clave and solape < UMBRAL_SOLAPE_MINIMO_NOMBRE_GENERICO:
                        continue

                    # El tipo incompatible ya se descarta arriba; ser compatible
                    # no prueba una relación y no debe sumar un offset constante.
                    confianza = PESO_NOMBRE * sim_nombre + PESO_VALORES * solape
                    if confianza < umbral_confianza:
                        continue

                    # Arma la razón en español, priorizando la pista más fuerte
                    partes_razon = []
                    if sim_nombre >= 0.8:
                        partes_razon.append("los nombres de columna son casi iguales")
                    elif sim_nombre >= 0.5:
                        partes_razon.append("los nombres de columna se parecen")

                    if solape >= 0.7:
                        partes_razon.append("la mayoría de los valores coinciden")
                    elif solape >= 0.3:
                        partes_razon.append("varios valores coinciden")

                    if not partes_razon:
                        partes_razon.append("hay una coincidencia parcial de nombre y tipo de dato")

                    cardinalidad, lado_padre = _clasificar_cardinalidad(df1[col1], df2[col2])
                    if lado_padre == 1:
                        principal, col_principal, certeza_direccion = t1, col1, "alta"
                    elif lado_padre == 2:
                        principal, col_principal, certeza_direccion = t2, col2, "alta"
                    else:
                        principal, col_principal, certeza_direccion = None, None, "sin_definir"

                    if cardinalidad == "uno_a_uno":
                        partes_razon.append("cada valor aparece una sola vez en ambas tablas (relación uno a uno, no hay tabla principal)")
                    elif cardinalidad == "muchos_a_muchos":
                        partes_razon.append("las dos columnas repiten valores (relación muchos a muchos); si sabes cuál tabla es la principal, indícalo con clic derecho")

                    razon = ", ".join(partes_razon).capitalize()

                    relaciones_del_par.append(
                        RelacionSugerida(
                            tabla_origen=t1,
                            columna_origen=col1,
                            tabla_destino=t2,
                            columna_destino=col2,
                            confianza=round(min(confianza, 1.0), 3),
                            razon=razon,
                            tabla_principal=principal,
                            columna_principal=col_principal,
                            certeza_direccion=certeza_direccion,
                            cardinalidad=cardinalidad,
                            detalle={"similitud_nombre": round(sim_nombre, 3), "solape_valores": round(solape, 3)},
                        )
                    )

            # De todas las combinaciones de columnas entre estas dos tablas,
            # nos quedamos solo con las mejores (para no llenar el diagrama
            # de líneas redundantes o ruidosas).
            relaciones_del_par.sort(key=lambda r: r.confianza, reverse=True)
            candidatas.extend(relaciones_del_par[:MAX_RELACIONES_POR_PAR_DE_TABLAS])

    candidatas.sort(key=lambda r: r.confianza, reverse=True)
    return candidatas


# ----------------------------------------------------------------------
# Enriquecimiento relacional para el ML multivariado (v1, simple)
# ----------------------------------------------------------------------
#
# Idea (inspirada en cómo Palantir Foundry arma su "Ontología"): antes de
# correr el Isolation Forest sobre una tabla, se le agregan columnas nuevas
# que resumen lo que pasa en sus tablas relacionadas 1-a-muchos. Así el
# modelo puede notar cosas como "un pedido con monto normal pero con una
# cantidad de líneas rarísima para ese cliente", que es invisible si cada
# tabla se mira sola.
#
# v1 a propósito simple: solo relaciones 1-a-muchos con dirección clara
# (certeza_direccion distinta de "sin_definir" -- si ni el propio motor de
# ontología está seguro de quién es el padre, mejor no inventar), y solo
# columnas NUMÉRICAS de la tabla hija (cantidad, suma, promedio). No toca
# texto ni fechas todavía.

def enriquecer_tabla_con_relaciones(
    nombre_tabla_principal: str,
    tablas: dict,
    relaciones: list,
) -> pd.DataFrame:
    """Devuelve una COPIA de tablas[nombre_tabla_principal] con columnas
    nuevas resumiendo cada tabla relacionada 1-a-muchos que la tenga a ella
    como tabla principal. No modifica ninguna tabla original. Si no hay
    relaciones claras que apliquen, devuelve la tabla tal cual (copia).

    Nombres de columnas nuevas: "{tabla_hija}__cantidad" (cuántas filas de
    la tabla hija le corresponden a cada fila) y, por cada columna numérica
    de la tabla hija, "{tabla_hija}__{columna}__suma" y
    "{tabla_hija}__{columna}__promedio".
    """
    df_principal = tablas[nombre_tabla_principal].copy()
    if df_principal.empty:
        return df_principal

    # Si dos relaciones distintas apuntan a la MISMA tabla hija (ej.
    # inferir_relaciones encontró dos columnas candidatas entre las mismas
    # dos tablas), solo se procesa la primera -- repetir la tabla hija
    # generaría columnas "__cantidad" duplicadas, y el merge las renombraría
    # solo (Ventas__cantidad_x / _y) en vez de sobrescribirlas, rompiendo la
    # línea siguiente que busca el nombre original.
    tablas_hijas_ya_procesadas = set()

    for rel in relaciones:
        if rel.certeza_direccion == "sin_definir":
            continue
        if rel.tabla_principal != nombre_tabla_principal:
            continue

        # La tabla "hija" (lado de muchos) es la que NO es la principal.
        if rel.tabla_origen == nombre_tabla_principal:
            tabla_hija, columna_hija, columna_principal = (
                rel.tabla_destino, rel.columna_destino, rel.columna_origen,
            )
        elif rel.tabla_destino == nombre_tabla_principal:
            tabla_hija, columna_hija, columna_principal = (
                rel.tabla_origen, rel.columna_origen, rel.columna_destino,
            )
        else:
            continue

        if tabla_hija in tablas_hijas_ya_procesadas:
            continue

        df_hija = tablas.get(tabla_hija)
        if df_hija is None or df_hija.empty:
            continue
        if columna_hija not in df_hija.columns or columna_principal not in df_principal.columns:
            continue

        # Si la columna en la tabla hija no se repite, no es 1-a-muchos de
        # verdad (sería 1-a-1) -- no hay nada útil que resumir.
        if not df_hija[columna_hija].duplicated().any():
            continue

        tablas_hijas_ya_procesadas.add(tabla_hija)
        prefijo = tabla_hija

        conteo = df_hija.groupby(columna_hija).size().rename(f"{prefijo}__cantidad")
        # Asignar por mapa conserva exactamente las filas y su orden, incluso
        # si la clave de la tabla principal está duplicada.
        mapa_conteo = conteo.to_dict()
        df_principal[f"{prefijo}__cantidad"] = (
            df_principal[columna_principal].map(mapa_conteo).fillna(0)
        )

        columnas_numericas = [
            c for c in df_hija.select_dtypes(include="number").columns
            if c != columna_hija and not _parece_columna_clave(c)
        ]
        for col in columnas_numericas:
            agregados = df_hija.groupby(columna_hija)[col].agg(["sum", "mean"]).rename(
                columns={"sum": f"{prefijo}__{col}__suma", "mean": f"{prefijo}__{col}__promedio"}
            )
            for nombre_agregado in agregados.columns:
                df_principal[nombre_agregado] = df_principal[columna_principal].map(
                    agregados[nombre_agregado].to_dict()
                )

    return df_principal


# ----------------------------------------------------------------------
# Demo / auto-test rápido con datos de ejemplo (Clientes / Pedidos / Productos)
# ----------------------------------------------------------------------

def _demo():
    clientes = pd.DataFrame({
        "id_cliente": [1, 2, 3, 4, 5],
        "nombre": ["Ana", "Beto", "Carla", "Diego", "Elena"],
        "region": ["RM", "Biobío", "RM", "Valparaíso", "RM"],
    })
    productos = pd.DataFrame({
        "id_producto": [100, 101, 102],
        "nombre_producto": ["Teclado", "Mouse", "Monitor"],
        "precio": [15990, 8990, 129990],
    })
    pedidos = pd.DataFrame({
        "id_pedido": range(1, 11),
        "id_cliente": [1, 1, 2, 3, 3, 3, 4, 5, 5, 2],
        "id_producto": [100, 101, 102, 100, 100, 101, 102, 100, 101, 102],
        "cantidad": [1, 2, 1, 1, 3, 1, 1, 2, 1, 1],
        "comentario": ["ok", "urgente", "", "", "regalo", "", "", "", "ok", ""],
    })
    tablas = {"Clientes": clientes, "Pedidos": pedidos, "Productos": productos}

    print("=== Relaciones sugeridas ===\n")
    relaciones = inferir_relaciones(tablas)
    if not relaciones:
        print("(no se encontraron relaciones por encima del umbral)")
    for r in relaciones:
        marca_principal = f" [tabla principal: {r.tabla_principal}]" if r.tabla_principal else ""
        print(" - " + r.como_texto() + marca_principal)

    print("\n=== Caso de 1 sola tabla (no debe sugerir nada) ===")
    print(inferir_relaciones({"Clientes": clientes}))


if __name__ == "__main__":
    _demo()