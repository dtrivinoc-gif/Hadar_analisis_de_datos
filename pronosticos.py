"""
pronosticos.py — Puerta de entrada de las «palabras clave» de la pestaña Predicción (sin Qt).

La pantalla solo necesita tres cosas de este módulo:

  PALABRAS_CLAVE           catálogo: nombre, pregunta, ejemplos y qué campos pide cada una
                           (con eso se puede dibujar el formulario sin conocer cada método).
  ConfigPronostico         lo que el usuario eligió (se guarda en el proyecto como diccionario).
  ejecutar(cfg, tablas)    corre la palabra clave y devuelve un `ResultadoPronostico`.
  sugerir_configuracion    propone columnas (fecha, valor, elemento…) para no empezar de cero.

Las palabras clave no saben de «ventas» ni de «máquinas»: trabajan con fechas,
valores y elementos. El dominio solo cambia el vocabulario.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

try:
    from .prediccion import _tokens, es_columna_id, tipo_columna
    from .pronostico_base import ResultadoPronostico
    from . import eventos as _ev
    from . import series_tiempo as _st
except ImportError:
    from prediccion import _tokens, es_columna_id, tipo_columna
    from pronostico_base import ResultadoPronostico
    import eventos as _ev
    import series_tiempo as _st


@dataclass
class ConfigPronostico:
    palabra: str = "proyeccion"
    tabla: str | None = None
    # --- serie de tiempo ---
    columna_fecha: str | None = None
    columna_valor: str | None = None
    columna_entidad: str | None = None       # máquina, producto, sucursal, cliente…
    entidad: str | None = None               # un elemento concreto (None = todos juntos)
    agregacion: str = "auto"                 # auto | suma | promedio | ultimo | max | min | cuenta
    frecuencia: str = "auto"                 # auto | H | D | W | M
    horizonte: int = 0                       # cuántos períodos mirar hacia adelante (0 = automático)
    nivel_intervalo: float = 0.8             # amplitud del rango (0.8 = 80%)
    ventana: int = 0                         # usar solo los últimos N períodos (0 = todos)
    # --- umbral / saturación ---
    umbral: float | None = None
    direccion: str = "auto"                  # auto | sube | baja
    capacidad: float | None = None
    # --- agotamiento ---
    stock_actual: float | None = None
    columna_stock: str | None = None
    # --- eventos (vida útil, falla, duración) ---
    columna_evento: str | None = None
    valor_evento: str | None = None
    modo_evento: str = "repetido"            # repetido (equipos, se reparan) | unico (clientes, vencimientos)
    metodo_evento: str = "estadistico"       # estadistico | modelo
    columna_inicio: str | None = None
    columna_fin: str | None = None
    columna_grupo: str | None = None
    unidad_duracion: str = "dias"
    ventana_dias: float | None = None        # ventana de la pregunta (alerta, probabilidad en N días…)
    fecha_corte: str | None = None
    # --- escenario ---
    columna_driver: str | None = None
    agregacion_driver: str = "promedio"
    cambio_pct: float = 10.0
    rezago: int = 0
    # --- prioridad / clasificación ---
    columna_riesgo: str | None = None
    columna_impacto: str | None = None
    top: int = 20
    columna_objetivo: str | None = None
    columnas_explicativas: list = field(default_factory=list)
    validacion: str = "aleatoria"
    semilla: int = 0

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d) -> "ConfigPronostico":
        d = d or {}
        nombres = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in nombres})


# ----------------------------------------------------------------------
# Catálogo (lo usa la pantalla para dibujar el formulario)
# ----------------------------------------------------------------------
def _c(campo, etiqueta, tipo, obligatorio=False, ayuda="", opciones=None, defecto=None):
    return {"campo": campo, "etiqueta": etiqueta, "tipo": tipo, "obligatorio": obligatorio,
            "ayuda": ayuda, "opciones": opciones, "defecto": defecto}


_FECHA = _c("columna_fecha", "Fecha", "fecha", True, "Columna con la fecha de cada registro.")
_VALOR = _c("columna_valor", "Valor a seguir", "numerica", False,
            "Lo que quieres proyectar (ventas, temperatura, consumo…). Si lo dejas vacío se cuentan los registros.")
_ENT = _c("columna_entidad", "Elemento (opcional)", "columna", False,
          "Máquina, producto, sucursal, cliente… Permite elegir uno o ver todos juntos.")
_ENT_UNO = _c("entidad", "Solo este elemento", "valor_de_columna", False, "Vacío = todos juntos.")
_AGREGACION = _c("agregacion", "Cómo juntar los datos", "opcion", False,
                 "Auto: suma lo que se acumula (ventas) y promedia lo que es un nivel (temperatura, stock).",
                 ["auto", "suma", "promedio", "ultimo", "max", "min", "cuenta"], "auto")
_FREC = _c("frecuencia", "Período", "opcion", False, "Auto lo detecta según tus fechas.", ["auto", "H", "D", "W", "M"], "auto")
_HORIZ = _c("horizonte", "Cuánto mirar hacia adelante", "entero", False,
            "En períodos (días, semanas…). 0 = automático.", None, 0)
_EVENTO_COLS = [
    _c("columna_entidad", "Elemento", "columna", True, "Qué identifica a cada máquina, cliente, equipo…"),
    _c("columna_evento", "Columna que marca el evento", "columna", False,
       "Si la tabla es un registro de eventos (solo fallas, por ejemplo), déjala vacía."),
    _c("valor_evento", "Valor que significa «ocurrió»", "texto", False, "Por ejemplo: falla. Vacío = números distintos de 0 o texto no vacío."),
    _c("modo_evento", "Cómo se comporta", "opcion", False,
       "repetido: tras el evento sigue (un equipo que se repara). unico: ocurre una vez (un cliente que se va).",
       ["repetido", "unico"], "repetido"),
    _c("columna_inicio", "Fecha de inicio del elemento (opcional)", "fecha", False, "Instalación, alta del cliente… Mejora la estimación."),
    _c("columna_grupo", "Agrupar por tipo (opcional)", "columna", False, "Modelo, categoría… cada grupo con suficientes eventos se estima aparte."),
]

PALABRAS_CLAVE = {
    "proyeccion": {
        "nombre": "Proyección", "mecanismo": "serie",
        "pregunta": "¿Qué valor tendrá más adelante?",
        "ejemplos": "ventas del próximo mes · temperatura de mañana · consumo de la semana",
        "campos": [_FECHA, _VALOR, _ENT, _ENT_UNO, _AGREGACION, _FREC, _HORIZ]},
    "tendencia": {
        "nombre": "Tendencia", "mecanismo": "serie",
        "pregunta": "¿Sube, baja o se mantiene? ¿A qué ritmo?",
        "ejemplos": "ganancias · tasa de fallas · rotación de personal",
        "campos": [_FECHA, _VALOR, _ENT, _ENT_UNO, _AGREGACION, _FREC,
                   _c("ventana", "Mirar solo los últimos N períodos", "entero", False, "0 = todo el historial.", None, 0)]},
    "estacionalidad": {
        "nombre": "Estacionalidad", "mecanismo": "serie",
        "pregunta": "¿Cuándo se repiten los picos y los valles?",
        "ejemplos": "días de más demanda · horas punta · meses flojos",
        "campos": [_FECHA, _VALOR, _ENT, _ENT_UNO, _AGREGACION, _FREC]},
    "escenario": {
        "nombre": "Escenario", "mecanismo": "serie",
        "pregunta": "¿Qué pasa si cambio una variable?",
        "ejemplos": "qué pasa con las ventas si subo el precio 10% · y con el consumo si cambio los turnos",
        "campos": [_FECHA, _VALOR, _c("columna_driver", "Variable que quiero mover", "numerica", True, "Precio, publicidad, carga…"),
                   _c("cambio_pct", "Cambio (%)", "numero", True, "Positivo sube, negativo baja.", None, 10.0),
                   _c("rezago", "El efecto tarda (períodos)", "entero", False, "0 = efecto inmediato.", None, 0),
                   _c("agregacion_driver", "Cómo juntar la variable", "opcion", False, "", ["promedio", "suma", "ultimo", "max", "min"], "promedio"),
                   _ENT, _ENT_UNO, _AGREGACION, _FREC]},
    "agotamiento": {
        "nombre": "Agotamiento", "mecanismo": "serie",
        "pregunta": "¿Cuándo se acaba un recurso?",
        "ejemplos": "stock de un producto · batería · presupuesto · espacio en disco · insumos",
        "campos": [_FECHA,
                   _c("columna_valor", "Consumo por registro", "numerica", False, "Lo que se gasta. Si no lo tienes, se deduce de las bajas del stock."),
                   _c("columna_stock", "Columna de stock (opcional)", "numerica", False, "Se usa el último valor de cada elemento."),
                   _c("stock_actual", "Stock actual (si no hay columna)", "numero", False, "Un número, para un solo recurso."),
                   _ENT, _FREC,
                   _c("ventana_dias", "Avisar si se acaba en menos de (días)", "numero", False, "", None, 14),
                   _c("ventana", "Mirar el consumo de los últimos N períodos", "entero", False, "0 = automático.", None, 0)]},
    "umbral": {
        "nombre": "Umbral", "mecanismo": "serie",
        "pregunta": "¿Cuándo una métrica cruzará un límite?",
        "ejemplos": "temperatura sobre 80° · deuda sobre un tope · uso de CPU sobre 90%",
        "campos": [_FECHA, _VALOR, _c("umbral", "Límite", "numero", True, "El valor que quieres vigilar."),
                   _c("direccion", "Qué vigilar", "opcion", False, "auto: según si hoy estás sobre o bajo el límite.", ["auto", "sube", "baja"], "auto"),
                   _ENT, _ENT_UNO, _AGREGACION, _FREC, _HORIZ]},
    "saturacion": {
        "nombre": "Saturación", "mecanismo": "serie",
        "pregunta": "¿Con qué probabilidad se superará la capacidad?",
        "ejemplos": "ancho de banda · aforo · cola de atención · carga de una planta",
        "campos": [_FECHA, _VALOR, _c("capacidad", "Capacidad máxima", "numero", True, "El valor desde el cual se satura."),
                   _ENT, _ENT_UNO, _AGREGACION, _FREC, _HORIZ]},
    "vida_util": {
        "nombre": "Vida útil", "mecanismo": "evento",
        "pregunta": "¿Cuánto falta para que ocurra un evento?",
        "ejemplos": "falla de un equipo · vencimiento · renovación de contrato",
        "campos": [_FECHA] + _EVENTO_COLS + [_c("ventana_dias", "Ventana de alerta (días)", "numero", False, "", None, 30)]},
    "falla": {
        "nombre": "Falla / evento", "mecanismo": "evento",
        "pregunta": "¿Qué probabilidad hay de que ocurra en los próximos N días?",
        "ejemplos": "falla de una máquina · abandono de un cliente · impago · fraude",
        "campos": [_FECHA] + _EVENTO_COLS + [
            _c("ventana_dias", "Próximos (días)", "numero", True, "La pregunta: ¿ocurre en estos días?", None, 30),
            _c("metodo_evento", "Método", "opcion", False,
               "estadistico: solo usa cuándo ocurren los eventos. modelo: además usa mediciones y condiciones de cada registro.",
               ["estadistico", "modelo"], "estadistico"),
            _c("columnas_explicativas", "Columnas que explican (solo método modelo)", "multi_columna", False, "Mediciones, uso, tipo…")]},
    "duracion": {
        "nombre": "Duración", "mecanismo": "evento",
        "pregunta": "¿Cuánto tardará algo?",
        "ejemplos": "entrega de un pedido · reparación · trámite · tiempo de atención",
        "campos": [_c("columna_inicio", "Fecha de inicio", "fecha", False, "Si tienes inicio y fin."),
                   _c("columna_fin", "Fecha de fin", "fecha", False, "Vacío = el caso sigue abierto."),
                   _c("columna_valor", "…o columna con la duración", "numerica", False, "Alternativa si ya viene calculada."),
                   _c("unidad_duracion", "Unidad de esa columna", "opcion", False, "", ["dias", "horas", "minutos", "semanas"], "dias"),
                   _c("columna_grupo", "Agrupar por tipo (opcional)", "columna", False, "Tipo de trámite, proveedor…"),
                   _c("columna_entidad", "Identificador del caso (opcional)", "columna", False, ""),
                   _c("ventana_dias", "Probabilidad de terminar en (días)", "numero", False, "", None, 7)]},
    "clasificacion": {
        "nombre": "Clasificación", "mecanismo": "tabla",
        "pregunta": "¿A qué categoría pertenece?",
        "ejemplos": "aprobado o rechazado · tipo de defecto · cliente que compra o no",
        "campos": [_c("columna_objetivo", "Categoría a predecir", "columna", True, ""),
                   _c("columnas_explicativas", "Columnas que explican (vacío = automático)", "multi_columna", False, "")]},
    "prioridad": {
        "nombre": "Prioridad", "mecanismo": "tabla",
        "pregunta": "¿Qué casos debo mirar primero?",
        "ejemplos": "los equipos más riesgosos · los clientes con más chance de irse y más valor",
        "campos": [_c("columna_riesgo", "Riesgo", "numerica", True, "Una probabilidad o puntaje (por ejemplo, el resultado de «falla»)."),
                   _c("columna_impacto", "Impacto (opcional)", "numerica", False, "Lo que se pierde si ocurre (dinero, horas, clientes)."),
                   _c("columna_entidad", "Identificador", "columna", False, ""),
                   _c("top", "Cuántos revisar primero", "entero", False, "", None, 20)]},
}

ORDEN_PALABRAS = ["proyeccion", "tendencia", "estacionalidad", "escenario", "agotamiento", "umbral", "saturacion",
                  "vida_util", "falla", "duracion", "clasificacion", "prioridad"]


# ----------------------------------------------------------------------
# Ejecutar
# ----------------------------------------------------------------------
_FUNCIONES = {
    "proyeccion": _st.analizar_proyeccion, "tendencia": _st.analizar_tendencia,
    "estacionalidad": _st.analizar_estacionalidad, "umbral": _st.analizar_umbral,
    "saturacion": _st.analizar_saturacion, "agotamiento": _st.analizar_agotamiento,
    "escenario": _st.analizar_escenario, "vida_util": _ev.analizar_vida_util, "falla": _ev.analizar_falla,
    "duracion": _ev.analizar_duracion, "prioridad": _ev.analizar_prioridad,
    "clasificacion": _ev.analizar_clasificacion,
}


def campos_faltantes(cfg: ConfigPronostico) -> list:
    """Nombres legibles de los campos obligatorios que aún no se han llenado."""
    spec = PALABRAS_CLAVE.get(cfg.palabra)
    if spec is None:
        return ["palabra clave"]
    faltan = []
    for c in spec["campos"]:
        if c["obligatorio"]:
            v = getattr(cfg, c["campo"], None)
            if v is None or v == "" or v == []:
                faltan.append(c["etiqueta"])
    return faltan


def ejecutar(cfg: ConfigPronostico, tablas: dict) -> ResultadoPronostico:
    """Corre la palabra clave sobre la tabla elegida. Lanza ValueError con un mensaje en español
    claro cuando falta algo o los datos no alcanzan."""
    if cfg.palabra not in _FUNCIONES:
        raise ValueError(f"Palabra clave desconocida: «{cfg.palabra}».")
    if not cfg.tabla or cfg.tabla not in tablas:
        raise ValueError("Elige la tabla con los datos.")
    faltan = campos_faltantes(cfg)
    if faltan:
        raise ValueError("Falta completar: " + ", ".join(faltan) + ".")
    df = tablas[cfg.tabla]
    if len(df) < 5:
        raise ValueError("La tabla tiene muy pocas filas para predecir algo.")
    res = _FUNCIONES[cfg.palabra](df, cfg)
    return res


# ----------------------------------------------------------------------
# Sugerir columnas
# ----------------------------------------------------------------------
_PISTAS_VALOR = {"venta", "ventas", "monto", "total", "cantidad", "consumo", "ingreso", "ingresos", "ganancia",
                 "ganancias", "costo", "precio", "valor", "temperatura", "uso", "carga", "stock", "unidades"}
_PISTAS_EVENTO = {"falla", "fallo", "evento", "error", "incidente", "averia", "avería", "tipo", "estado", "status",
                  "alarma", "alerta"}


def sugerir_configuracion(df: pd.DataFrame, palabra: str = "proyeccion") -> dict:
    """Propone columnas para empezar (el usuario puede cambiarlas). Devuelve {campo: valor}."""
    sug = {}
    tipos = {c: tipo_columna(df[c]) for c in df.columns}
    fechas = [c for c, t in tipos.items() if t in ("fecha", "fecha_texto")]
    if fechas:
        sug["columna_fecha"] = fechas[0]
    nums = [c for c, t in tipos.items() if t == "numerica" and not es_columna_id(c)]
    puntaje = lambda c: len(set(_tokens(c)) & _PISTAS_VALOR)
    if nums:
        sug["columna_valor"] = sorted(nums, key=lambda c: (-puntaje(c), list(df.columns).index(c)))[0]
    n = len(df)
    ents = []
    for c, t in tipos.items():
        if t == "categorica" or (es_columna_id(c) and t in ("numerica", "numerica_texto", "categorica")):
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= max(n // 3, 2) and nun <= 5000 and c not in sug.values():
                ents.append((0 if es_columna_id(c) else 1, c))
    if ents:
        sug["columna_entidad"] = sorted(ents)[0][1]
    for c, t in tipos.items():
        if t in ("categorica", "booleana") and set(_tokens(c)) & _PISTAS_EVENTO and c != sug.get("columna_entidad"):
            sug["columna_evento"] = c
            break
    if palabra in ("duracion",):
        ini = [c for c in fechas if set(_tokens(c)) & {"inicio", "apertura", "creacion", "creación", "ingreso", "solicitud"}]
        fin = [c for c in fechas if set(_tokens(c)) & {"fin", "cierre", "termino", "término", "entrega", "resolucion", "resolución"}]
        if ini:
            sug["columna_inicio"] = ini[0]
        if fin:
            sug["columna_fin"] = fin[0]
    return sug
