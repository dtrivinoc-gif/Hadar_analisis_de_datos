"""
ajustes.py — Ajustes del proyecto para la pestaña Predicción (sin Qt).

Son las decisiones que valen para TODO el proyecto (no para una pregunta puntual). Se guardan en el .hadarproy
dentro de la configuración de Predicción, bajo la clave «ajustes»; un proyecto antiguo sin esa clave abre con los
valores por defecto. Cada ajuste se COPIA a la configuración de cada análisis (ver `aplicar_ajustes`), así queda
registrado qué se usó y los resultados siguen siendo reproducibles.

La pantalla (ajustes_ui.py) se dibuja a partir de `OPCIONES`: aquí están el texto, las opciones y el «por qué
importa» de cada ajuste, para que no haya lógica escondida en la pantalla.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

try:
    from . import pronostico_base as _base
except ImportError:
    import pronostico_base as _base

# (clave, título, por qué importa, [(valor, etiqueta)])
OPCIONES = [
    ("formato_numerico", "Formato de los números",
     "Cómo se escriben los números en los textos y resultados.",
     [("chileno", "Chileno: 1.234,5"), ("estandar", "Estándar: 1,234.5")]),
    ("fin_observacion", "Hasta cuándo se observa a cada elemento",
     "Para vida útil y probabilidad de falla. «Automático» usa el último registro de cada elemento si la tabla trae "
     "lecturas o estados, y la fecha más reciente de la tabla si solo trae eventos. Si un elemento se dio de baja, "
     "contarlo como «sigue sin fallar» alarga la vida útil estimada.",
     [("auto", "Automático (recomendado)"), ("ultimo_registro", "Hasta su último registro"),
      ("global", "Hasta la fecha más reciente de la tabla")]),
    ("ceros_mismo_dia", "Casos que terminan el mismo día (0 días)",
     "Con fechas sin hora, un caso de 0 días significa «menos de un día». Descartarlos alarga el tiempo típico "
     "(se verían solo los casos que tardaron más).",
     [("medio_dia", "Contarlos como medio día (recomendado)"), ("descartar", "No usarlos (solo avisar)")]),
    ("validacion_con_fecha", "Cómo probar un modelo cuando la tabla tiene fechas",
     "Para saber si un modelo sirve, Hadar lo prueba con casos que no vio. «Por fecha» usa los casos más nuevos para "
     "probar y los más viejos para aprender, que es lo que pasará de verdad al predecir el futuro. Mezclar las filas al "
     "azar suele dar una nota más alta que la real cuando los datos cambian con el tiempo.",
     [("auto", "Por fecha si hay una fecha clara (recomendado)"), ("aleatoria", "Mezclar al azar (solo avisar)")]),
    ("dias_aviso_datos_antiguos", "Avisar si el último dato es más viejo que…",
     "Las fechas y plazos se cuentan desde el último dato de la tabla, no desde hoy. Si ese dato es muy antiguo, "
     "Hadar lo avisa.",
     [(15, "15 días"), (45, "45 días (recomendado)"), (90, "90 días"), (365, "1 año")]),
]
_VALIDOS = {k: [v for v, _ in ops] for k, _, _, ops in OPCIONES}


@dataclass
class AjustesPrediccion:
    formato_numerico: str = "chileno"
    fin_observacion: str = "auto"
    ceros_mismo_dia: str = "medio_dia"
    validacion_con_fecha: str = "auto"
    dias_aviso_datos_antiguos: int = 45

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d) -> "AjustesPrediccion":
        """Lee ajustes guardados. Lo que falte o tenga un valor que ya no existe vuelve al valor por defecto."""
        a = cls()
        for k, v in (d or {}).items() if isinstance(d, dict) else []:
            if k in _VALIDOS and v in _VALIDOS[k]:
                setattr(a, k, v)
        return a

    def activar(self) -> None:
        """Fija lo que es global en la pantalla (el formato de los números)."""
        _base.set_formato_numerico(self.formato_numerico)


def aplicar_ajustes(cfg, ajustes) -> object:
    """Copia los ajustes del proyecto a la configuración de un análisis (devuelve la misma `cfg`)."""
    a = ajustes if isinstance(ajustes, AjustesPrediccion) else AjustesPrediccion.from_dict(ajustes)
    cfg.fin_observacion = a.fin_observacion
    cfg.ceros_mismo_dia = a.ceros_mismo_dia
    cfg.validacion = a.validacion_con_fecha
    cfg.dias_aviso_datos_antiguos = a.dias_aviso_datos_antiguos
    _base.set_formato_numerico(a.formato_numerico)
    return cfg