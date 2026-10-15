"""
Guardar y abrir un "proyecto" de Hadar: una foto del trabajo en curso para
poder cerrar la app y seguir exactamente donde quedaste otro día.

Qué SÍ guarda un proyecto:
  - Las tablas cargadas (todas, tal cual quedaron -- con columnas nuevas
    de "Excel: Calcular y Arrastrar" incluidas).
  - Cuál era la tabla activa, y el filtro que tenías puesto en Datos.
  - El esquema de relaciones (si abriste "Ver esquema" y lo confirmaste).
  - Tus indicadores.
  - Las notas de celda que escribiste TÚ a mano (no las automáticas de
    Narrativa, que se regeneran solas al volver a generar el informe).

Qué NO guarda (a propósito, por ahora):
  - Las reglas de Alarma: ya persisten solas, de forma global, en su
    propio archivo (ver alarmas.py) -- no dependen de qué proyecto tengas
    abierto, así que no hay nada que duplicar acá. Al reabrir un proyecto,
    las alarmas que apliquen a sus indicadores/columnas se enganchan solas.
  - El informe de Narrativa y el lienzo de Reporte: se regeneran con un
    clic una vez que el resto del proyecto está cargado. Podrían agregarse
    en una etapa futura si hace falta.

Formato en disco: un archivo .hadarproy, que por dentro es un .zip (misma
idea que un .docx de Word) con:
  proyecto.json           -- todo lo que no son datos de tabla
  tablas/<nombre>.parquet -- cada tabla, tal cual (mismo formato que ya
                             usa Hadar para datasets grandes)
"""
import io
import json
import os
import zipfile
from dataclasses import asdict, fields

import numpy as np
import pandas as pd

from .ontologia import RelacionSugerida

VERSION_FORMATO = 1
EXTENSION = ".hadarproy"

_CAMPOS_RELACION = {f.name for f in fields(RelacionSugerida)}


def _valor_json_seguro(valor):
    """Convierte tipos de numpy/pandas (que json.dump no entiende de
    fábrica) a tipos nativos de Python."""
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    return valor


def ruta_carpeta_proyectos():
    """Carpeta sugerida por defecto para guardar/buscar proyectos --
    reutiliza la misma carpeta de datos de usuario que ya usa memoria.py,
    para no desparramar archivos de Hadar por todos lados."""
    from .memoria import ruta_directorio_datos_usuario
    carpeta = os.path.join(ruta_directorio_datos_usuario(), "Proyectos")
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def guardar_proyecto(ruta, *, tablas, nombre_tabla_activa, relaciones_ontologia,
                      filtro_columna, filtro_valores, notas_manuales, indicadores,
                      ml_activado=False, linea_base_ml=None, fuentes_datos=None,
                      ml_multivariado_activado=False, procedencia=None, prediccion=None,
                      panel=None, tablero_meta=None, tablero_tablas=None, tablero_pronosticos=None,
                      tablero_retornos=None):
    """
    ... (ver parámetros existentes arriba)
    panel: dict con lo que el usuario eligió en Personalizar del Panel (qué
        indicadores, qué gráficos propios, etc.). Solo textos, números y
        listas, así que se guarda tal cual.
    prediccion: dict con la configuración de la pestaña Predicción (qué tabla,
        qué columna predecir, qué columnas usar o ignorar). El modelo entrenado
        NO se guarda: un .hadarproy es un .zip común y los modelos de Python
        (pickle) pueden ejecutar código al abrirse. Se reentrena con un clic.
    tablero_meta / tablero_tablas: las predicciones por fila que la pestaña
        Predicción dejó para las demás pestañas (ver tablero.py). Meta = textos
        (qué se predijo, cuándo); tablas = {nombre_tabla: DataFrame} con SOLO los
        números predichos. No es un modelo: no hay nada que se ejecute al abrir.
        Los proyectos guardados antes no traen estas claves y abren igual.
    tablero_pronosticos: lista de dicts con los pronósticos que el usuario llevó
        al Panel («Lo que se viene»): solo texto (titular con rango, confianza,
        fecha). Tampoco es un modelo.
    tablero_retornos: lista de dicts con la última comparación de lo predicho con
        lo real, por tabla (ver retorno.py): solo texto y números.
    fuentes_datos: dict {nombre_tabla: {"tipo": "archivo"|"sql_server"|"api_pos", ...}}
        de dónde vino cada tabla, para poder "Actualizar desde la fuente"
        más adelante sin volver a pedir todo desde cero.
        Para "api_pos" (sincronización con el POS por la red local), se
        guarda {"tipo": "api_pos", "url_base": ..., "tabla": ...} -- la
        API key NUNCA se guarda acá, mismo criterio que con la contraseña
        de SQL Server: se lee de la configuración local de Analytics al
        momento de actualizar, no queda escrita dentro del .hadarproy.
        NUNCA incluye la contraseña de SQL Server -- esa se pide de nuevo cada vez que se
        actualiza (decisión del usuario, más seguro que guardarla en un
        .hadarproy que es un .zip común, fácil de abrir con 7-Zip).
    procedencia: lista de dicts (BitacoraProcedencia.a_lista()) con la historia
        de las columnas -- hoy, cuáles calculó Hadar y con qué fórmula. Los
        proyectos guardados antes de que existiera simplemente no traen esta
        clave, y abren igual (con la bitácora vacía).
    """
    metadata = {
        "version": VERSION_FORMATO,
        "nombre_tabla_activa": nombre_tabla_activa,
        "tablas": list(tablas.keys()),
        "filtro_columna": filtro_columna,
        "filtro_valores": [_valor_json_seguro(v) for v in (filtro_valores or [])],
        "relaciones_ontologia": (
            [asdict(r) for r in relaciones_ontologia] if relaciones_ontologia is not None else None
        ),
        "notas_manuales": [
            [_valor_json_seguro(row_label), str(col_name), texto]
            for (row_label, col_name), texto in (notas_manuales or {}).items()
        ],
        "indicadores": [ind.to_dict() for ind in (indicadores or [])],
        "ml_activado": bool(ml_activado),
        "linea_base_ml": linea_base_ml or {},
        "fuentes_datos": fuentes_datos or {},
        "ml_multivariado_activado": bool(ml_multivariado_activado),
        "procedencia": procedencia or [],
        "prediccion": prediccion or {},
        # default=str por si algún valor raro se coló en la configuración.
        "panel": json.loads(json.dumps(panel or {}, default=str)),
        "tablero": {},
        "tablero_pronosticos": json.loads(json.dumps(tablero_pronosticos or [], default=str)),
        "tablero_retornos": json.loads(json.dumps(tablero_retornos or [], default=str)),
    }

    # Predicciones del tablero: guardarlas nunca debe impedir guardar el
    # proyecto. Si alguna tabla de predicciones no se puede escribir (por
    # ejemplo, un índice raro), esa se omite y el resto se guarda igual.
    tablero_ok = {}
    for nombre, datos in (tablero_tablas or {}).items():
        try:
            buffer = io.BytesIO()
            datos.to_parquet(buffer)
            tablero_ok[nombre] = buffer.getvalue()
            metadata["tablero"][nombre] = (tablero_meta or {}).get(nombre, {})
        except Exception:
            continue

    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)

    with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("proyecto.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        for nombre, df in tablas.items():
            buffer = io.BytesIO()
            df.to_parquet(buffer)
            z.writestr(f"tablas/{nombre}.parquet", buffer.getvalue())
        for nombre, contenido in tablero_ok.items():
            z.writestr(f"tablero/{nombre}.parquet", contenido)


class ProyectoCargado:
    """Lo que devuelve abrir_proyecto(): datos ya reconstruidos, listos
    para que HadarApp los aplique a su propio estado."""

    def __init__(self, tablas, nombre_tabla_activa, relaciones_ontologia,
                 filtro_columna, filtro_valores, notas_manuales, indicadores_dict,
                 ml_activado=False, linea_base_ml=None, fuentes_datos=None,
                 ml_multivariado_activado=False, procedencia=None, prediccion=None,
                 panel=None, tablero_meta=None, tablero_tablas=None, tablero_pronosticos=None,
                 tablero_retornos=None):
        self.tablero_retornos = tablero_retornos or []         # [] en proyectos antiguos
        self.tablero_pronosticos = tablero_pronosticos or []   # [] en proyectos antiguos
        self.tablero_meta = tablero_meta or {}       # {} en proyectos antiguos
        self.tablero_tablas = tablero_tablas or {}   # {nombre_tabla: DataFrame con las predicciones}
        self.panel = panel or {}               # config del Panel (dict); {} en proyectos antiguos
        self.tablas = tablas
        self.nombre_tabla_activa = nombre_tabla_activa
        self.relaciones_ontologia = relaciones_ontologia
        self.filtro_columna = filtro_columna
        self.filtro_valores = filtro_valores
        self.notas_manuales = notas_manuales
        self.indicadores_dict = indicadores_dict  # lista de dicts -- convertir con Indicador.from_dict
        self.ml_activado = ml_activado
        self.linea_base_ml = linea_base_ml or {}
        self.fuentes_datos = fuentes_datos or {}
        self.ml_multivariado_activado = ml_multivariado_activado
        self.prediccion = prediccion or {}     # config de la pestaña Predicción (dict)
        self.procedencia = procedencia or []   # lista de dicts -- ver procedencia.BitacoraProcedencia.desde_lista


def abrir_proyecto(ruta):
    with zipfile.ZipFile(ruta, "r") as z:
        metadata = json.loads(z.read("proyecto.json").decode("utf-8"))

        tablas = {}
        for nombre in metadata["tablas"]:
            with z.open(f"tablas/{nombre}.parquet") as f:
                tablas[nombre] = pd.read_parquet(f)

        tablero_meta = {}
        tablero_tablas = {}
        for nombre, meta in (metadata.get("tablero") or {}).items():
            ruta_interna = f"tablero/{nombre}.parquet"
            if ruta_interna in z.namelist():
                with z.open(ruta_interna) as f:
                    tablero_tablas[nombre] = pd.read_parquet(f)
                tablero_meta[nombre] = meta

        relaciones_data = metadata.get("relaciones_ontologia")
        relaciones = None
        if relaciones_data is not None:
            relaciones = [
                RelacionSugerida(**{k: v for k, v in r.items() if k in _CAMPOS_RELACION})
                for r in relaciones_data
            ]

        notas_manuales = {
            (fila[0], fila[1]): fila[2] for fila in metadata.get("notas_manuales", [])
        }

        return ProyectoCargado(
            tablas=tablas,
            nombre_tabla_activa=metadata.get("nombre_tabla_activa"),
            relaciones_ontologia=relaciones,
            filtro_columna=metadata.get("filtro_columna"),
            filtro_valores=metadata.get("filtro_valores", []),
            notas_manuales=notas_manuales,
            indicadores_dict=metadata.get("indicadores", []),
            ml_activado=metadata.get("ml_activado", False),
            linea_base_ml=metadata.get("linea_base_ml", {}),
            fuentes_datos=metadata.get("fuentes_datos", {}),
            ml_multivariado_activado=metadata.get("ml_multivariado_activado", False),
            procedencia=metadata.get("procedencia", []),
            prediccion=metadata.get("prediccion", {}),
            panel=metadata.get("panel", {}),
            tablero_meta=tablero_meta,
            tablero_tablas=tablero_tablas,
            tablero_pronosticos=metadata.get("tablero_pronosticos", []),
            tablero_retornos=metadata.get("tablero_retornos", []),
        )


def info_rapida_proyecto(ruta):
    """Nombre + cuántas tablas tiene un .hadarproy, sin cargar los datos
    completos -- pensado para la pantalla de inicio con la lista de
    proyectos recientes. Devuelve None si el archivo no es válido."""
    try:
        with zipfile.ZipFile(ruta, "r") as z:
            metadata = json.loads(z.read("proyecto.json").decode("utf-8"))
        return {
            "nombre_archivo": os.path.splitext(os.path.basename(ruta))[0],
            "n_tablas": len(metadata.get("tablas", [])),
            "tabla_activa": metadata.get("nombre_tabla_activa"),
            "modificado": os.path.getmtime(ruta),
            "peso_bytes": os.path.getsize(ruta),
            "ml_activado": metadata.get("ml_activado", False),
        }
    except Exception:
        return None


def listar_proyectos_recientes(carpeta=None):
    """Todos los .hadarproy válidos de una carpeta, más recientes primero.
    Pensado para la pantalla de inicio (Etapa 2)."""
    carpeta = carpeta or ruta_carpeta_proyectos()
    resultados = []
    if not os.path.isdir(carpeta):
        return resultados
    for nombre_archivo in os.listdir(carpeta):
        if nombre_archivo.lower().endswith(EXTENSION):
            info = info_rapida_proyecto(os.path.join(carpeta, nombre_archivo))
            if info:
                info["ruta"] = os.path.join(carpeta, nombre_archivo)
                resultados.append(info)
    resultados.sort(key=lambda i: i["modificado"], reverse=True)
    return resultados