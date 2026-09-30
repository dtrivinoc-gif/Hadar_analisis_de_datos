"""
Carga y lectura de archivos de datos (CSV, Excel, Parquet, SQL) y la
consulta online de la UF. Depende solo de config.py.
"""
import os
import re
import sqlite3

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import requests
except ImportError:
    requests = None

try:
    import pymssql
except ImportError:
    pymssql = None

from .config import FILAS_UMBRAL_MASIVO, TAMANO_UMBRAL_MASIVO_BYTES



class SqlMultipleTablesError(Exception):
    """Se lanza cuando un archivo .sql contiene más de una tabla y hay que
    preguntarle al usuario cuál cargar."""
    def __init__(self, tables):
        self.tables = tables
        super().__init__("El archivo .sql contiene varias tablas")


def read_sql_file(file_path: str, table_name: str = None) -> pd.DataFrame:
    """Ejecuta un script .sql (CREATE TABLE / INSERT, típico de un dump) en una
    base SQLite en memoria y devuelve una tabla como DataFrame."""
    import sqlite3

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        script = f.read()

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(script)
    except Exception as e:
        conn.close()
        raise ValueError(f"No se pudo ejecutar el script SQL: {e}")

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]

    if not tables:
        conn.close()
        raise ValueError("El archivo .sql no contiene ninguna tabla (CREATE TABLE) reconocible.")

    if table_name is None:
        if len(tables) == 1:
            table_name = tables[0]
        else:
            conn.close()
            raise SqlMultipleTablesError(tables)

    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    return df


def _parquet_row_count(file_path: str):
    """Intenta leer el N° de filas desde los metadatos del Parquet, sin
    cargar los datos. Devuelve None si no se puede determinar."""
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(file_path).metadata.num_rows
    except Exception:
        return None


def _decidir_motor(file_path: str, ext: str) -> str:
    """Decide si conviene usar 'polars' o 'pandas' para este archivo.

    - Para .parquet: se consulta el N° de filas real vía metadatos (barato).
    - Para .csv/.xlsx/.xls: no hay forma barata de saber el N° de filas sin
      leer el archivo, así que se usa el tamaño en disco como heurística.
    - Para .sql: siempre pandas (pasa por sqlite de todas formas).
    """
    if pl is None:
        return "pandas"

    if ext == ".parquet":
        n_rows = _parquet_row_count(file_path)
        if n_rows is not None:
            return "polars" if n_rows >= FILAS_UMBRAL_MASIVO else "pandas"
        # si no pudimos leer metadatos, caemos al tamaño en disco
    if ext == ".sql":
        return "pandas"

    try:
        tamano = os.path.getsize(file_path)
    except OSError:
        return "pandas"
    return "polars" if tamano >= TAMANO_UMBRAL_MASIVO_BYTES else "pandas"


def _cargar_con_polars(file_path: str, ext: str) -> pd.DataFrame:
    """Lee el archivo con Polars (rápido, multi-thread) y lo convierte a
    pandas al final, para que el resto de la app no necesite saber qué
    motor se usó."""
    if ext == ".parquet":
        df_pl = pl.read_parquet(file_path)
    elif ext == ".csv":
        try:
            df_pl = pl.read_csv(file_path, infer_schema_length=10_000, try_parse_dates=True)
        except Exception:
            # Polars es más estricto con encodings/separadores raros;
            # si falla, probamos forzando encoding latin-1 antes de rendirnos
            # a Pandas más abajo.
            df_pl = pl.read_csv(file_path, infer_schema_length=10_000,
                                 try_parse_dates=True, encoding="utf8-lossy")
    elif ext in (".xlsx", ".xls"):
        df_pl = pl.read_excel(file_path)
    else:
        raise ValueError(f"Extensión no soportada por el motor Polars: {ext}")

    return df_pl.to_pandas()


def _excel_sheet_names(file_path: str):
    """Devuelve la lista de nombres de hojas de un archivo Excel (.xlsx/.xls),
    o None si no se pudo leer (archivo dañado, no es Excel, etc.)."""
    try:
        with pd.ExcelFile(file_path) as xls:
            return xls.sheet_names
    except Exception:
        return None


def load_data(file_path: str):
    """Carga un archivo de datos y devuelve (DataFrame de pandas, motor_usado).

    Decide automáticamente entre Polars (datasets masivos) y Pandas (el resto).
    Sin importar el motor elegido, el resultado que ve el resto de la app es
    siempre un pandas.DataFrame -- así la tabla, los filtros, las estadísticas
    y los gráficos no necesitan saber nada sobre Polars.
    """
    ext = os.path.splitext(file_path)[1].lower()
    motor = _decidir_motor(file_path, ext)

    if motor == "polars" and ext in (".parquet", ".csv", ".xlsx", ".xls"):
        try:
            return _cargar_con_polars(file_path, ext), "polars"
        except Exception:
            # Si Polars falla por cualquier motivo (formato raro, encoding,
            # etc.), no bloqueamos al usuario: reintentamos con Pandas.
            motor = "pandas"

    # ---- Camino Pandas (archivos chicos/medianos, .sql, o fallback) ----
    if ext == ".csv":
        encodings = ["utf-8", "latin-1", "cp1252"]
        last_error = None
        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc, sep=None, engine="python"), "pandas"
            except Exception as e:
                last_error = e
                continue
        raise ValueError(f"Error al leer CSV: {last_error}")
    elif ext == ".xls":
        try:
            return pd.read_excel(file_path, engine="xlrd"), "pandas"
        except Exception:
            return pd.read_excel(file_path), "pandas"
    elif ext == ".parquet":
        try:
            return pd.read_parquet(file_path), "pandas"
        except ImportError as e:
            raise ValueError(
                "Para leer archivos Parquet hace falta instalar 'pyarrow' "
                "(ejecuta: pip install pyarrow)."
            ) from e
        except Exception as e:
            raise ValueError(f"Error al leer Parquet: {e}")
    elif ext == ".sql":
        return read_sql_file(file_path), "pandas"
    else:
        return pd.read_excel(file_path), "pandas"


class SqlServerNoDisponible(Exception):
    """Se lanza si falta pymssql instalado, o si la conexión/consulta falló
    (servidor apagado, credenciales malas, firewall, etc.) -- siempre con
    un mensaje en español, listo para mostrarlo tal cual en un QMessageBox."""
    pass


def _conectar_sql_server(servidor, puerto, base_datos, usuario, password):
    if pymssql is None:
        raise SqlServerNoDisponible(
            "Falta instalar la librería 'pymssql' para conectarse a SQL Server "
            "(ejecuta: pip install pymssql)."
        )
    try:
        return pymssql.connect(
            server=servidor, port=str(puerto), database=base_datos,
            user=usuario, password=password, timeout=8, login_timeout=8,
        )
    except Exception as e:
        raise SqlServerNoDisponible(
            f"No se pudo conectar a {servidor}:{puerto} ({base_datos}). "
            f"Revisa que el servidor esté encendido, acepte conexiones remotas, "
            f"y que el usuario/contraseña sean correctos.\n\nDetalle: {e}"
        ) from e


def listar_tablas_sql_server(servidor, puerto, base_datos, usuario, password):
    """Nombres de todas las tablas de esa base de datos -- para dejar
    elegir cuál(es) cargar, igual que ya se hace con un archivo .sql."""
    conn = _conectar_sql_server(servidor, puerto, base_datos, usuario, password)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME"
        )
        return [row[0] for row in cur.fetchall()]
    except Exception as e:
        raise SqlServerNoDisponible(f"No se pudo leer la lista de tablas: {e}") from e
    finally:
        conn.close()


def leer_tabla_sql_server(servidor, puerto, base_datos, usuario, password, tabla):
    """Trae una tabla completa de un SQL Server en vivo (local o de otro
    PC en la red) como DataFrame -- a diferencia de read_sql_file(), esto
    SÍ es una conexión real a un servidor, no un archivo .sql leído en una
    base temporal. Los nombres de tabla se arman con corchetes para que
    funcionen aunque tengan espacios o guiones."""
    conn = _conectar_sql_server(servidor, puerto, base_datos, usuario, password)
    try:
        return pd.read_sql_query(f"SELECT * FROM [{tabla}]", conn)
    except Exception as e:
        raise SqlServerNoDisponible(f"No se pudo leer la tabla '{tabla}': {e}") from e
    finally:
        conn.close()


class PosNoDisponible(Exception):
    """Se lanza si no se pudo conectar a la API del POS (apagado, dirección
    equivocada, clave inválida, etc.) -- siempre con un mensaje en español,
    listo para mostrarlo tal cual en un QMessageBox. Rol equivalente a
    SqlServerNoDisponible, pero para la fuente 'api_pos'."""
    pass


def verificar_conexion_pos(url_base):
    """Prueba si el POS está prendido y accesible en la red, SIN necesitar
    la API key (usa /salud). Pensado para un botón 'Probar conexión' en el
    diálogo de configuración de la fuente, antes de guardar nada. Devuelve
    True/False en vez de lanzar excepción."""
    if requests is None:
        return False
    try:
        resp = requests.get(f"{url_base.rstrip('/')}/salud", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def _pedir_api_pos(url_base, api_key, ruta):
    """Función interna compartida por listar_tablas_pos() y leer_tabla_pos():
    arma el pedido HTTP, valida el resultado y traduce cualquier problema a
    un mensaje en español entendible."""
    if requests is None:
        raise PosNoDisponible(
            "Falta instalar la librería 'requests' para conectarse al POS "
            "(ejecuta: pip install requests)."
        )
    try:
        resp = requests.get(
            f"{url_base.rstrip('/')}{ruta}",
            headers={"X-API-Key": api_key},
            timeout=6.0,
        )
    except Exception as e:
        raise PosNoDisponible(
            f"No se pudo conectar al POS en {url_base}. Revisa que esté "
            f"encendido, conectado a la misma red wifi, y que la dirección "
            f"sea correcta.\n\nDetalle: {e}"
        ) from e

    if resp.status_code == 401:
        raise PosNoDisponible("La API key configurada no es válida.")
    if resp.status_code == 404:
        raise PosNoDisponible(f"El POS no reconoce '{ruta}'. ¿La tabla existe?")
    if resp.status_code != 200:
        detalle = ""
        try:
            cuerpo = resp.json()
            detalle = f" Detalle: {cuerpo.get('tipo', '')}: {cuerpo.get('error', '')}"
        except Exception:
            pass
        raise PosNoDisponible(f"El POS respondió con un error ({resp.status_code}).{detalle}")

    return resp.json()


def listar_tablas_pos(url_base, api_key):
    """Nombres de todas las tablas que el POS deja sincronizar -- para
    dejar elegir cuál(es) cargar, igual que listar_tablas_sql_server()."""
    data = _pedir_api_pos(url_base, api_key, "/tablas")
    return data.get("tablas", [])


def leer_tabla_pos(url_base, api_key, tabla):
    """Trae una tabla completa desde la API del POS (otro PC en la misma
    red wifi) como DataFrame -- mismo rol que leer_tabla_sql_server(), pero
    hablando HTTP con api_pos.py en vez de un servidor SQL Server."""
    filas = _pedir_api_pos(url_base, api_key, f"/tabla/{tabla}")
    return pd.DataFrame(filas)


def fetch_uf_online():
    if requests is None:
        return None
    try:
        resp = requests.get("https://mindicador.cl/api/uf", timeout=4.0)
        resp.raise_for_status()
        data = resp.json()
        return float(data["serie"][0]["valor"])
    except Exception:
        return None


def cast_valor_a_dtype(valor_str: str, dtype):
    """Intenta convertir un string ingresado por el usuario al dtype original
    de la columna. Si no calza, cae de vuelta a string. Un '' vacío se
    interpreta como NaN."""
    if valor_str == "":
        return np.nan
    try:
        if pd.api.types.is_integer_dtype(dtype):
            return int(float(valor_str.replace(",", ".")))
        if pd.api.types.is_float_dtype(dtype):
            return float(valor_str.replace(",", "."))
        if pd.api.types.is_bool_dtype(dtype):
            return valor_str.strip().lower() in ("1", "true", "verdadero", "si", "sí", "x")
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.to_datetime(valor_str, dayfirst=True, errors="raise")
    except (ValueError, TypeError):
        pass
    return valor_str