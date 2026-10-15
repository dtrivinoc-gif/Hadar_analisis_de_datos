"""
retorno.py — El circuito de retorno: comparar lo que Hadar predijo con lo que
de verdad pasó. Sin Qt.

Una predicción que nadie vuelve a mirar no enseña nada. Aquí se toma lo que
la pestaña Predicción dejó en el tablero (tablero.py) y se compara con la
tabla donde ya está lo real. El resultado dice, en palabras simples, si el
modelo acertó tanto como prometía cuando se entrenó.

Reglas que se respetan:
  * Honesto: con pocos casos lo dice («una señal, no una medición») y no
    pone nota de «bien» o «mal».
  * Siempre contra algo: contra lo que el modelo prometía al entrenar y
    contra la respuesta tonta (siempre el promedio / siempre la clase más
    frecuente).
  * Sin inventar: si no se puede emparejar cada predicción con su fila real,
    se dice por qué y no se calcula nada.
  * Sin pickle ni modelos: solo se usan los números ya guardados.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

try:
    from .formato import dec
    from .prediccion import _fmt, evaluar_calibracion
except ImportError:
    from formato import dec
    from prediccion import _fmt, evaluar_calibracion

MIN_CASOS = 30            # bajo esto: «una señal, no una medición»
TOLERANCIA_ERROR = 1.25   # error real hasta 1,25 veces el esperado = «como se esperaba»
ERROR_GRAVE = 2.0         # desde 2 veces el esperado = «mucho peor»

# Qué se le dice al usuario según el nivel (los mismos textos en el cuadro de resultado y en el Panel)
TITULAR_NIVEL = {
    "bien": "Acertó como se esperaba.",
    "atencion": "Acertó menos de lo esperado.",
    "alerta": "Acertó mucho menos de lo esperado.",
    "info": "Resultado de la comparación.",
}

_NIVEL_ORDEN = {"bien": 0, "info": 1, "atencion": 2, "alerta": 3}


def _peor(a, b):
    return a if _NIVEL_ORDEN.get(a, 0) >= _NIVEL_ORDEN.get(b, 0) else b


def _pct(x):
    return f"{dec(float(x) * 100, 0)}%"


def _norm_clase(v):
    """1, 1.0 y «1» son la misma clase; «Sí» y «sí » también."""
    try:
        f = float(v)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
        return str(v).strip()
    except (TypeError, ValueError):
        return str(v).strip()


def referencia_de(resultado) -> dict:
    """Los números de validación con los que se entrenó el modelo (lo que 'prometía'),
    en un dict simple de floats, para guardarlos junto a las predicciones."""
    ref = {}
    m = getattr(resultado, "metricas", None) or {}
    b = getattr(resultado, "baseline", None) or {}
    for k in ("mae", "rmse", "r2", "accuracy", "bal_acc", "auc"):
        v = m.get(k)
        if v is not None and np.isfinite(v):
            ref[k] = float(v)
    for k in ("accuracy", "bal_acc", "rmse"):
        v = b.get(k)
        if v is not None and np.isfinite(v):
            ref["base_" + k] = float(v)
    cal = getattr(resultado, "calibracion", None) or {}
    if cal.get("brier") is not None:
        ref["brier"] = float(cal["brier"])
    n = getattr(resultado, "n_validacion", None)
    if n:
        ref["n_validacion"] = int(n)
    return ref


def leer_tabla_archivo(ruta: str) -> pd.DataFrame:
    """Lee un archivo con lo real (CSV, Excel o Parquet) en una tabla. Errores en palabras simples."""
    ext = str(ruta).lower().rsplit(".", 1)[-1] if "." in str(ruta) else ""
    try:
        if ext in ("csv", "txt", "tsv"):
            for codificacion in ("utf-8-sig", "latin-1"):
                try:
                    return pd.read_csv(ruta, sep=None, engine="python", encoding=codificacion)
                except UnicodeDecodeError:
                    continue
            raise ValueError("no pude leer las letras del archivo (acentos, ñ…)")
        if ext in ("xlsx", "xlsm", "xls"):
            return pd.read_excel(ruta)
        if ext == "parquet":
            return pd.read_parquet(ruta)
    except ValueError as e:
        raise ValueError(f"No pude leer el archivo: {e}") from e
    except Exception as e:                                   # archivo dañado, protegido, vacío…
        raise ValueError(f"No pude leer el archivo ({type(e).__name__}). ¿Está abierto en otro programa o dañado?") from e
    raise ValueError("Ese tipo de archivo no lo conozco. Usa CSV, Excel (.xlsx) o Parquet.")


# ----------------------------------------------------------------------
# Emparejar cada predicción con su fila real
# ----------------------------------------------------------------------
def emparejar(pub, df_real: pd.DataFrame, col_real: str, col_id: str | None = None):
    """Devuelve (predicciones, reales, como) con las filas en el mismo orden.
    como = «id» (por la columna que identifica cada fila) u «orden» (misma
    posición en las dos tablas). Lanza ValueError, en palabras simples, si no
    se puede emparejar con seguridad."""
    if col_real not in df_real.columns:
        raise ValueError(f"La tabla con lo real no tiene la columna «{col_real}».")
    if col_id:
        if col_id not in df_real.columns:
            raise ValueError(f"La tabla con lo real no tiene la columna «{col_id}».")
        if pub.ids is None:
            raise ValueError(
                "Estas predicciones no guardaron una columna que identifique cada fila, así que no se pueden "
                "emparejar por id. Elige «mismo orden de filas» si las dos tablas tienen las mismas filas, "
                "en el mismo orden.")
        ids_real = df_real[col_id]
        if not ids_real.is_unique:
            raise ValueError(f"«{col_id}» se repite en la tabla con lo real: no sirve para emparejar las filas.")
        idx = pd.Index(ids_real)
        pos = idx.get_indexer(pub.ids)
        if (pos < 0).all():                                   # ids guardados como número en uno y texto en el otro
            pos = pd.Index(ids_real.astype(str)).get_indexer(pub.ids.astype(str))
        hallados = pos >= 0
        if not hallados.any():
            raise ValueError(f"Ninguna fila de las predicciones aparece en la tabla con lo real (por «{col_id}»).")
        real = df_real[col_real].iloc[pos[hallados]].reset_index(drop=True)
        pred = pub.datos[hallados].reset_index(drop=True)
        return pred, real, "id"
    if len(df_real) != len(pub.datos):
        raise ValueError(
            f"Sin una columna que identifique las filas, las dos tablas deben tener exactamente las mismas filas "
            f"en el mismo orden: las predicciones son {len(pub.datos)} y la tabla con lo real tiene {len(df_real)}.")
    return pub.datos.reset_index(drop=True), df_real[col_real].reset_index(drop=True), "orden"


# ----------------------------------------------------------------------
# Un comparador por tipo de salida
# ----------------------------------------------------------------------
def _regresion(pred, real, pub, col_real):
    p = pd.to_numeric(pred.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(real, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    n = int(ok.sum())
    if n == 0:
        raise ValueError(f"«{col_real}» no tiene números en la tabla con lo real.")
    p, y = p[ok], y[ok]
    e = p - y
    mae, sesgo = float(np.mean(np.abs(e))), float(np.mean(e))
    escala = float(np.mean(np.abs(y)))
    base = float(np.mean(np.abs(y - y.mean())))
    nums = {"n": n, "mae": mae, "sesgo": sesgo, "mae_base": base}
    lineas = [f"Se compararon {dec(n, 0, True)} filas con su valor real de «{col_real}»."]
    t = f"Error promedio: {_fmt(mae)}"
    if escala > 0:
        t += f" ({_pct(mae / escala)} del valor típico)"
    lineas.append(t + ".")
    nivel = "info"
    ref = pub.referencia.get("mae")
    if ref and ref > 0:
        ratio = mae / ref
        nums["veces_lo_esperado"] = ratio
        lineas.append(f"Al entrenar, el error esperado era {_fmt(ref)}; ahora es {dec(ratio, 1)} veces ese.")
        nivel = "bien" if ratio <= TOLERANCIA_ERROR else ("atencion" if ratio <= ERROR_GRAVE else "alerta")
    if mae >= base:
        lineas.append("Ni siquiera supera a responder siempre el promedio de lo real.")
        nivel = _peor(nivel, "atencion")
    if mae > 0 and abs(sesgo) > 0.25 * mae:
        lineas.append(f"Tendió a predecir de {'más' if sesgo > 0 else 'menos'}: en promedio {_fmt(abs(sesgo))}.")
    return n, nums, lineas, nivel


def _clase(pred, real, pub, col_real):
    if pub.modo == "probabilidades":
        cols = list(pred.columns)
        elegido = pred.to_numpy(dtype=float).argmax(axis=1)
        yp = np.array([_norm_clase(pub.etiquetas.get(cols[i], cols[i])) for i in elegido])
    else:
        yp = np.array([_norm_clase(v) for v in pred.iloc[:, 0]])
    yr = np.array([_norm_clase(v) for v in real])
    n = len(yr)
    acc = float(np.mean(yp == yr))
    clases, cuentas = np.unique(yr, return_counts=True)
    frecuente, base = str(clases[cuentas.argmax()]), float(cuentas.max() / n)
    recalls = [float(np.mean(yp[yr == c] == c)) for c in clases]
    bal = float(np.mean(recalls))
    nums = {"n": n, "accuracy": acc, "bal_acc": bal, "base_accuracy": base}
    lineas = [f"Se compararon {dec(n, 0, True)} filas con su resultado real de «{col_real}».",
              f"Acertó {_pct(acc)} de los casos."]
    nivel = "info"
    ref_bal, ref_acc = pub.referencia.get("bal_acc"), pub.referencia.get("accuracy")
    ref, medido, nombre = (ref_bal, bal, "con error parejo entre clases") if ref_bal is not None else \
        (ref_acc, acc, "")
    if ref is not None:
        tol = 0.05 + 0.5 / np.sqrt(n)
        lineas.append(f"Al entrenar acertaba {_pct(ref)}{(' ' + nombre) if nombre else ''}; ahora {_pct(medido)}.")
        nivel = "bien" if medido >= ref - tol else ("atencion" if medido >= ref - 2 * tol else "alerta")
    if acc <= base + 0.005:
        lineas.append(f"No mejora a responder siempre «{frecuente}» ({_pct(base)}).")
        nivel = _peor(nivel, "atencion")
    return n, nums, lineas, nivel


def _probabilidad(pred, real, pub, col_real):
    p = pd.to_numeric(pred.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    pos = _norm_clase(pub.clase_positiva)
    y = np.array([1 if _norm_clase(v) == pos else 0 for v in real])
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    n = len(y)
    if n == 0:
        raise ValueError("Las predicciones no traen probabilidades que comparar.")
    positivos = int(y.sum())
    cal = evaluar_calibracion(p, y)
    nums = {"n": n, "positivos": positivos, "brier": cal.get("brier"), "brier_base": cal.get("brier_base"),
            "ece": cal.get("ece"), "tolerancia": cal.get("tolerancia")}
    lineas = [f"Se compararon {dec(n, 0, True)} filas con su resultado real de «{col_real}»: "
              f"«{pub.clase_positiva}» ocurrió en {dec(positivos, 0, True)} ({_pct(positivos / n)})."]
    nivel = "info"
    if cal.get("brier") is not None:
        lineas.append(f"Error de las probabilidades (Brier): {dec(cal['brier'], 3)}; responder siempre la frecuencia "
                      f"general ({_pct(positivos / n)}) daría {dec(cal['brier_base'], 3)}.")
        if cal["brier"] >= cal["brier_base"]:
            lineas.append("No mejora a decir siempre la frecuencia general.")
            nivel = _peor(nivel, "atencion")
    if cal.get("verificable"):
        tabla, tol, ece = cal["tabla"], cal["tolerancia"], cal["ece"]
        veredicto = "bien" if ece <= tol else ("atencion" if ece <= 2 * tol else "alerta")
        nivel = _peor("bien" if nivel == "info" else nivel, veredicto)   # «info» = todavía sin juicio
        alto = tabla[-1]
        lineas.append(f"De los {dec(alto['casos'], 0, True)} casos con la probabilidad más alta (entre {_pct(alto['desde'])} "
                      f"y {_pct(alto['hasta'])}), ocurrió en el {_pct(alto['real'])}.")
        if ece <= tol:
            lineas.append("Las probabilidades decían la verdad: lo que prometían se parece a lo que ocurrió.")
        else:
            peor = max(tabla, key=lambda f: abs(f["estimada"] - f["real"]))
            lineas.append(f"Ojo: a los casos con probabilidad entre {_pct(peor['desde'])} y {_pct(peor['hasta'])} les dio "
                          f"cerca de {_pct(peor['estimada'])}, pero ocurrió en el {_pct(peor['real'])} "
                          f"({dec(peor['casos'], 0, True)} casos).")
    elif cal.get("motivo"):
        lineas.append("No se puede juzgar si las probabilidades dicen la verdad: " + cal["motivo"] + ".")
    return n, nums, lineas, nivel


def _ranking(pred, real, pub, col_real):
    from sklearn.metrics import roc_auc_score
    p = pd.to_numeric(pred.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    pos = _norm_clase(pub.clase_positiva)
    y = np.array([1 if _norm_clase(v) == pos else 0 for v in real])
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    n = len(y)
    if n == 0 or y.min() == y.max():
        raise ValueError("Para juzgar el orden hacen falta casos donde ocurrió y casos donde no ocurrió.")
    auc = float(roc_auc_score(y, p))
    nums = {"n": n, "positivos": int(y.sum()), "auc": auc}
    lineas = [f"Se compararon {dec(n, 0, True)} filas con su resultado real de «{col_real}».",
              f"Ordenó bien los casos: si tomas uno que sí ocurrió y uno que no, el puntaje era mayor en el que "
              f"ocurrió el {_pct(auc)} de las veces (azar puro sería 50%)."]
    nivel = "info"
    ref = pub.referencia.get("auc")
    if ref is not None:
        tol = 0.05 + 0.5 / np.sqrt(n)
        lineas.append(f"Al entrenar era {_pct(ref)}; ahora {_pct(auc)}.")
        nivel = "bien" if auc >= ref - tol else ("atencion" if auc >= ref - 2 * tol else "alerta")
    if auc <= 0.55:
        lineas.append("Casi no ordena mejor que el azar.")
        nivel = _peor(nivel, "atencion")
    lineas.append("Es un puntaje para ordenar, no una probabilidad: no se juzga si «dice la verdad».")
    return n, nums, lineas, nivel


_COMPARADORES = {"regresion": _regresion, "clase": _clase, "probabilidades": _clase,
                 "probabilidad": _probabilidad, "ranking": _ranking}


def comparar(pub, df_real: pd.DataFrame, col_real: str, col_id: str | None = None, comparado_en=None) -> dict:
    """Compara las predicciones `pub` (PrediccionesPublicadas) con la tabla `df_real`.
    Devuelve un dict simple (listo para guardar en el proyecto) con las frases en palabras simples."""
    pred, real, como = emparejar(pub, df_real, col_real, col_id)
    con_real = real.notna().to_numpy()
    n_sin_real = int((~con_real).sum())
    pred, real = pred[con_real].reset_index(drop=True), real[con_real].reset_index(drop=True)
    if len(real) == 0:
        raise ValueError(f"En la tabla con lo real, esas filas no traen el valor de «{col_real}» (están vacías).")
    modo = pub.modo or ("regresion" if pub.tipo_salida == "valor" else "clase")
    comparador = _COMPARADORES.get(modo)
    if comparador is None:
        raise ValueError("No sé comparar este tipo de predicción todavía.")
    n, numeros, lineas, nivel = comparador(pred, real, pub, col_real)
    if como == "orden":
        lineas.append("Las filas se emparejaron por su orden (misma posición en las dos tablas).")
    if n_sin_real:
        lineas.append(f"{dec(n_sin_real, 0, True)} filas no se compararon porque todavía no tienen valor real.")
    if n < MIN_CASOS:
        lineas.append(f"Son pocos casos ({n}): tómalo como una señal, no como una medición.")
        nivel = "info"
    return {
        "tabla": pub.tabla, "objetivo": pub.objetivo, "modo": modo, "emparejadas_por": como,
        "n_predichas": int(len(pub.datos)), "n_comparadas": int(n), "n_sin_real": n_sin_real,
        "predicho_en": pub.calculado_en,
        "comparado_en": comparado_en or datetime.now().strftime("%d/%m/%Y %H:%M"),
        "col_real": str(col_real), "col_id": str(col_id or ""),
        "numeros": {k: (float(v) if v is not None else None) for k, v in numeros.items()},
        "lineas": lineas, "nivel": nivel,
    }
