"""
foto_comportamiento.py — «Foto» del comportamiento de entrenar() y analizar_agotamiento() con datos inventados y semillas fijas.
Sirve para comprobar que un refactor NO cambió ningún resultado: se saca la foto ANTES, se cambia el código, se saca DESPUÉS
y se comparan.

    python herramientas/foto_comportamiento.py guardar antes.json
    ... (se cambia el código) ...
    python herramientas/foto_comportamiento.py comparar antes.json     # debe decir «IDÉNTICO»
"""
import json, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import prediccion as P
import series_tiempo as st
from prediccion import ConfigPrediccion as CP
from pronosticos import ConfigPronostico as C
import formato
formato.set_formato_numerico("estandar")


def canon(x, nd=6):
    """Convierte cualquier cosa a algo comparable (JSON), redondeando decimales."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return "nan" if x != x else ("inf" if x == float("inf") else ("-inf" if x == float("-inf") else round(float(x), nd)))
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x))
    if isinstance(x, pd.DataFrame):
        return {"columnas": [str(c) for c in x.columns], "filas": [[canon(v, nd) for v in f] for f in x.itertuples(index=False)]}
    if isinstance(x, pd.Series):
        return [canon(v, nd) for v in x.tolist()]
    if isinstance(x, np.ndarray):
        return [canon(v, nd) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): canon(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [canon(v, nd) for v in (sorted(x, key=str) if isinstance(x, set) else x)]
    return f"<{type(x).__name__}>"


def foto_entrenamiento(r, df_pred):
    pred = P.predecir(r, df_pred)
    return canon({
        "tipo": r.tipo, "objetivo": r.objetivo, "columnas": r.columnas, "clases": r.clases, "metricas": r.metricas,
        "metricas_sd": r.metricas_sd, "baseline": r.baseline, "importancias": r.importancias,
        "n": (r.n_entrenamiento, r.n_validacion, r.n_iteraciones, r.n_pliegues), "advertencias": [str(a) for a in r.advertencias],
        "log_objetivo": r.log_objetivo, "salida": r.salida, "modo": r.modo_validacion, "balanceado": r.balanceado,
        "clase_minoritaria": r.clase_minoritaria, "tipo_salida": r.tipo_salida, "semilla": r.semilla,
        "calibracion": {k: v for k, v in (r.calibracion or {}).items() if k != "calibrador"}, "hay_calibrador": r.calibrador is not None,
        "validacion": r.validacion, "revision": r.revision, "mejoras_pliegue": r.mejoras_pliegue,
        "veredicto": r.veredicto(), "tarjetas": r.tarjetas(), "lineas": r.lineas_resumen(), "experto": r.detalle_experto(),
        "pred": pred, "pred_avisos": pred.attrs.get("avisos"), "pred_tipo": pred.attrs.get("tipo_salida"),
    })


def escenarios_entrenar():
    out = {}
    rng = np.random.default_rng(0)
    n = 700
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(list("abcd"), n); cat2 = rng.choice(["x", "y", None], n)
    texto = [f"cliente_{i % 120}" for i in range(n)]
    fecha = pd.date_range("2022-01-01", periods=n, freq="D"); grupo = rng.integers(0, 40, n)
    base = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "cat2": cat2, "texto": texto, "fecha": fecha, "grupo": grupo})
    # 1 regresión
    d = base.copy(); d["y"] = 3 * x1 - 2 * x2 + (cat == "a") * 4 + rng.normal(0, 1, n)
    out["regresion_aleatoria"] = (d, CP(objetivo="y", tipo="regresion", pliegues=3), ["x1", "x2", "cat", "cat2", "texto"])
    # 2 regresión asimétrica (log) y temporal
    d = base.copy(); d["y"] = np.exp(0.5 * x1 + rng.normal(0, .4, n)) * 100
    out["regresion_log_temporal"] = (d, CP(objetivo="y", tipo="regresion", validacion="temporal", columna_fecha="fecha", pliegues=2), ["x1", "x2", "cat"])
    # 3 binaria desbalanceada con pesos y probabilidad
    d = base.copy(); z = 1.4 * x1 + 0.5 * x2 - 3.2; d["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    out["binaria_pesos_prob"] = (d, CP(objetivo="y", tipo="clasificacion", salida="probabilidad", pliegues=3, balancear="si"), ["x1", "x2", "cat"])
    # 4 binaria equilibrada temporal
    d = base.copy(); d["y"] = (x1 + rng.normal(0, .8, n) > 0).astype(int)
    out["binaria_temporal"] = (d, CP(objetivo="y", tipo="clasificacion", validacion="temporal", columna_fecha="fecha", pliegues=2), ["x1", "x2", "cat", "fecha"])
    # 5 multiclase
    d = base.copy(); d["y"] = np.where(x1 > 0.5, "alto", np.where(x1 < -0.5, "bajo", "medio"))
    out["multiclase"] = (d, CP(objetivo="y", tipo="clasificacion", pliegues=1), ["x1", "x2", "cat"])
    # 6 grupos + duplicados + fuga numérica
    d = base.copy(); d["y"] = (x2 + rng.normal(0, .7, n) > 0).astype(int); d["pista"] = d.y.astype(float)
    d = pd.concat([d, d.sample(120, random_state=1)], ignore_index=True)
    out["grupos_duplicados_fuga"] = (d, CP(objetivo="y", tipo="clasificacion", validacion="grupos", columna_grupo="grupo", pliegues=3), ["x1", "x2", "pista", "texto"])
    # 7 pocos datos
    d = base.head(90).copy(); d["y"] = (d.x1 > 0).astype(int)
    out["pocos_datos"] = (d, CP(objetivo="y", tipo="clasificacion", pliegues=1), ["x1", "x2"])
    return out


def escenarios_agotamiento():
    rng = np.random.default_rng(1)
    dias = pd.date_range("2024-03-01", periods=150, freq="D")
    out = {}
    out["constante_stock_unico"] = (pd.DataFrame({"f": dias, "c": np.full(150, 5.0)}),
                                    C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", stock_actual=100))
    filas = []
    tasas = {"A": 10, "B": 3, "C": 50, "D": 0, "E": 7}; stock = {"A": 100, "B": 300, "C": 400, "D": 50, "E": 20}
    for p, tasa in tasas.items():
        n = 150 if p != "E" else 40                               # E sin registros recientes
        for d in dias[:n]:
            filas.append({"f": d, "p": p, "c": rng.poisson(tasa) * 1.0, "s": stock[p]})
    multi = pd.DataFrame(filas)
    out["varios_elementos"] = (multi, C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", columna_stock="s", columna_entidad="p"))
    stk = pd.DataFrame({"f": dias, "s": 2000 - np.cumsum(rng.poisson(6, 150)) * 1.0})
    out["sin_columna_de_consumo"] = (stk, C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_stock="s"))
    sube = pd.DataFrame({"f": dias, "c": np.linspace(2, 12, 150) + rng.normal(0, .5, 150), "p": "z", "s": 300.0})
    out["consumo_que_sube"] = (sube, C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", columna_stock="s", columna_entidad="p"))
    neg = pd.DataFrame({"f": dias, "c": rng.poisson(4, 150) * 1.0 - (rng.uniform(size=150) < .1) * 3})
    out["consumos_negativos"] = (neg, C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", stock_actual=60))
    out["fecha_de_corte"] = (multi, C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", columna_stock="s", columna_entidad="p", fecha_corte="2024-06-15"))
    out["stock_enorme_larga_duracion"] = (pd.DataFrame({"f": dias, "c": np.full(150, .2)}),
                                          C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", stock_actual=5000))
    return out


def tomar_foto():
    foto = {"entrenar": {}, "agotamiento": {}}
    for nombre, (df, cfg, cols) in escenarios_entrenar().items():
        try:
            r = P.entrenar(df, cfg, cols)
            foto["entrenar"][nombre] = foto_entrenamiento(r, df.head(60))
        except Exception as e:
            foto["entrenar"][nombre] = {"error": f"{type(e).__name__}: {e}"}
    for nombre, (df, cfg) in escenarios_agotamiento().items():
        try:
            r = st.analizar_agotamiento(df, cfg)
            foto["agotamiento"][nombre] = canon({"titular": r.titular, "nivel": r.nivel, "confianza": r.confianza, "porque": r.por_que_confianza,
                                                 "tarjetas": r.tarjetas, "detalle": r.detalle, "supuestos": r.supuestos,
                                                 "advertencias": [str(a) for a in r.advertencias], "tabla": r.tabla, "metricas": r.metricas,
                                                 "texto": r.texto()})
        except Exception as e:
            foto["agotamiento"][nombre] = {"error": f"{type(e).__name__}: {e}"}
    return foto


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("guardar", "comparar"):
        sys.exit(__doc__)
    foto = tomar_foto()
    if sys.argv[1] == "guardar":
        json.dump(foto, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1, sort_keys=True)
        n = sum(len(v) for v in foto.values())
        print(f"Foto guardada en {sys.argv[2]} ({n} escenarios).")
        for g, esc in foto.items():
            for k, v in esc.items():
                if isinstance(v, dict) and "error" in v:
                    print(f"  (aviso) {g}/{k}: {v['error']}")
    else:
        antes = json.load(open(sys.argv[2], encoding="utf-8"))
        despues = json.loads(json.dumps(foto, ensure_ascii=False, sort_keys=True))
        difs = []
        def comparar(a, b, ruta):
            if type(a) != type(b):
                difs.append((ruta, a, b)); return
            if isinstance(a, dict):
                for k in sorted(set(a) | set(b)):
                    if k not in a or k not in b: difs.append((f"{ruta}/{k}", a.get(k, "<falta>"), b.get(k, "<falta>")))
                    else: comparar(a[k], b[k], f"{ruta}/{k}")
            elif isinstance(a, list):
                if len(a) != len(b): difs.append((ruta + "[largo]", len(a), len(b))); return
                for i, (x, y) in enumerate(zip(a, b)): comparar(x, y, f"{ruta}[{i}]")
            elif a != b:
                difs.append((ruta, a, b))
        comparar(antes, despues, "")
        if not difs:
            print("IDÉNTICO: el refactor no cambió ningún resultado.")
        else:
            print(f"HAY {len(difs)} DIFERENCIAS (primeras 15):")
            for ruta, a, b in difs[:15]:
                print(f"  {ruta}\n     antes:   {str(a)[:150]}\n     después: {str(b)[:150]}")
            sys.exit(1)
