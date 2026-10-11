"""
autoprueba_pronosticos.py — Comprueba, con datos inventados donde la respuesta correcta se conoce,
que las «palabras clave» de Predicción entregan rangos que contienen la verdad.

Cómo usarla (con los .py en la misma carpeta):
    python autoprueba_pronosticos.py

Debe terminar con «TODO OK». Si algo falla, copia el texto de la línea con ✗ y mándamelo.
Tarda unos 20-40 segundos. Usa semillas fijas, así que el resultado es siempre el mismo.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from pronosticos import ConfigPronostico as C, ORDEN_PALABRAS, PALABRAS_CLAVE, ejecutar, sugerir_configuracion
except ImportError:
    from .pronosticos import ConfigPronostico as C, ORDEN_PALABRAS, PALABRAS_CLAVE, ejecutar, sugerir_configuracion

_resultados = []


def comprobar(nombre: str, condicion: bool, detalle: str = ""):
    _resultados.append(bool(condicion))
    print(("  ✓ " if condicion else "  ✗ ") + nombre + (f"  → {detalle}" if detalle else ""))


def correr(cfg, tablas):
    return ejecutar(cfg, tablas)


def prueba_series():
    print("Series de tiempo")
    rng = np.random.default_rng(1)
    f = pd.date_range("2023-01-01", periods=730, freq="D")
    dow = np.array([1.0, 1.0, 1.05, 1.1, 1.4, 1.7, 1.5])
    ventas = (100 + 0.08 * np.arange(730)) * dow[f.dayofweek] * rng.normal(1, 0.08, 730)
    df = pd.DataFrame({"fecha": f, "ventas": ventas})
    T = {"t": df}
    r = correr(C(palabra="estacionalidad", tabla="t", columna_fecha="fecha", columna_valor="ventas"), T)
    comprobar("estacionalidad: el día más alto es sábado", r.tarjetas[1][0] == "sábado", r.tarjetas[1][0])
    r = correr(C(palabra="tendencia", tabla="t", columna_fecha="fecha", columna_valor="ventas"), T)
    comprobar("tendencia: detecta que sube", r.metricas["sentido"] == "sube", r.titular[:90])
    # proyección: ocultar los últimos 30 días y ver si el rango los contiene
    pasado, real = df.iloc[:-30], df.ventas.iloc[-30:].sum()
    r = correr(C(palabra="proyeccion", tabla="t", columna_fecha="fecha", columna_valor="ventas", horizonte=30), {"t": pasado})
    esp = r.tabla["Esperado"].sum()
    comprobar("proyección: el valor esperado está a menos de 6% de lo real", abs(esp / real - 1) < 0.06,
              f"esperado {esp:,.0f} vs real {real:,.0f}")

    # umbral: la temperatura cruza 80 en un día conocido
    n = 300
    f2 = pd.date_range("2024-01-01", periods=n, freq="D")
    temp = 60 + 0.05 * np.arange(n) + rng.normal(0, 1.2, n)
    cruce = f2[0] + pd.Timedelta(days=(80 - 60) / 0.05)
    r = correr(C(palabra="umbral", tabla="t", columna_fecha="fecha", columna_valor="temperatura", umbral=80, horizonte=180),
               {"t": pd.DataFrame({"fecha": f2, "temperatura": temp})})
    t = r.tabla
    antes = t.loc[t["Máximo probable"] >= 80, "Fecha"].min()
    despues = t.loc[t["Mínimo probable"] >= 80, "Fecha"].min()
    comprobar("umbral: la fecha real de cruce cae dentro del rango", antes <= cruce <= despues,
              f"real {cruce.date()}, rango {antes.date()} a {despues.date()}")

    # agotamiento: tres productos con ritmo conocido (A: 10 días, B: 100 días, C: 8 días)
    dias = pd.date_range("2024-06-01", periods=90, freq="D")
    filas = []
    tasas, stock = {"A": 10, "B": 3, "C": 50}, {"A": 100, "B": 300, "C": 400}
    for p, tasa in tasas.items():
        for d in dias:
            filas.append({"fecha": d, "producto": p, "vendidas": rng.poisson(tasa), "stock": stock[p]})
    r = correr(C(palabra="agotamiento", tabla="t", columna_fecha="fecha", columna_valor="vendidas", columna_stock="stock",
                 columna_entidad="producto"), {"t": pd.DataFrame(filas)})
    tb = r.tabla.set_index("Elemento")
    verdad = {"A": 10, "B": 100, "C": 8}
    for k, v in verdad.items():
        lo, hi = tb.loc[k, "Si se gasta rápido (días)"], tb.loc[k, "Si se gasta lento (días)"]
        comprobar(f"agotamiento {k}: los {v} días reales están en el rango", lo <= v <= hi, f"rango {lo:.1f} a {hi:.1f}")
    comprobar("agotamiento: ordena primero el que se acaba antes", list(r.tabla["Elemento"])[0] == "C")

    # escenario: elasticidad real de -1.5 (precio +10% => ventas -13.3%)
    n = 400
    f3 = pd.date_range("2023-01-01", periods=n, freq="D")
    precio = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.5 + rng.normal(0, 0.08, n))
    vent = 1000 * (precio / 10) ** -1.5 * np.array([1, 1, 1, 1, 1.3, 1.6, 1.4])[f3.dayofweek] * rng.normal(1, 0.05, n)
    r = correr(C(palabra="escenario", tabla="t", columna_fecha="fecha", columna_valor="ventas", columna_driver="precio",
                 cambio_pct=10), {"t": pd.DataFrame({"fecha": f3, "ventas": vent, "precio": precio})})
    real = 1.1 ** -1.5 - 1
    comprobar("escenario: el efecto real (-13.3%) está en el rango", r.metricas["efecto_min"] <= real <= r.metricas["efecto_max"],
              f"rango {r.metricas['efecto_min'] * 100:.1f}% a {r.metricas['efecto_max'] * 100:.1f}%")

    # saturación
    h = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    trafico = (30 + 0.05 * np.arange(len(h))) * (1 + 0.4 * np.sin(2 * np.pi * (h.hour - 6) / 24)) + rng.normal(0, 2, len(h))
    r = correr(C(palabra="saturacion", tabla="t", columna_fecha="momento", columna_valor="trafico", capacidad=100,
                 frecuencia="D", horizonte=30), {"t": pd.DataFrame({"momento": h, "trafico": trafico})})
    comprobar("saturación: avisa riesgo (la demanda sube hacia la capacidad)", r.nivel in ("atencion", "alerta"), r.titular[:90])


def prueba_eventos():
    print("Eventos")
    rng = np.random.default_rng(0)
    ini = pd.Timestamp("2024-01-01")
    filas = []
    for m in range(60):
        t = 0.0
        while True:
            t += 100.0 * rng.weibull(2.0)
            if t > 400:
                break
            filas.append({"maquina": f"M{m:02d}", "fecha": ini + pd.Timedelta(days=float(t)), "tipo": "falla"})
        for _ in range(3):
            filas.append({"maquina": f"M{m:02d}", "fecha": ini + pd.Timedelta(days=float(rng.uniform(0, 400))), "tipo": "mantenimiento"})
    df = pd.DataFrame(filas)
    df["instalada"] = ini
    r = correr(C(palabra="vida_util", tabla="t", columna_fecha="fecha", columna_entidad="maquina", columna_evento="tipo",
                 valor_evento="falla", columna_inicio="instalada"), {"t": df})
    med = r.metricas["mediana_dias"]
    comprobar("vida útil: la mediana (real 83 días) queda a menos de 20%", abs(med / 83.3 - 1) < 0.2, f"estimada {med:.0f} días")
    r = correr(C(palabra="falla", tabla="t", columna_fecha="fecha", columna_entidad="maquina", columna_evento="tipo",
                 valor_evento="falla", columna_inicio="instalada", ventana_dias=30), {"t": df})
    col = [c for c in r.tabla.columns if c.startswith("Probabilidad en")][0]
    comprobar("falla: probabilidades entre 0 y 1", float(r.tabla[col].min()) >= 0 and float(r.tabla[col].max()) <= 1)
    edad = r.tabla.iloc[:, 1].to_numpy()
    comprobar("falla: con desgaste (forma 2) los más viejos tienen más riesgo",
              np.corrcoef(edad, r.tabla[col].to_numpy())[0, 1] > 0.5)
    # duración con casos abiertos
    n = 600
    inicio = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    dur = np.exp(rng.normal(np.log(10), 0.6, n))
    tipo = rng.choice(["A", "B"], n)
    dur = np.where(tipo == "B", dur * 2, dur)
    fin = inicio + pd.to_timedelta(dur, unit="D")
    corte = pd.Timestamp("2024-07-20")
    dd = pd.DataFrame({"inicio": inicio, "fin": fin.where(fin <= corte), "tipo": tipo})
    dd = dd[dd.inicio <= corte]
    r = correr(C(palabra="duracion", tabla="t", columna_inicio="inicio", columna_fin="fin", columna_grupo="tipo"), {"t": dd})
    tb = r.tabla.set_index("Grupo")["Tiempo típico (días)"]
    comprobar("duración: tipo A (real 10 días) y B (real 20) con menos de 20% de error",
              abs(tb["A"] / 10 - 1) < 0.2 and abs(tb["B"] / 20 - 1) < 0.2, f"A {tb['A']:.1f}, B {tb['B']:.1f}")
    # prioridad
    pr = pd.DataFrame({"e": [f"E{i}" for i in range(30)], "riesgo": rng.beta(1, 6, 30), "costo": rng.lognormal(7, 1, 30)})
    r = correr(C(palabra="prioridad", tabla="t", columna_riesgo="riesgo", columna_impacto="costo", columna_entidad="e"), {"t": pr})
    esperado = (pr.riesgo * pr.costo).idxmax()
    comprobar("prioridad: el primero es el de mayor riesgo × impacto", r.tabla.iloc[0]["Elemento"] == f"E{esperado}")


def prueba_errores():
    print("Mensajes claros cuando falta algo")
    casos = [
        ("faltan columnas", C(palabra="proyeccion", tabla="t"), {"t": pd.DataFrame({"a": range(10)})}),
        ("pocos datos", C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="v"),
         {"t": pd.DataFrame({"f": pd.date_range("2024-01-01", periods=8), "v": range(8)})}),
        ("columna de números como fecha", C(palabra="proyeccion", tabla="t", columna_fecha="v", columna_valor="v"),
         {"t": pd.DataFrame({"v": np.arange(40) * 1.5})}),
        ("agotamiento sin stock", C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="v"),
         {"t": pd.DataFrame({"f": pd.date_range("2024-01-01", periods=30), "v": range(30)})}),
    ]
    for nombre, cfg, tabla in casos:
        try:
            ejecutar(cfg, tabla)
            comprobar(f"{nombre}: debía avisar", False)
        except ValueError as e:
            comprobar(f"{nombre}: avisa con un mensaje en español", len(str(e)) > 10, str(e)[:70])
        except Exception as e:  # noqa
            comprobar(f"{nombre}: error NO controlado ({type(e).__name__})", False, str(e)[:70])
    comprobar("catálogo completo (12 palabras clave)", all(k in PALABRAS_CLAVE for k in ORDEN_PALABRAS) and len(ORDEN_PALABRAS) == 12)
    cfg = C(palabra="umbral", tabla="t", umbral=5.0, columnas_explicativas=["a"])
    comprobar("la configuración se guarda y se recupera igual", C.from_dict(cfg.to_dict()) == cfg)
    sug = sugerir_configuracion(pd.DataFrame({"fecha": pd.date_range("2024-01-01", periods=50), "ventas": range(50),
                                              "sucursal": ["a", "b"] * 25}))
    comprobar("sugiere fecha, valor y elemento", sug.get("columna_fecha") == "fecha" and sug.get("columna_valor") == "ventas"
              and sug.get("columna_entidad") == "sucursal", str(sug))


if __name__ == "__main__":
    t0 = time.time()
    prueba_series()
    prueba_eventos()
    prueba_errores()
    ok, total = sum(_resultados), len(_resultados)
    print(f"\n{ok}/{total} comprobaciones correctas ({time.time() - t0:.0f} s).")
    print("TODO OK" if ok == total else "HAY COMPROBACIONES CON ✗: revisa las líneas de arriba.")
