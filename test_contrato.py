import unittest, warnings
from tests.base import *
from pronostico_base import ResultadoPronostico, num_txt, dias_txt, pct_txt, duracion_txt, fecha_txt

def _casos_buenos():
    rng = np.random.default_rng(0)
    d = pd.date_range("2024-01-01", periods=240, freq="D")
    dow = np.array([1, 1, 1.05, 1.1, 1.4, 1.7, 1.5])[d.dayofweek]
    serie_df = pd.DataFrame({"f": d, "v": (100 + 0.1*np.arange(240)) * dow * rng.normal(1, 0.05, 240),
                             "x": 10 + rng.normal(0, 1, 240), "e": ["a", "b"] * 120, "stock": 5000.0,
                             "temp": 60 + 0.05*np.arange(240) + rng.normal(0, 1, 240)})
    ini = pd.Timestamp("2024-01-01"); filas = []
    for m in range(40):
        t = 0.0
        while True:
            t += 100.0 * rng.weibull(2.0)
            if t > 300: break
            filas.append({"maq": f"M{m}", "fecha": ini + pd.Timedelta(days=float(t)), "tipo": "falla"})
        filas.append({"maq": f"M{m}", "fecha": ini + pd.Timedelta(days=float(rng.uniform(0, 300))), "tipo": "otro"})
    ev_df = pd.DataFrame(filas); ev_df["inst"] = ini
    n = 300; inicio = ini + pd.to_timedelta(rng.integers(0, 150, n), unit="D")
    du = pd.DataFrame({"a": inicio, "b": inicio + pd.to_timedelta(np.exp(rng.normal(2, .5, n)), unit="D")})
    cl = pd.DataFrame({"x1": rng.normal(size=300), "x2": rng.normal(size=300)}); cl["y"] = (cl.x1 + rng.normal(0, .5, 300) > 0).astype(int)
    pri = pd.DataFrame({"e": [f"E{i}" for i in range(30)], "riesgo": rng.beta(1, 6, 30), "costo": rng.lognormal(7, 1, 30)})
    S = dict(tabla="s", columna_fecha="f", columna_valor="v")
    return {
        "proyeccion": (C(palabra="proyeccion", horizonte=14, **S), {"s": serie_df}),
        "tendencia": (C(palabra="tendencia", **S), {"s": serie_df}),
        "estacionalidad": (C(palabra="estacionalidad", **S), {"s": serie_df}),
        "escenario": (C(palabra="escenario", columna_driver="x", cambio_pct=10, **S), {"s": serie_df}),
        "agotamiento": (C(palabra="agotamiento", columna_stock="stock", **S), {"s": serie_df}),
        "umbral": (C(palabra="umbral", tabla="s", columna_fecha="f", columna_valor="temp", umbral=80, horizonte=90), {"s": serie_df}),
        "saturacion": (C(palabra="saturacion", capacidad=300, horizonte=30, **S), {"s": serie_df}),
        "vida_util": (C(palabra="vida_util", tabla="e", columna_fecha="fecha", columna_entidad="maq", columna_evento="tipo",
                        valor_evento="falla", columna_inicio="inst"), {"e": ev_df}),
        "falla": (C(palabra="falla", tabla="e", columna_fecha="fecha", columna_entidad="maq", columna_evento="tipo",
                    valor_evento="falla", columna_inicio="inst", ventana_dias=30), {"e": ev_df}),
        "duracion": (C(palabra="duracion", tabla="d", columna_inicio="a", columna_fin="b"), {"d": du}),
        "prioridad": (C(palabra="prioridad", tabla="p", columna_riesgo="riesgo", columna_impacto="costo", columna_entidad="e"), {"p": pri}),
    }

class Contrato(unittest.TestCase):
    """F0h: toda palabra clave devuelve ResultadoPronostico con titular, confianza válida y supuestos."""
    def test_todas_cumplen_el_contrato(self):
        casos = _casos_buenos()
        self.assertEqual(set(casos) | {"clasificacion"}, set(pr.ORDEN_PALABRAS))
        for k, (cfg, tablas) in casos.items():
            with self.subTest(palabra=k):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    r = pr.ejecutar(cfg, tablas)
                self.assertIsInstance(r, ResultadoPronostico)
                self.assertTrue(r.titular.strip())
                self.assertIn(r.confianza, ("alta", "media", "baja"))
                self.assertIn(r.nivel, ("info", "atencion", "alerta"))
                self.assertTrue(r.supuestos, "sin supuestos")
                self.assertTrue(r.por_que_confianza)
                self.assertEqual(r.palabra, k)
                self.assertNotIn("nan", r.titular.lower().replace("financ", ""))
    def test_config_round_trip(self):
        for k in pr.ORDEN_PALABRAS:
            cfg = C(palabra=k, tabla="t", umbral=5.0, columnas_explicativas=["a"], fecha_corte="2024-05-31", fin_observacion="global")
            self.assertEqual(C.from_dict(cfg.to_dict()), cfg)
    def test_config_antigua_sin_campos_nuevos_sigue_abriendo(self):
        d = C(palabra="proyeccion", tabla="t").to_dict()
        d.pop("fin_observacion", None); d.pop("fecha_corte", None)       # un .hadarproy guardado antes de estos cambios
        cfg = C.from_dict(d)
        self.assertEqual(cfg.fin_observacion, "auto"); self.assertIsNone(cfg.fecha_corte)
    def test_config_prediccion_round_trip(self):
        from prediccion import ConfigPrediccion as CP
        c = CP(objetivo="y", tipo="clasificacion", validacion="temporal", columna_fecha="f", semilla=7)
        self.assertEqual(CP.from_dict(c.to_dict()), c)

class CasosLimite(unittest.TestCase):
    """F0g: serie constante, muy corta, ceros, columna sin números → error controlado en español o resultado, sin
    excepción cruda ni RuntimeWarning."""
    def test_las_12_con_datos_raros(self):
        d = pd.date_range("2024-01-01", periods=60, freq="D")
        casos = {"constante": pd.DataFrame({"f": d, "v": np.full(60, 5.0), "e": ["a", "b"] * 30}),
                 "ceros": pd.DataFrame({"f": d, "v": np.zeros(60), "e": ["a", "b"] * 30}),
                 "corta": pd.DataFrame({"f": d[:12], "v": np.arange(12.0), "e": ["a", "b"] * 6}),
                 "sin_numeros": pd.DataFrame({"f": d, "v": ["x"] * 60, "e": ["a", "b"] * 30})}
        extra = dict(umbral=7.0, capacidad=7.0, stock_actual=100.0, columna_driver="v", columna_inicio="f",
                     columna_riesgo="v", columna_objetivo="e", ventana_dias=10.0)
        for nombre, df in casos.items():
            for k in pr.ORDEN_PALABRAS:
                cfg = C(palabra=k, tabla="t", columna_fecha="f", columna_valor="v", columna_entidad="e", **extra)
                with self.subTest(datos=nombre, palabra=k):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        try:
                            r = pr.ejecutar(cfg, {"t": df}); self.assertTrue(r.titular)
                        except ValueError as e:
                            self.assertGreater(len(str(e)), 10)

class FormatoNumeros(unittest.TestCase):
    """F0e: num_txt, dias_txt, pct_txt, duracion_txt con casos borde."""
    def test_num_txt(self):
        self.assertEqual(num_txt(None), "—"); self.assertEqual(num_txt(float("nan")), "—"); self.assertEqual(num_txt("x"), "—")
        self.assertEqual(num_txt(float("inf")), "—"); self.assertEqual(num_txt(0), "0")
        self.assertEqual(num_txt(1234.6), "1,235")      # formato estándar (el chileno se prueba en test_ajustes)
        self.assertEqual(num_txt(3.14159), "3.14"); self.assertEqual(num_txt(150.25), "150.2")
        self.assertEqual(num_txt(12.0), "12"); self.assertEqual(num_txt(0.00123), "0.00123"); self.assertEqual(num_txt(-1500), "-1,500")
    def test_dias_txt(self):
        self.assertEqual(dias_txt(0.5), "12 horas"); self.assertEqual(dias_txt(45), "45 días"); self.assertEqual(dias_txt(1), "1 día")
        self.assertEqual(dias_txt(400), "13 meses"); self.assertEqual(dias_txt(1 / 1440), "1 minuto"); self.assertEqual(dias_txt(None), "—")
        self.assertEqual(dias_txt(float("inf")), "—")
    def test_pct_txt(self):
        self.assertEqual(pct_txt(0.153), "15%"); self.assertEqual(pct_txt(0.153, signo=True), "+15%")
        self.assertEqual(pct_txt(-0.0001, 0), "0%"); self.assertEqual(pct_txt(float("nan")), "—"); self.assertEqual(pct_txt(None), "—")
    def test_duracion_txt(self):
        self.assertEqual(duracion_txt(1), "1 día"); self.assertEqual(duracion_txt(3, "W"), "3 semanas")
        self.assertEqual(duracion_txt(2.5), "2.5 días"); self.assertEqual(duracion_txt(float("inf")), "—")
    def test_fecha_txt(self):
        self.assertEqual(fecha_txt(pd.Timestamp("2024-05-31")), "31 may 2024"); self.assertEqual(fecha_txt(None), "—"); self.assertEqual(fecha_txt(pd.NaT), "—")

@LENTO
class KaplanMeier(unittest.TestCase):
    """F0b: tiempos exponenciales (media 100, mediana real 69.3) con casos abiertos censurados en la fecha de corte."""
    def _mediana(self, seed, n=500):
        rng = np.random.default_rng(seed)
        ini = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
        fin = ini + pd.to_timedelta(rng.exponential(100.0, n), unit="D")
        corte = pd.Timestamp("2024-10-27")
        df = pd.DataFrame({"a": ini, "b": fin.where(fin <= corte)})
        abiertos = float(df.b.isna().mean())
        r = ev.analizar_duracion(df, C(palabra="duracion", tabla="t", columna_inicio="a", columna_fin="b",
                                       fecha_corte="2024-10-27"))
        return r.metricas["mediana_dias"], abiertos, df
    def test_mediana_con_censura_se_acerca_a_la_real(self):
        errores = []
        for s in range(5):
            med, abiertos, _ = self._mediana(s)
            self.assertGreater(abiertos, 0.2)                      # hay censura de verdad
            errores.append(abs(med / 69.3 - 1))
        self.assertLess(float(np.mean(errores)), 0.15, errores)
    def test_ignorar_la_censura_sesga_hacia_abajo_y_km_no(self):
        med, _, df = self._mediana(0)
        ingenua = float(np.median((df.b - df.a).dt.days.dropna()))   # solo los que terminaron: subestima
        self.assertLess(ingenua, 69.3 * 0.9)
        self.assertLess(abs(med - 69.3), abs(ingenua - 69.3))
if __name__ == "__main__":
    unittest.main()