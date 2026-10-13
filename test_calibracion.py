import unittest
from tests.base import *
import prediccion as P
from prediccion import ConfigPrediccion as CP

def _sig(z): return 1 / (1 + np.exp(-z))

class EvaluarCalibracion(unittest.TestCase):
    def test_probabilidades_honestas_se_reconocen(self):  # F2.3
        rng = np.random.default_rng(0); n = 4000
        p = rng.uniform(0.02, 0.9, n); y = (rng.uniform(size=n) < p).astype(int)
        r = P.evaluar_calibracion(p, y)
        self.assertTrue(r["verificable"] and r["calibrada"]); self.assertIsNone(r["calibrador"]); self.assertIsNone(r["metodo"])
        self.assertLess(r["brier"], r["brier_base"])
    def test_probabilidades_distorsionadas_se_corrigen(self):
        rng = np.random.default_rng(1); n = 4000
        p_real = rng.uniform(0.02, 0.6, n); y = (rng.uniform(size=n) < p_real).astype(int)
        p_dist = _sig(np.log(p_real / (1 - p_real)) + 2.0)            # el modelo «exagera» (como al dar más peso a la clase rara)
        r = P.evaluar_calibracion(p_dist, y)
        self.assertGreater(r["ece"], 0.15)
        self.assertTrue(r["calibrada"] and r["calibrador"] is not None); self.assertLess(r["ece_calibrado"], 0.06)
        corregidas = r["calibrador"].transformar(p_dist)
        self.assertLess(abs(corregidas.mean() - y.mean()), 0.02)
    def test_pocos_casos_no_se_inventa_una_probabilidad(self):
        rng = np.random.default_rng(2); p = rng.uniform(size=120); y = (rng.uniform(size=120) < p).astype(int)
        r = P.evaluar_calibracion(p, y)
        self.assertFalse(r["verificable"]); self.assertIn("pocos casos", r["motivo"]); self.assertIsNone(r["calibrador"])
    def test_pocos_positivos_tampoco(self):
        rng = np.random.default_rng(3); y = np.zeros(1000, int); y[:8] = 1
        r = P.evaluar_calibracion(rng.uniform(0, 0.1, 1000), y)
        self.assertFalse(r["verificable"])
    def test_calibracion_es_determinista(self):
        rng = np.random.default_rng(4); p = rng.uniform(0.05, 0.7, 1500); y = (rng.uniform(size=1500) < p ** 2).astype(int)
        a, b = P.evaluar_calibracion(p, y, 5), P.evaluar_calibracion(p, y, 5)
        self.assertEqual((a["ece"], a["ece_calibrado"], a["metodo"]), (b["ece"], b["ece_calibrado"], b["metodo"]))
    def test_con_pocos_datos_usa_ajuste_suave_y_con_muchos_isotonica(self):
        rng = np.random.default_rng(5)
        for n, esperado in ((500, "platt"), (3000, "isotonica")):
            p_real = rng.uniform(0.05, 0.6, n); y = (rng.uniform(size=n) < p_real).astype(int)
            r = P.evaluar_calibracion(_sig(np.log(p_real / (1 - p_real)) + 2.0), y)
            self.assertEqual(r["metodo"], esperado, n)

def _datos_desbalanceados(n=3200, seed=0, tasa=0.06):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    z = 1.2 * x1 + 0.6 * x2 + np.log(tasa / (1 - tasa)) - 0.3
    y = (rng.uniform(size=n) < _sig(z)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

class SalidaDelModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _datos_desbalanceados()
        cfg = lambda **k: CP(objetivo="y", tipo="clasificacion", salida="probabilidad", pliegues=3, **k)
        cls.bal = P.entrenar(cls.df, cfg(balancear="si"), ["x1", "x2"])
        cls.nobal = P.entrenar(cls.df, cfg(balancear="no"), ["x1", "x2"])
        cls.poco = P.entrenar(cls.df.head(150), cfg(balancear="si"), ["x1", "x2"])
    def test_balanceado_con_datos_suficientes_se_calibra_y_queda_como_probabilidad(self):
        r = self.bal
        print(f"\n[calibración] balanceado: tipo={r.tipo_salida} metodo={r.calibracion.get('metodo')} ece={r.calibracion['ece']:.3f} -> {r.calibracion.get('ece_calibrado')}")
        self.assertTrue(r.balanceado); self.assertEqual(r.tipo_salida, "probabilidad"); self.assertIsNotNone(r.calibrador)
        pred = P.predecir(r, self.df.head(1500), salida="probabilidad")
        self.assertLess(abs(pred["y"].mean() - self.df.head(1500).y.mean()), 0.03)      # coincide con la tasa real
        self.assertEqual(pred.attrs["tipo_salida"], "probabilidad")
    def test_sin_calibrar_el_balanceo_infla_las_probabilidades(self):
        r = self.bal; sin_cal = P.Calibrador  # referencia
        r2 = P.entrenar(self.df, CP(objetivo="y", tipo="clasificacion", salida="probabilidad", pliegues=3, balancear="si"), ["x1", "x2"])
        bruto = r2.modelo.predict_proba(r2.preprocesador.transformar(self.df[["x1", "x2"]]))[:, 1]
        self.assertGreater(bruto.mean(), self.df.y.mean() * 2)                           # sin corregir: muy por encima de lo real
    def test_pocos_datos_se_entrega_como_ranking_con_motivo(self):
        r = self.poco
        self.assertEqual(r.tipo_salida, "ranking"); self.assertIsNone(r.calibrador)
        self.assertIn("ORDENAR", r.texto_salida()); self.assertIn("pocos casos", r.texto_salida())
    def test_sin_balancear_y_bien_calibrado_es_probabilidad(self):
        r = self.nobal
        self.assertEqual(r.tipo_salida, "probabilidad"); self.assertIn("Es una probabilidad", r.texto_salida())
    def test_regresion_es_valor(self):
        rng = np.random.default_rng(0); df = pd.DataFrame({"x": rng.normal(size=400)}); df["y"] = df.x * 3 + rng.normal(size=400)
        r = P.entrenar(df, CP(objetivo="y", tipo="regresion", pliegues=1), ["x"])
        self.assertEqual(r.tipo_salida, "valor"); self.assertEqual(P.predecir(r, df.head(5)).attrs["tipo_salida"], "valor")
    def test_el_texto_entra_en_el_resumen(self):
        self.assertTrue(any("Es una probabilidad" in l for l in self.nobal.lineas_resumen()))
        self.assertTrue(any("puntaje para ORDENAR" in l for l in self.poco.lineas_resumen()))
if __name__ == "__main__":
    unittest.main()


class PalabrasClave(unittest.TestCase):
    """El tipo de salida llega a cada resultado y Prioridad no confunde un puntaje con una probabilidad."""
    def test_todas_las_palabras_declaran_tipo_de_salida(self):
        from tests.test_contrato import _casos_buenos
        esperado = {"proyeccion": "valor", "tendencia": "valor", "estacionalidad": "valor", "escenario": "valor",
                    "agotamiento": "valor", "umbral": "valor", "saturacion": "valor", "vida_util": "valor",
                    "duracion": "valor", "prioridad": "ranking", "falla": "probabilidad"}
        for k, (cfg, tablas) in _casos_buenos().items():
            with self.subTest(palabra=k):
                r = pr.ejecutar(cfg, tablas)
                self.assertEqual(r.tipo_salida, esperado[k])
                from pronostico_base import TEXTO_TIPO_SALIDA
                self.assertIn(TEXTO_TIPO_SALIDA[esperado[k]], r.detalle)
    def test_clasificacion_declara_tipo_segun_calibracion(self):
        df = _datos_desbalanceados()
        r = pr.ejecutar(C(palabra="clasificacion", tabla="t", columna_objetivo="y"), {"t": df})
        self.assertIn(r.tipo_salida, ("probabilidad", "ranking"))
        self.assertTrue(any(l.startswith(("Es una probabilidad", "El número es un puntaje")) for l in r.detalle), r.detalle)
    def _prioridad(self, col_riesgo, valores):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"e": [f"E{i}" for i in range(20)], col_riesgo: valores, "costo": rng.lognormal(7, 1, 20)})
        return pr.ejecutar(C(palabra="prioridad", tabla="p", columna_riesgo=col_riesgo, columna_impacto="costo", columna_entidad="e"), {"p": df})
    def test_prioridad_con_probabilidad_habla_de_perdida_esperada(self):
        r = self._prioridad("prob_falla", np.random.default_rng(1).uniform(0, 0.5, 20))
        self.assertIn("Pérdida esperada", list(r.tabla.columns)); self.assertFalse(any("puntaje u orden" in a for a in r.advertencias))
    def test_prioridad_con_puntaje_no_habla_de_perdida_esperada(self):
        for nombre, vals in (("riesgo_score", np.random.default_rng(1).uniform(0, 0.5, 20)),
                             ("riesgo", np.random.default_rng(1).uniform(0, 80, 20))):
            with self.subTest(columna=nombre):
                r = self._prioridad(nombre, vals)
                self.assertNotIn("Pérdida esperada", list(r.tabla.columns))
                self.assertTrue(any("puntaje u orden" in a for a in r.advertencias), r.advertencias)
    def test_falla_con_modelo_etiqueta_segun_calibracion(self):
        rng = np.random.default_rng(0); filas = []; ini = pd.Timestamp("2024-01-01")
        for m in range(60):
            for d in range(0, 300, 7):
                x = rng.normal(); falla = rng.uniform() < (0.03 if x < 1 else 0.3)
                filas.append({"maq": f"M{m}", "fecha": ini + pd.Timedelta(days=d), "temp": x, "tipo": "falla" if falla else "lectura"})
        r = pr.ejecutar(C(palabra="falla", tabla="t", columna_fecha="fecha", columna_entidad="maq", columna_evento="tipo",
                          valor_evento="falla", ventana_dias=14, columnas_explicativas=["temp"], metodo_evento="modelo"),
                        {"t": pd.DataFrame(filas)})
        self.assertIn(r.tipo_salida, ("probabilidad", "ranking"))
        self.assertEqual(r.tabla.columns[1].startswith("Probabilidad"), r.tipo_salida == "probabilidad")
        self.assertTrue(any(l.startswith(("Es una probabilidad", "El número es un puntaje")) for l in r.detalle))
if __name__ == "__main__":
    unittest.main()