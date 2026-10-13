import unittest, json
from tests.base import *
import prediccion as P
from prediccion import ConfigPrediccion as CP
import formato as F
from tests.test_calibracion import _datos_desbalanceados

def _serie_df(n=240):
    rng = np.random.default_rng(0); d = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"f": d, "v": 100 + 10 * np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, 2, n), "t": 60 + 0.05 * np.arange(n) + rng.normal(0, 1, n)})

class ModoExperto(unittest.TestCase):
    def test_proyeccion_lleva_sus_detalles_tecnicos(self):  # F3.2
        r = pr.ejecutar(C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="v", horizonte=14), {"t": _serie_df()})
        x = r.experto
        self.assertIsNotNone(x)
        etiquetas = [e for e, _ in x["filas"]]
        for esperada in ("Método elegido", "Pruebas pasadas (orígenes)", "Horizonte comprobado", "Evaluación honesta",
                         "Error típico del método (RMSE)", "Semilla"):
            self.assertIn(esperada, etiquetas)
        errores = [v for _, v in x["metodos"]]
        self.assertGreaterEqual(len(errores), 3); self.assertEqual(errores, sorted(errores))          # el mejor primero
        self.assertTrue(0 < len(x["residuos"]) <= 400 and all(np.isfinite(x["residuos"])))
        self.assertIn(r.metricas["modelo"], [n for n, _ in x["metodos"]])
        json.dumps(x)                                                                              # solo datos simples
    def test_umbral_y_saturacion_tambien(self):
        df = _serie_df()
        u = pr.ejecutar(C(palabra="umbral", tabla="t", columna_fecha="f", columna_valor="t", umbral=80, horizonte=60), {"t": df})
        s = pr.ejecutar(C(palabra="saturacion", tabla="t", columna_fecha="f", columna_valor="v", capacidad=200, horizonte=30), {"t": df})
        self.assertTrue(u.experto and s.experto)
    def test_los_detalles_son_deterministas(self):
        a = pr.ejecutar(C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="v"), {"t": _serie_df()}).experto
        b = pr.ejecutar(C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="v"), {"t": _serie_df()}).experto
        self.assertEqual(a, b)
    def test_clasificacion_lleva_detalles_del_modelo(self):
        r = pr.ejecutar(C(palabra="clasificacion", tabla="t", columna_objetivo="y"), {"t": _datos_desbalanceados(1500)})
        x = r.experto; d = dict(x["filas"])
        self.assertIn("Semilla", d); self.assertNotEqual(d["Semilla"], "—")
        self.assertIn("Cómo se probó", d); self.assertIn("Mejora sobre la respuesta fácil", d)
        self.assertTrue(len(x["metodos"]) >= 2 and all(v >= 0 for _, v in x["metodos"]))
        self.assertTrue(any(e.startswith("Calibración") or e.startswith("Brier") for e, _ in x["filas"]))
    def test_regresion_incluye_residuos(self):
        rng = np.random.default_rng(0); df = pd.DataFrame({"x": rng.normal(size=500)}); df["y"] = df.x * 3 + rng.normal(size=500)
        x = P.entrenar(df, CP(objetivo="y", tipo="regresion", pliegues=1), ["x"]).detalle_experto()
        self.assertTrue(x["residuos"] and abs(np.mean(x["residuos"])) < 0.5); self.assertTrue(x["residuos_titulo"])
    def test_respeta_el_formato_de_numeros(self):
        F.set_formato_numerico("chileno")
        try:
            x = pr.ejecutar(C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="v"), {"t": _serie_df()}).experto
            err = dict(x["filas"])["Error típico del método (RMSE)"]
            self.assertIn(",", err); self.assertNotIn(".", err)
        finally:
            F.set_formato_numerico("estandar")
    def test_el_resto_de_palabras_no_inventa_detalles(self):
        d = pd.date_range("2024-01-01", periods=120, freq="D")
        r = pr.ejecutar(C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", stock_actual=500), {"t": pd.DataFrame({"f": d, "c": np.full(120, 5.0)})})
        self.assertIsNone(r.experto)
if __name__ == "__main__":
    unittest.main()
