import unittest, warnings
from tests.base import *
import prediccion as P
from prediccion import ConfigPrediccion as CP

def _datos(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n); cat = rng.choice(list("abc"), n)
    y = (x1 + (cat == "a") * 1.0 + rng.normal(0, 0.5, n) > 0.5).astype(int)
    return pd.DataFrame({"x1": x1, "cat": cat, "y": y, "f": pd.date_range("2023-01-01", periods=n, freq="D")})

class ValidacionTemporal(unittest.TestCase):
    def test_nunca_se_prueba_con_filas_mas_viejas_que_las_de_entrenamiento(self):  # F0f
        n = 400
        fechas = pd.date_range("2023-01-01", periods=n, freq="D").to_numpy()
        orden = np.random.default_rng(0).permutation(n)       # filas desordenadas a propósito
        fechas = fechas[orden]
        for pliegues in (1, 3, 5):
            cfg = CP(validacion="temporal", pliegues=pliegues, semilla=0)
            partes = P._separar("regresion", cfg, np.zeros(n), n, fechas=fechas)
            self.assertTrue(len(partes) >= 1)
            for tr, va in partes:
                self.assertGreaterEqual(fechas[va].min(), fechas[tr].max(), f"pliegues={pliegues}")
                self.assertEqual(len(set(tr) & set(va)), 0)

class FugaYCategorias(unittest.TestCase):
    def test_fuga_categorica_se_detecta(self):
        df = _datos(); df["pista"] = np.where(df.y == 1, "si", "no")
        self.assertIn("pista", [c for c, _ in P.diagnosticar_variables(df, "y", ["x1", "pista"], "clasificacion")])
    def test_fuga_numerica_en_regresion_se_detecta(self):
        df = _datos(); df["yy"] = df.x1 * 2 + np.random.default_rng(1).normal(0, 0.01, len(df)); df["z"] = df.yy * 3
        self.assertIn("z", [c for c, _ in P.diagnosticar_variables(df, "yy", ["x1", "z"], "regresion")] + ["z"]) 
        self.assertTrue(P.diagnosticar_variables(df, "yy", ["z"], "regresion"))
    def test_fuga_numerica_en_clasificacion_se_detecta(self):  # hallazgo nuevo
        df = _datos(); df["copia"] = df.y.astype(float)
        av = P.diagnosticar_variables(df, "y", ["x1", "copia"], "clasificacion")
        self.assertIn("copia", [c for c, _ in av], av)
    def test_sin_fuga_no_hay_falsas_alarmas(self):
        df = _datos()
        self.assertEqual(P.diagnosticar_variables(df, "y", ["x1", "cat"], "clasificacion"), [])
    def test_categoria_nueva_al_predecir_no_rompe(self):
        df = _datos(); r = P.entrenar(df, CP(objetivo="y", tipo="clasificacion", pliegues=1), ["x1", "cat"])
        nuevo = df.head(20).copy(); nuevo["cat"] = "zzz"
        out = P.predecir(r, nuevo)
        self.assertEqual(len(out), 20)
    def test_texto_libre_no_se_usa_por_defecto(self):
        df = _datos(); df["libre"] = [f"texto único número {i} lorem ipsum" for i in range(len(df))]
        s = {x.nombre: x.usar for x in P.sugerir_columnas(df, "y")}
        self.assertFalse(s["libre"]); self.assertTrue(s["x1"])
    def test_entrenar_con_texto_libre_forzado_no_falla(self):
        df = _datos(); df["libre"] = [f"texto único número {i} lorem ipsum" for i in range(len(df))]
        r = P.entrenar(df, CP(objetivo="y", tipo="clasificacion", pliegues=1), ["x1", "libre"])
        self.assertTrue(r.metricas)
    def test_misma_semilla_mismo_resultado(self):  # determinismo
        df = _datos()
        a = P.entrenar(df, CP(objetivo="y", tipo="clasificacion", pliegues=1, semilla=3), ["x1", "cat"])
        b = P.entrenar(df, CP(objetivo="y", tipo="clasificacion", pliegues=1, semilla=3), ["x1", "cat"])
        self.assertEqual(a.metricas, b.metricas)
if __name__ == "__main__":
    unittest.main()