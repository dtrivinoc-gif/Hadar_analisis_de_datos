import unittest
from unittest import mock
from tests.base import *
import prediccion as P
from prediccion import ConfigPrediccion as CP

def _datos(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n); x2 = rng.normal(size=n); cat = rng.choice(list("abc"), n)
    y = (x1 + 0.3 * (cat == "a") + rng.normal(0, 0.5, n) > 0.3).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})

CFG = lambda **k: CP(objetivo="y", tipo="clasificacion", pliegues=1, **k)

class Duplicados(unittest.TestCase):
    def test_filas_repetidas_se_avisan_y_no_se_borran(self):  # F2.2
        base = _datos(400); df = pd.concat([base, base.sample(200, random_state=1)], ignore_index=True)
        r = P.entrenar(df, CFG(), ["x1", "x2", "cat"])
        self.assertTrue(any("son idénticas a otra" in a for a in r.advertencias), r.advertencias)
        self.assertEqual(len(df), 600)                                  # no se borró nada de la tabla
    def test_sin_repetidas_no_avisa(self):
        r = P.entrenar(_datos(), CFG(), ["x1", "x2", "cat"])
        self.assertFalse(any("son idénticas a otra" in a for a in r.advertencias))
    def test_pocos_valores_distintos_no_es_alarma(self):
        rng = np.random.default_rng(0); n = 600
        df = pd.DataFrame({"a": rng.choice(list("xyz"), n), "b": rng.choice(list("pq"), n)}); df["y"] = (df.a == "x").astype(int)
        r = P.entrenar(df, CFG(), ["a", "b"])
        self.assertFalse(any("son idénticas a otra" in a for a in r.advertencias))

class CategoriasDesconocidas(unittest.TestCase):
    def test_predecir_cuenta_y_avisa(self):  # F2.4
        df = _datos(); r = P.entrenar(df, CFG(), ["x1", "x2", "cat"])
        nuevo = df.head(50).copy(); nuevo.loc[nuevo.index[:10], "cat"] = "zzz"
        out = P.predecir(r, nuevo)
        av = out.attrs.get("avisos", [])
        self.assertEqual(len(av), 1); self.assertIn("10 filas", av[0]); self.assertIn("«zzz»", av[0]); self.assertIn("«cat»", av[0])
    def test_sin_categorias_nuevas_no_hay_aviso(self):
        df = _datos(); r = P.entrenar(df, CFG(), ["x1", "x2", "cat"])
        self.assertEqual(P.predecir(r, df.head(30)).attrs.get("avisos"), [])
    def test_valores_vacios_no_cuentan_como_desconocidos(self):
        df = _datos(); r = P.entrenar(df, CFG(), ["x1", "x2", "cat"])
        nuevo = df.head(20).copy(); nuevo.loc[nuevo.index[:5], "cat"] = None
        self.assertEqual(P.predecir(r, nuevo).attrs.get("avisos"), [])
    def test_texto_de_alta_cardinalidad_tambien_se_cuenta(self):
        rng = np.random.default_rng(0); n = 800
        df = pd.DataFrame({"x": rng.normal(size=n), "id_texto": [f"cliente_{i % 300}" for i in range(n)]})
        df["y"] = (df.x > 0).astype(int)
        r = P.entrenar(df, CFG(), ["x", "id_texto"])
        nuevo = df.head(20).copy(); nuevo["id_texto"] = "cliente_nuevo"
        av = P.predecir(r, nuevo).attrs.get("avisos", [])
        self.assertTrue(av and "20 filas" in av[0] and "«id_texto»" in av[0], av)

class Importancia(unittest.TestCase):
    def test_la_columna_que_manda_sale_primera(self):
        r = P.entrenar(_datos(), CFG(), ["x1", "x2", "cat"])
        self.assertTrue(r.importancias); self.assertEqual(r.importancias[0][0], "x1")
    def test_si_falla_se_avisa_en_vez_de_callar(self):  # F2.5
        with mock.patch("sklearn.inspection.permutation_importance", side_effect=RuntimeError("boom")):
            r = P.entrenar(_datos(), CFG(), ["x1", "x2", "cat"])
        self.assertEqual(r.importancias, [])
        self.assertTrue(any("No se pudo calcular qué columnas pesan" in a and "boom" in a for a in r.advertencias), r.advertencias)
    def test_frase_de_limitaciones(self):
        r = P.entrenar(_datos(), CFG(), ["x1", "x2", "cat"])
        self.assertTrue(any("no explican un caso puntual" in l for l in r.lineas_resumen()))
if __name__ == "__main__":
    unittest.main()
