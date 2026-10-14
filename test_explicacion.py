import unittest
from tests.base import *
import prediccion as P
from prediccion import ConfigPrediccion as CP

def _datos(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(2.0 * x1 + 0.0 * x2 - 1.0)))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

class ExplicacionPorFila(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _datos()
        cls.r = P.entrenar(cls.df, CP(objetivo="y", tipo="clasificacion", salida="probabilidad", pliegues=1), ["x1", "x2", "x3"])
    def test_la_columna_que_manda_es_la_que_empuja(self):  # F2.5: explicación por elemento
        df = self.df.head(300).reset_index(drop=True)
        altos = list(df.index[df.x1 > 1.5][:15])
        self.assertTrue(len(altos) >= 5)
        for a in P.explicar_filas(self.r, df, altos):
            self.assertTrue(a and a[0][0] == "x1", a)
    def test_una_fila_tipica_casi_no_tiene_empuje(self):
        df = self.df.head(300).reset_index(drop=True)
        medio = int(np.argmin(np.abs(df.x1.to_numpy()) + np.abs(df.x2.to_numpy()) + np.abs(df.x3.to_numpy())))
        a = P.explicar_filas(self.r, df, [medio])[0]
        self.assertTrue(all(d < 0.08 for _, d in a), a)
    def test_es_determinista_y_no_modifica_los_datos(self):
        df = self.df.head(100).reset_index(drop=True); copia = df.copy()
        a, b = P.explicar_filas(self.r, df, [0, 1, 2]), P.explicar_filas(self.r, df, [0, 1, 2])
        self.assertEqual(a, b); pd.testing.assert_frame_equal(df, copia)
    def test_agrupa_las_columnas_de_fecha(self):
        d = self.df.copy(); d["fecha"] = pd.date_range("2022-01-01", periods=len(d), freq="D")
        r = P.entrenar(d, CP(objetivo="y", tipo="clasificacion", salida="probabilidad", pliegues=1), ["x1", "fecha"])
        a = P.explicar_filas(r, d.head(50).reset_index(drop=True), [0, 1, 2])
        nombres = {c for fila in a for c, _ in fila}
        self.assertTrue(nombres <= {"x1", "fecha"}, nombres)                    # nunca «fecha (mes)», «fecha (hora)»…
    def test_regresion_tambien(self):
        rng = np.random.default_rng(0); n = 600
        df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}); df["y"] = 5 * df.a + rng.normal(size=n)
        r = P.entrenar(df, CP(objetivo="y", tipo="regresion", pliegues=1), ["a", "b"])
        altos = list(df.head(200).reset_index(drop=True).index[df.head(200).a > 1.5][:8])
        for fila in P.explicar_filas(r, df.head(200).reset_index(drop=True), altos):
            self.assertTrue(fila and fila[0][0] == "a")
    def test_falla_con_modelo_muestra_que_lo_empuja(self):
        rng = np.random.default_rng(0); filas = []; ini = pd.Timestamp("2024-01-01")
        for m in range(80):
            for d in range(0, 300, 7):
                x = rng.normal(); falla = rng.uniform() < (0.03 if x < 1 else 0.35)
                filas.append({"maq": f"M{m}", "fecha": ini + pd.Timedelta(days=d), "temp": x, "ruido": rng.normal(), "tipo": "falla" if falla else "lectura"})
        r = pr.ejecutar(C(palabra="falla", tabla="t", columna_fecha="fecha", columna_entidad="maq", columna_evento="tipo", valor_evento="falla",
                          ventana_dias=14, columnas_explicativas=["temp", "ruido"], metodo_evento="modelo"), {"t": pd.DataFrame(filas)})
        self.assertIn("Qué lo empuja", r.tabla.columns)
        llenas = r.tabla["Qué lo empuja"].head(10)
        self.assertTrue((llenas != "").all()); self.assertTrue((r.tabla["Qué lo empuja"].iloc[10:] == "").all())
        self.assertTrue(any("temp" in t for t in llenas), list(llenas))
        self.assertTrue(any("explicación aproximada" in l for l in r.detalle))
if __name__ == "__main__":
    unittest.main()
