import unittest
from tests.base import *
import pronostico_base as pb

class FechaCorte(unittest.TestCase):
    def _df(self, n=400, ini="2024-01-01"):
        f = pd.date_range(ini, periods=n, freq="D")
        return pd.DataFrame({"f": f, "v": 100 + 0.1*np.arange(n) + np.random.default_rng(0).normal(0, 2, n)})
    def _cfg(self, **k):
        return C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="v", horizonte=10, **k)
    def test_formulario_ofrece_fecha_corte_en_las_palabras_de_series_y_eventos(self):  # F1.5
        for k, spec in pr.PALABRAS_CLAVE.items():
            if spec["mecanismo"] in ("serie", "evento"):
                self.assertIn("fecha_corte", [c["campo"] for c in spec["campos"]], k)
    def test_las_series_ignoran_lo_posterior_al_corte(self):
        df = self._df()
        con_corte = pr.ejecutar(self._cfg(fecha_corte="2024-12-31"), {"t": df})
        recortada = pr.ejecutar(self._cfg(), {"t": df[df.f <= "2024-12-31"]})
        self.assertEqual(con_corte.titular, recortada.titular)
        self.assertEqual(con_corte.tabla.iloc[0, 0], recortada.tabla.iloc[0, 0])
        self.assertEqual(con_corte.tabla.iloc[0, 0], pd.Timestamp("2025-01-01"))
    def test_agotamiento_respeta_el_corte(self):
        f = pd.date_range("2024-01-01", periods=200, freq="D")
        df = pd.DataFrame({"f": f, "c": np.r_[np.full(120, 5.0), np.full(80, 50.0)]})
        cfg = lambda **k: C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", stock_actual=100, **k)
        a = pr.ejecutar(cfg(fecha_corte="2024-04-29"), {"t": df}).titular     # solo consumo de 5/día
        b = pr.ejecutar(cfg(), {"t": df}).titular                              # consumo reciente de 50/día
        self.assertIn("20 días", a); self.assertNotEqual(a, b)
    def test_fecha_invalida_da_error_claro(self):
        with self.assertRaises(ValueError) as cm:
            pr.ejecutar(self._cfg(fecha_corte="ayer por la tarde"), {"t": self._df()})
        self.assertIn("fecha de corte", str(cm.exception))
    def test_corte_anterior_a_todos_los_datos(self):
        with self.assertRaises(ValueError) as cm:
            pr.ejecutar(self._cfg(fecha_corte="2020-01-01"), {"t": self._df()})
        self.assertIn("fecha de corte", str(cm.exception))
    def test_avisa_si_el_ultimo_dato_esta_lejos_de_hoy(self):
        df = self._df(ini="2020-01-01")
        r = pr.ejecutar(self._cfg(), {"t": df})
        self.assertTrue(any("se cuentan desde ahí" in a for a in r.advertencias), r.advertencias)
        self.assertIn("se cuentan desde ahí", r.advertencias[0])
    def test_no_avisa_si_los_datos_son_recientes(self):
        hoy = pd.Timestamp.now().normalize()
        df = self._df(n=200, ini=(hoy - pd.Timedelta(days=199)).strftime("%Y-%m-%d"))
        r = pr.ejecutar(self._cfg(), {"t": df})
        self.assertFalse(any("se cuentan desde ahí" in a for a in r.advertencias))
    def test_aviso_funciona_con_hoy_inyectado(self):
        df = self._df(n=50, ini="2024-01-01")
        self.assertIsNone(pb.aviso_datos_antiguos(df, ["f"], hoy="2024-03-01"))
        self.assertIn("hace", pb.aviso_datos_antiguos(df, ["f"], hoy="2024-12-01"))
if __name__ == "__main__":
    unittest.main()