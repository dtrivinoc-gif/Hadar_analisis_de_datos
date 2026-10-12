import unittest
from tests.base import *
import prediccion as P
from prediccion import ConfigPrediccion as CP
import ajustes as A

def _datos_con_deriva(n=500, seed=0, con_fecha=True):
    """La relación cambia con el tiempo: primera mitad y > 0 si x > 0; segunda mitad al revés."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n); t = np.arange(n)
    signo = np.where(t < n // 2, 1.0, -1.0)
    y = ((signo * x + rng.normal(0, 0.4, n)) > 0).astype(int)
    df = pd.DataFrame({"x": x, "x2": rng.normal(size=n), "y": y})
    if con_fecha:
        df["fecha"] = pd.date_range("2022-01-01", periods=n, freq="D")
    return df

class ValidacionPorFecha(unittest.TestCase):
    @classmethod
    def setUpClass(cls):          # cada análisis se calcula una sola vez (los tests son más rápidos)
        cls.df = _datos_con_deriva()
        cfg = lambda **k: C(palabra="clasificacion", tabla="t", columna_objetivo="y", **k)
        cls.auto = pr.ejecutar(cfg(), {"t": cls.df})
        cls.azar = pr.ejecutar(cfg(validacion="aleatoria"), {"t": cls.df})
        cls.sin_fecha = pr.ejecutar(cfg(), {"t": _datos_con_deriva(con_fecha=False)})
        dos = cls.df.copy(); dos["otra_fecha"] = dos.fecha + pd.Timedelta(days=3)
        cls.varias = pr.ejecutar(cfg(), {"t": dos})
        cls.manual = pr.ejecutar(cfg(columna_fecha="fecha"), {"t": dos})
    def test_clasificacion_elige_fecha_sola_si_hay_una(self):  # F2.1
        self.assertTrue(any("casos más nuevos" in s for s in self.auto.supuestos), self.auto.supuestos)
        self.assertFalse(any("mezcló las filas al azar" in a for a in self.auto.advertencias))
    def test_la_prueba_aleatoria_es_mas_optimista_que_la_temporal(self):
        azar, fecha = self.azar.metricas["metricas"], self.auto.metricas["metricas"]
        clave = "balanced_accuracy" if "balanced_accuracy" in azar else sorted(azar)[0]
        print(f"\n[validación] {clave}: al azar {azar[clave]:.2f} | por fecha {fecha[clave]:.2f}")
        self.assertGreater(azar[clave], fecha[clave] + 0.05)
    def test_aleatoria_con_fecha_avisa_claramente(self):
        self.assertTrue(any("mezcló las filas al azar" in a for a in self.azar.advertencias), self.azar.advertencias)
    def test_sin_fecha_no_avisa_ni_inventa(self):
        self.assertFalse(any("mezcló las filas al azar" in a for a in self.sin_fecha.advertencias))
        self.assertFalse(any("casos más nuevos" in s for s in self.sin_fecha.supuestos))
    def test_varias_fechas_pide_elegir(self):
        self.assertTrue(any("elige una en «Fecha»" in a for a in self.varias.advertencias), self.varias.advertencias)
    def test_fecha_elegida_a_mano_se_respeta(self):
        self.assertTrue(any("«fecha»" in s for s in self.manual.supuestos), self.manual.supuestos)
    def test_el_ajuste_aleatoria_se_respeta(self):
        cfg = A.aplicar_ajustes(C(palabra="clasificacion", tabla="t", columna_objetivo="y"), {"validacion_con_fecha": "aleatoria"})
        self.assertEqual(cfg.validacion, "aleatoria")
        cfg = A.aplicar_ajustes(C(palabra="clasificacion", tabla="t", columna_objetivo="y"), {})
        self.assertEqual(cfg.validacion, "auto")
    def test_formulario_de_clasificacion_ofrece_fecha(self):
        self.assertIn("columna_fecha", [c["campo"] for c in pr.PALABRAS_CLAVE["clasificacion"]["campos"]])
    def test_entrenar_avisa_cuando_hay_fecha_y_la_prueba_es_aleatoria(self):
        r = P.entrenar(self.df, CP(objetivo="y", tipo="clasificacion", pliegues=1), ["x", "x2"])
        self.assertTrue(any("mezcló las filas al azar" in a for a in r.advertencias))
        r2 = P.entrenar(self.df, CP(objetivo="y", tipo="clasificacion", pliegues=1, validacion="temporal", columna_fecha="fecha"), ["x", "x2"])
        self.assertFalse(any("mezcló las filas al azar" in a for a in r2.advertencias))
if __name__ == "__main__":
    unittest.main()