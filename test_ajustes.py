import unittest
from tests.base import *
import ajustes as A
import pronostico_base as pb

class Ajustes(unittest.TestCase):
    def setUp(self): pb.set_formato_numerico("estandar")
    def tearDown(self): pb.set_formato_numerico("estandar")   # los demás tests asumen el formato estándar
    def test_round_trip_y_defectos(self):
        a = A.AjustesPrediccion(formato_numerico="estandar", fin_observacion="global", ceros_mismo_dia="descartar",
                                dias_aviso_datos_antiguos=90)
        self.assertEqual(A.AjustesPrediccion.from_dict(a.to_dict()), a)
        self.assertEqual(A.AjustesPrediccion.from_dict(None), A.AjustesPrediccion())
        self.assertEqual(A.AjustesPrediccion().formato_numerico, "chileno")
    def test_proyecto_antiguo_sin_ajustes_abre_con_defectos(self):
        self.assertEqual(A.AjustesPrediccion.from_dict({}), A.AjustesPrediccion())
    def test_valores_invalidos_vuelven_al_defecto(self):
        a = A.AjustesPrediccion.from_dict({"fin_observacion": "inventado", "dias_aviso_datos_antiguos": -3,
                                           "formato_numerico": "estandar", "clave_rara": 1})
        self.assertEqual(a.fin_observacion, "auto"); self.assertEqual(a.dias_aviso_datos_antiguos, 45)
        self.assertEqual(a.formato_numerico, "estandar")
    def test_opciones_tienen_texto_y_defecto_valido(self):
        d = A.AjustesPrediccion().to_dict()
        self.assertEqual({k for k, *_ in A.OPCIONES}, set(d))
        for k, titulo, porque, ops in A.OPCIONES:
            self.assertTrue(titulo and porque and len(ops) >= 2)
            self.assertIn(d[k], [v for v, _ in ops], k)
    def test_aplicar_copia_a_la_config_del_analisis(self):
        cfg = C(palabra="duracion", tabla="t")
        A.aplicar_ajustes(cfg, {"fin_observacion": "global", "ceros_mismo_dia": "descartar", "dias_aviso_datos_antiguos": 15})
        self.assertEqual((cfg.fin_observacion, cfg.ceros_mismo_dia, cfg.dias_aviso_datos_antiguos), ("global", "descartar", 15))
        self.assertEqual(C.from_dict(cfg.to_dict()), cfg)
    def test_formato_chileno(self):
        pb.set_formato_numerico("chileno")
        self.assertEqual(pb.num_txt(1234567.8), "1.234.568"); self.assertEqual(pb.num_txt(3.14159), "3,14")
        self.assertEqual(pb.pct_txt(0.1534, 1, True), "+15,3%"); self.assertEqual(pb.num_txt(0.00123), "0,00123")
        pb.set_formato_numerico("estandar")
        self.assertEqual(pb.num_txt(1234567.8), "1,234,568"); self.assertEqual(pb.num_txt(3.14159), "3.14")
    def test_ceros_se_pueden_descartar_con_aviso(self):
        rng = np.random.default_rng(0); n = 300
        ini = pd.date_range("2024-01-01", periods=n, freq="D"); dur = rng.choice([0, 0, 0, 1, 2, 3, 5, 8], n)
        df = pd.DataFrame({"a": ini, "b": ini + pd.to_timedelta(dur, unit="D")})
        cfg = lambda **k: C(palabra="duracion", tabla="t", columna_inicio="a", columna_fin="b", **k)
        medio = ev.analizar_duracion(df, cfg()).metricas["mediana_dias"]
        desc = ev.analizar_duracion(df, cfg(ceros_mismo_dia="descartar"))
        self.assertGreater(desc.metricas["mediana_dias"], medio)
        self.assertTrue(any("como pediste en Ajustes" in a for a in desc.advertencias))
    def test_umbral_de_aviso_de_datos_antiguos_es_configurable(self):
        f = pd.date_range("2024-01-01", periods=60, freq="D")
        df = pd.DataFrame({"f": f, "v": np.arange(60.0)})
        hoy = "2024-03-31"                                        # 31 días después del último dato (1 mar)
        self.assertIsNotNone(pb.aviso_datos_antiguos(df, ["f"], hoy=hoy, umbral_dias=15))
        self.assertIsNone(pb.aviso_datos_antiguos(df, ["f"], hoy=hoy, umbral_dias=45))
if __name__ == "__main__":
    unittest.main()
