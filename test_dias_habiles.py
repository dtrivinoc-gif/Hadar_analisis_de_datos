import unittest
from tests.base import *
import ajustes as A

def _laboral(n_dias=210, tasa=5.0, seed=0, con_finde=False):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2024-01-01", periods=n_dias, freq="D")        # el 1 ene 2024 es lunes
    v = rng.poisson(tasa, n_dias).astype(float)
    df = pd.DataFrame({"f": d, "v": v})
    return df if con_finde else df[d.dayofweek < 5].reset_index(drop=True)

class DiasHabiles(unittest.TestCase):
    def test_por_defecto_no_cambia_nada(self):
        s = st.preparar_serie(_laboral(), "f", "v", "suma", "D")
        self.assertEqual(s.freq, "D")
        s2 = st.preparar_serie(_laboral(), "f", "v", "suma", "D", fines_de_semana="rellenar")
        self.assertEqual(s2.freq, "D"); self.assertEqual(len(s.y), len(s2.y))
    def test_ignorar_cuenta_en_dias_habiles(self):
        s = st.preparar_serie(_laboral(), "f", "v", "suma", "D", fines_de_semana="ignorar")
        self.assertEqual(s.freq, "B")
        self.assertTrue((s.fechas.dayofweek < 5).all())
        self.assertEqual(len(s.y), len(_laboral()))                      # ningún día hábil inventado ni perdido
        self.assertTrue(any("Se ignoraron los fines de semana" in a for a in s.advertencias))
        self.assertFalse(any("sábado y domingo" in a and "Los días sin datos" in a for a in s.advertencias))
    def test_si_hay_datos_en_fin_de_semana_no_se_ignora(self):
        s = st.preparar_serie(_laboral(con_finde=True), "f", "v", "suma", "D", fines_de_semana="ignorar")
        self.assertEqual(s.freq, "D")
        self.assertTrue(any("No se ignoraron los fines de semana" in a for a in s.advertencias))
    def test_la_proyeccion_cae_en_dias_habiles_y_el_horizonte_es_en_dias_habiles(self):
        r = pr.ejecutar(C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="v", horizonte=10, fines_de_semana="ignorar"), {"t": _laboral()})
        fechas = pd.to_datetime(r.tabla.iloc[:, 0])
        self.assertEqual(len(fechas), 10); self.assertTrue((fechas.dt.dayofweek < 5).all())
        ult = _laboral().f.max()
        self.assertEqual(fechas.iloc[0], ult + pd.offsets.BDay(1))
        self.assertIn("días hábiles", r.titular + " ".join(r.detalle))
    def test_con_rellenar_la_proyeccion_incluye_fines_de_semana(self):
        r = pr.ejecutar(C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="v", horizonte=10), {"t": _laboral()})
        self.assertTrue((pd.to_datetime(r.tabla.iloc[:, 0]).dt.dayofweek >= 5).any())
    def test_estacionalidad_semanal_tiene_cinco_dias(self):
        rng = np.random.default_rng(1); d = pd.date_range("2024-01-01", periods=300, freq="D"); d = d[d.dayofweek < 5]
        f = np.array([1.0, 1.0, 1.0, 1.0, 2.0])[d.dayofweek]
        df = pd.DataFrame({"f": d, "v": 100 * f * rng.normal(1, .05, len(d))})
        r = pr.ejecutar(C(palabra="estacionalidad", tabla="t", columna_fecha="f", columna_valor="v", fines_de_semana="ignorar"), {"t": df})
        self.assertEqual(r.tarjetas[1][0], "viernes"); self.assertEqual(len(r.tabla), 5)
    def test_agotamiento_cuenta_dias_habiles(self):
        d = pd.date_range("2024-03-04", periods=120, freq="D"); d = d[d.dayofweek < 5]          # empieza un lunes
        df = pd.DataFrame({"f": d, "c": np.full(len(d), 5.0)})
        cfg = lambda **k: C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", stock_actual=100, **k)
        hab = pr.ejecutar(cfg(fines_de_semana="ignorar"), {"t": df})
        self.assertIn("20 días hábiles", hab.titular)                 # 100 de stock a 5 por día hábil
        corr = pr.ejecutar(cfg(), {"t": df})
        self.assertIn("28 días", corr.titular)                         # relleno con 0: 5·5/7 por día → 28 días corridos
    def test_el_ajuste_se_copia_a_la_config_y_se_guarda(self):
        cfg = A.aplicar_ajustes(C(palabra="proyeccion", tabla="t"), {"fines_de_semana": "ignorar"})
        self.assertEqual(cfg.fines_de_semana, "ignorar"); self.assertEqual(C.from_dict(cfg.to_dict()), cfg)
        self.assertEqual(A.AjustesPrediccion().fines_de_semana, "rellenar")
        self.assertEqual(A.AjustesPrediccion.from_dict({"fines_de_semana": "otra cosa"}).fines_de_semana, "rellenar")
    def test_series_cortas_no_activan_dias_habiles(self):
        d = pd.date_range("2024-01-01", periods=5, freq="D")             # lunes a viernes: no se puede afirmar nada del fin de semana
        with self.assertRaises(ValueError) as cm:
            st.preparar_serie(pd.DataFrame({"f": d, "v": [1.0, 2, 3, 4, 5]}), "f", "v", "suma", "D", fines_de_semana="ignorar")
        self.assertIn("días de historial", str(cm.exception)); self.assertNotIn("hábiles", str(cm.exception))
    def test_todas_las_palabras_de_series_corren_en_dias_habiles(self):
        df = _laboral(300); df["t"] = 60 + 0.05 * np.arange(len(df)) + np.random.default_rng(0).normal(0, 1, len(df))
        for k, extra in (("proyeccion", {}), ("tendencia", {}), ("estacionalidad", {}), ("umbral", dict(umbral=80, columna_valor="t")),
                         ("saturacion", dict(capacidad=40)), ("escenario", dict(columna_driver="t", cambio_pct=5)),
                         ("agotamiento", dict(stock_actual=500))):
            with self.subTest(palabra=k):
                kw = dict(palabra=k, tabla="t", columna_fecha="f", columna_valor="v", fines_de_semana="ignorar"); kw.update(extra)
                r = pr.ejecutar(C(**kw), {"t": df}); self.assertTrue(r.titular)
if __name__ == "__main__":
    unittest.main()
