import unittest, json
from tests.base import *
import historial as H

class Res:           # resultado mínimo
    def __init__(self, mejora=None, confianza="media", palabra="proyeccion", tipo=""):
        self.metricas = {"mejora": mejora} if mejora is not None else {}
        self.confianza, self.palabra, self.tipo_salida = confianza, palabra, tipo

CFG = lambda **k: C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="v", **k)

class Historial(unittest.TestCase):
    def test_primera_vez_no_hay_aviso_y_se_guarda(self):  # F2.6
        h = []
        self.assertEqual(H.registrar_y_comparar(h, CFG(), Res(0.4), 500, "2026-10-01"), [])
        self.assertEqual(len(h), 1); self.assertEqual(h[0]["mejora"], 0.4); self.assertEqual(h[0]["n"], 500)
    def test_avisa_si_la_mejora_cae(self):
        h = []; H.registrar_y_comparar(h, CFG(), Res(0.40), 500, "2026-10-01")
        av = H.registrar_y_comparar(h, CFG(), Res(0.12), 600, "2026-10-12")
        self.assertEqual(len(av), 1)
        self.assertIn("1 oct 2026", av[0]); self.assertIn("mejoraba 40%", av[0]); self.assertIn("hoy mejora 12%", av[0])
        self.assertIn("comportamiento haya cambiado", av[0])
    def test_avisa_si_la_mejora_sube_sin_alarmar(self):
        h = []; H.registrar_y_comparar(h, CFG(), Res(0.10), 500, "2026-10-01")
        av = H.registrar_y_comparar(h, CFG(), Res(0.35), 500, "2026-10-12")
        self.assertEqual(len(av), 1); self.assertNotIn("conviene revisar", av[0])
    def test_cambio_chico_no_molesta(self):
        h = []; H.registrar_y_comparar(h, CFG(), Res(0.40), 500, "2026-10-01")
        self.assertEqual(H.registrar_y_comparar(h, CFG(), Res(0.36), 500, "2026-10-12"), [])
    def test_compara_con_la_ultima_vez_del_mismo_analisis_y_no_con_otro(self):
        h = []; H.registrar_y_comparar(h, CFG(), Res(0.40), 500, "2026-10-01")
        otro = C(palabra="proyeccion", tabla="t", columna_fecha="f", columna_valor="otra_columna")
        self.assertEqual(H.registrar_y_comparar(h, otro, Res(0.05), 500, "2026-10-02"), [])
        self.assertEqual(len(H.registrar_y_comparar(h, CFG(), Res(0.05), 500, "2026-10-03")), 1)
    def test_el_horizonte_no_cambia_la_identidad_del_analisis(self):
        self.assertEqual(H.clave_analisis(CFG(horizonte=7)), H.clave_analisis(CFG(horizonte=60)))
        self.assertNotEqual(H.clave_analisis(CFG()), H.clave_analisis(CFG(columna_entidad="x")))
    def test_avisa_si_baja_la_confianza(self):
        h = []; H.registrar_y_comparar(h, CFG(), Res(None, "alta"), 500, "2026-10-01")
        av = H.registrar_y_comparar(h, CFG(), Res(None, "baja"), 500, "2026-10-12")
        self.assertTrue(av and "confianza era alta" in av[0])
    def test_sin_mejora_ni_confianza_comparable_no_inventa(self):
        h = []; H.registrar_y_comparar(h, CFG(), Res(None, "media"), 500, "2026-10-01")
        self.assertEqual(H.registrar_y_comparar(h, CFG(), Res(None, "media"), 500, "2026-10-12"), [])
    def test_se_recorta_y_es_json(self):
        h = []
        for i in range(30):
            H.registrar_y_comparar(h, CFG(), Res(0.3), 500, pd.Timestamp("2026-01-01") + pd.Timedelta(days=i))
        self.assertEqual(len(h), H.MAX_POR_ANALISIS)
        self.assertEqual(h[-1]["fecha"][:10], "2026-01-30")
        self.assertEqual(json.loads(json.dumps(h)), h)                         # cabe en el .hadarproy
        for i in range(50):
            H.registrar_y_comparar(h, C(palabra="tendencia", tabla="t", columna_valor=f"v{i}"), Res(0.1), 10, "2026-02-01")
        self.assertLessEqual(len(h), H.MAX_TOTAL)
    def test_lo_guardado_danado_se_ignora(self):
        malo = [None, 5, {"clave": 3}, {"clave": "x", "fecha": "2026-01-01 10:00"}, "texto"]
        self.assertEqual(len(H.limpiar(malo)), 1); self.assertEqual(H.limpiar(None), [])
    def test_formato_chileno_en_el_aviso(self):
        import pronostico_base as pb
        pb.set_formato_numerico("chileno")
        try:
            h = []; H.registrar_y_comparar(h, CFG(), Res(0.405), 500, "2026-10-01")
            av = H.registrar_y_comparar(h, CFG(), Res(0.05), 500, "2026-10-12")
            self.assertTrue(av)
        finally:
            pb.set_formato_numerico("estandar")
    def test_con_resultados_reales(self):
        rng = np.random.default_rng(0); d = pd.date_range("2024-01-01", periods=200, freq="D")
        df = pd.DataFrame({"f": d, "v": 100 + 10 * np.sin(np.arange(200) * 2 * np.pi / 7) + rng.normal(0, 2, 200)})
        ruido = pd.DataFrame({"f": d, "v": rng.normal(0, 30, 200)})
        h = []
        r1 = pr.ejecutar(CFG(horizonte=14), {"t": df}); self.assertIsNotNone(r1.metricas.get("mejora"))
        self.assertEqual(H.registrar_y_comparar(h, CFG(), r1, 200, "2026-10-01"), [])
        r2 = pr.ejecutar(CFG(horizonte=14), {"t": ruido})
        av = H.registrar_y_comparar(h, CFG(), r2, 200, "2026-10-12")
        print(f"\n[historial] mejora antes {r1.metricas['mejora']:.2f} → ahora {r2.metricas['mejora']:.2f}: {av}")
        self.assertTrue(av)
    def test_clasificacion_y_falla_modelo_exponen_mejora(self):
        rng = np.random.default_rng(0); n = 500; x = rng.normal(size=n)
        df = pd.DataFrame({"x": x, "y": (x + rng.normal(0, .5, n) > 0).astype(int)})
        r = pr.ejecutar(C(palabra="clasificacion", tabla="t", columna_objetivo="y"), {"t": df})
        self.assertIsNotNone(r.metricas.get("mejora"))

class HistorialDeModelos(unittest.TestCase):
    """«Entrenar un modelo» también compara con la última vez."""
    def _res(self, ruido, n=500, seed=0):
        import prediccion as P
        rng = np.random.default_rng(seed); x = rng.normal(size=n)
        df = pd.DataFrame({"x": x, "y": (x + rng.normal(0, ruido, n) > 0).astype(int)})
        cfg = P.ConfigPrediccion(objetivo="y", tipo="clasificacion", pliegues=1)
        return cfg, P.entrenar(df, cfg, ["x"])
    def test_un_modelo_registra_su_mejora_y_avisa_si_cae(self):
        cfg, r1 = self._res(0.3); _, r2 = self._res(5.0)
        h = []
        self.assertEqual(H.registrar_y_comparar(h, cfg.to_dict(), r1, 500, "2026-10-01"), [])
        self.assertIsNotNone(h[0]["mejora"]); self.assertEqual(h[0]["palabra"], "modelo")
        av = H.registrar_y_comparar(h, cfg.to_dict(), r2, 500, "2026-10-12")
        print(f"\n[historial modelo] {h[0]['mejora']:.2f} → {h[1]['mejora']:.2f}: {av}")
        self.assertTrue(av and "conviene revisar" in av[0])
    def test_modelos_distintos_no_se_comparan_entre_si(self):
        import prediccion as P
        cfg, r = self._res(0.3)
        otro = P.ConfigPrediccion(objetivo="otra_columna", tipo="clasificacion", pliegues=1)
        h = []; H.registrar_y_comparar(h, cfg.to_dict(), r, 500, "2026-10-01")
        self.assertEqual(H.registrar_y_comparar(h, otro.to_dict(), r, 500, "2026-10-02"), [])
if __name__ == "__main__":
    unittest.main()