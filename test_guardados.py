import unittest, json
from tests.base import *
import analisis_guardados as G

class AnalisisGuardados(unittest.TestCase):
    def _cfg(self, **k): return C(palabra="proyeccion", tabla="ventas", columna_fecha="fecha", columna_valor="monto", **k).to_dict()
    def test_guardar_y_abrir_devuelve_la_misma_configuracion(self):  # F3.4
        g = []; G.guardar(g, "Ventas por sucursal", self._cfg(columna_entidad="sucursal"))
        d = G.obtener(g, "Ventas por sucursal")
        self.assertEqual(C.from_dict(d), C.from_dict(self._cfg(columna_entidad="sucursal")))
        self.assertEqual(G.nombres(g), ["Ventas por sucursal"])
    def test_el_mismo_nombre_reemplaza_sin_importar_mayusculas_ni_espacios(self):
        g = []; G.guardar(g, "Mi análisis", self._cfg(horizonte=7)); G.guardar(g, "  mi   ANÁLISIS ", self._cfg(horizonte=60))
        self.assertEqual(len(g), 1); self.assertEqual(G.obtener(g, "Mi análisis")["horizonte"], 60)
    def test_no_guarda_historial_ni_otros_guardados(self):
        g = []; G.guardar(g, "x", {**self._cfg(), "historial": [1, 2], "guardados": [3]})
        self.assertNotIn("historial", g[0]["config"]); self.assertNotIn("guardados", g[0]["config"])
    def test_nombres_invalidos_dan_error_claro(self):
        for malo in ("", "   ", None):
            with self.assertRaises(ValueError): G.guardar([], malo, self._cfg())
        with self.assertRaises(ValueError) as cm: G.guardar([], "a" * 100, self._cfg())
        self.assertIn("muy largo", str(cm.exception))
    def test_limite_de_guardados(self):
        g = []
        for i in range(G.MAX_GUARDADOS): G.guardar(g, f"a{i}", self._cfg())
        with self.assertRaises(ValueError): G.guardar(g, "uno más", self._cfg())
        G.guardar(g, "a0", self._cfg(horizonte=5))                  # reemplazar uno existente sí se puede
    def test_borrar(self):
        g = []; G.guardar(g, "a", self._cfg()); G.guardar(g, "b", self._cfg())
        self.assertTrue(G.borrar(g, " A ")); self.assertFalse(G.borrar(g, "no existe")); self.assertEqual(G.nombres(g), ["b"])
    def test_cabe_en_el_proyecto_y_tolera_datos_danados(self):
        g = []; G.guardar(g, "a", self._cfg(), "2026-10-01")
        self.assertEqual(json.loads(json.dumps(g)), g)
        self.assertEqual(G.nombres([None, 3, {"nombre": "x"}, {"nombre": "ok", "config": {}}, "t"]), ["ok"])
        self.assertEqual(G.limpiar(None), []); self.assertIsNone(G.obtener(None, "x"))
    def test_un_analisis_guardado_se_puede_ejecutar(self):
        d = pd.date_range("2024-01-01", periods=120, freq="D")
        df = pd.DataFrame({"fecha": d, "monto": 100 + np.sin(np.arange(120) * 2 * np.pi / 7) * 10})
        g = []; G.guardar(g, "Ventas", self._cfg(horizonte=7))
        r = pr.ejecutar(C.from_dict(G.obtener(g, "Ventas")), {"ventas": df})
        self.assertTrue(r.titular)
if __name__ == "__main__":
    unittest.main()
