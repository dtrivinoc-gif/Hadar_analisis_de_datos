import unittest, warnings
from tests.base import *

class CoberturaFueraDeMuestra(unittest.TestCase):
    """F0a: el rango del 80% debe acertar entre ~70% y ~90% sobre datos que el modelo NO vio."""
    def _cob(self, kind, h, n_tr=150, reps=20):
        out = []
        for s in range(reps):
            y = gen(kind, n_tr + h, np.random.default_rng(s))
            p = st.proyectar(serie(y[:n_tr]), h, 0.8)
            out.append(((y[n_tr:] >= p.bajo) & (y[n_tr:] <= p.alto)).mean())
        return float(np.mean(out))
    def test_camino_aleatorio_corto(self):
        c = self._cob("rw", 10); self.assertTrue(0.70 <= c <= 0.90, c)
    def test_estacional(self):
        c = self._cob("estacional", 14); self.assertTrue(0.70 <= c <= 0.90, c)
    def test_camino_aleatorio_horizonte_largo(self):
        # F1.2: más allá de h_eval (= n//5) el rango se extrapola. Antes acertaba ~0.73 en ese tramo; ahora ~0.78.
        out = []
        for s in range(30):
            y = gen("rw", 240, np.random.default_rng(300 + s))
            p = st.proyectar(serie(y[:150]), 90, 0.8)
            out.append(((y[150:] >= p.bajo) & (y[150:] <= p.alto))[p.h_eval:].mean())
        c = float(np.mean(out)); self.assertTrue(0.75 <= c <= 0.90, c)
    def test_avisa_cuando_el_horizonte_supera_lo_comprobado(self):
        y = gen("rw", 150, np.random.default_rng(0)); sr = serie(y)
        p = st.proyectar(sr, 90, 0.8)
        self.assertTrue(any("extrapolación" in a for a in st._avisos_proyeccion(sr, p, 90)))
    def test_series_con_tendencia_no_quedan_absurdamente_anchas(self):
        out = []
        for s in range(20):
            y = gen("tendencia", 240, np.random.default_rng(300 + s))
            p = st.proyectar(serie(y[:150]), 90, 0.8)
            out.append(((y[150:] >= p.bajo) & (y[150:] <= p.alto))[p.h_eval:].mean())
        self.assertLessEqual(float(np.mean(out)), 0.99)

class AgregacionAuto(unittest.TestCase):
    def test_ingles_y_espanol(self):  # F1.3
        for col in ["temperature", "balance", "price", "temperatura"]:
            self.assertEqual(st.agregacion_auto(col), "promedio", col)
    def test_ventas_suman(self):
        for col in ["ventas", "sales", "revenue"]:
            self.assertEqual(st.agregacion_auto(col), "suma", col)

class Agotamiento(unittest.TestCase):
    def _df(self, cons, n=120):
        return pd.DataFrame({"f": pd.date_range("2024-01-01", periods=n, freq="D"), "consumo": cons})
    def _cfg(self, **k):
        return C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="consumo", **k)
    def test_consumo_constante(self):  # F0c
        r = st.analizar_agotamiento(self._df(np.full(120, 5.0)), self._cfg(stock_actual=100))
        self.assertIn("20 días", r.titular)
    def test_sin_consumo_no_dice_alcanza(self):  # F1.7
        r = st.analizar_agotamiento(self._df(np.zeros(120)), self._cfg(stock_actual=100))
        self.assertNotIn("alcanza", r.titular)
        self.assertNotIn("el stock sin consumo", r.titular)
    def test_stock_actual_con_varios_elementos_explica(self):  # F1.7: el error ya no culpa a «la columna de stock»
        d = pd.date_range("2024-01-01", periods=120, freq="D")
        df = pd.DataFrame({"f": np.tile(d, 2), "consumo": np.full(240, 5.0), "prod": ["a"]*120 + ["b"]*120})
        with self.assertRaises(ValueError) as cm:
            st.analizar_agotamiento(df, self._cfg(columna_entidad="prod", stock_actual=100))
        self.assertIn("a cuál corresponde", str(cm.exception))
    def test_stock_actual_con_un_elemento(self):
        d = pd.date_range("2024-01-01", periods=120, freq="D")
        df = pd.DataFrame({"f": d, "consumo": np.full(120, 5.0), "prod": ["a"]*120})
        r = st.analizar_agotamiento(df, self._cfg(columna_entidad="prod", stock_actual=100))
        self.assertIn("20 días", r.titular)
    def test_cobertura_larga_titular_legible(self):
        r = st.analizar_agotamiento(self._df(np.full(120, 0.05)), self._cfg(stock_actual=100))
        self.assertIn("dura más de", r.titular); self.assertNotIn("alcanza", r.titular)
    def test_elemento_sin_registros_recientes_se_lista(self):
        d = pd.date_range("2024-01-01", periods=120, freq="D")
        df = pd.DataFrame({"f": list(d) + list(d[:30]), "consumo": np.full(150, 5.0),
                           "prod": ["a"]*120 + ["b"]*30, "stock": np.full(150, 100.0)})
        r = st.analizar_agotamiento(df, C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="consumo",
                                          columna_entidad="prod", columna_stock="stock"))
        self.assertTrue(any("«b»" in a or " b" in a or "b." in a for a in r.advertencias), r.advertencias)

class Escenario(unittest.TestCase):
    """F0d: con una elasticidad real conocida (-1.5), el rango del 95% debe contenerla la gran mayoría de las veces."""
    def test_elasticidad_conocida_dentro_del_rango(self):
        dentro, reps = 0, 30
        for s in range(reps):
            rng = np.random.default_rng(1000 + s); n = 400
            x = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.5 + rng.normal(0, 0.08, n))
            y = 1000 * (x / 10) ** -1.5 * rng.normal(1, 0.1, n)
            df = pd.DataFrame({"f": pd.date_range("2024-01-01", periods=n, freq="D"), "y": y, "x": x})
            r = st.analizar_escenario(df, C(palabra="escenario", tabla="t", columna_fecha="f", columna_valor="y",
                                            columna_driver="x", cambio_pct=10.0))
            dentro += r.metricas["efecto_min"] <= 1.1 ** -1.5 - 1 <= r.metricas["efecto_max"]
        self.assertGreaterEqual(dentro / reps, 0.80, dentro / reps)   # nominal 95%; medido ≈ 0.90-0.92
    def test_sin_variacion_en_la_variable_da_error_claro(self):
        df = pd.DataFrame({"f": pd.date_range("2024-01-01", periods=60, freq="D"), "y": np.arange(60) + 5.0, "x": np.full(60, 3.0)})
        with self.assertRaises(ValueError) as cm:
            st.analizar_escenario(df, C(palabra="escenario", tabla="t", columna_fecha="f", columna_valor="y", columna_driver="x"))
        self.assertIn("nunca cambió", str(cm.exception))

if __name__ == "__main__":
    unittest.main()


class CoberturaHonesta(unittest.TestCase):
    """F1.1: la cobertura que decide la confianza debe medirse con pruebas distintas de las que fijan el ancho."""
    def test_marca_honesta_en_series_largas(self):
        p = st.proyectar(serie(gen("estacional", 150, np.random.default_rng(0))), 10, 0.8)
        self.assertTrue(p.cobertura_honesta); self.assertFalse(np.isnan(p.cobertura))
    def test_series_cortas_no_pueden_tener_confianza_alta(self):
        y = gen("tendencia", 30, np.random.default_rng(0))
        p = st.proyectar(serie(y), 6, 0.8)
        self.assertNotEqual(st._confianza_proy(p, 30)[0], "alta")
    def test_cobertura_medida_se_parece_a_la_real_fuera_de_muestra(self):
        medidas, reales = [], []
        for s in range(30):
            y = gen("estacional", 160, np.random.default_rng(500 + s))
            p = st.proyectar(serie(y[:150]), 10, 0.8)
            medidas.append(p.cobertura)
            reales.append(((y[150:] >= p.bajo) & (y[150:] <= p.alto)).mean())
        self.assertLess(abs(np.nanmean(medidas) - np.mean(reales)), 0.10, (np.nanmean(medidas), np.mean(reales)))
    def test_confianza_alta_exige_cobertura_honesta(self):
        p = st.proyectar(serie(gen("tendencia", 150, np.random.default_rng(1))), 10, 0.8)
        p.cobertura_honesta = False
        self.assertNotEqual(st._confianza_proy(p, 150)[0], "alta")


class FinesDeSemanaYUltimoPeriodo(unittest.TestCase):
    def _df(self, solo_laborales=True, n=120):
        f = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame({"f": f, "v": np.random.default_rng(0).poisson(20, n).astype(float)})
        return df[df.f.dt.dayofweek < 5] if solo_laborales else df
    def test_aviso_especifico_si_solo_faltan_sabado_y_domingo(self):  # F1.8
        s = st.preparar_serie(self._df(), "f", "v", "suma", "D")
        self.assertTrue(any("sábado y domingo" in a for a in s.advertencias), s.advertencias)
        self.assertFalse(any("poco fiables" in a for a in s.advertencias))
    def test_faltantes_al_azar_siguen_con_el_aviso_general(self):
        df = self._df(solo_laborales=False)
        df = df.drop(df.sample(frac=0.3, random_state=1).index)
        s = st.preparar_serie(df, "f", "v", "suma", "D")
        self.assertTrue(any("poco fiables" in a for a in s.advertencias), s.advertencias)
    def test_ultimo_dia_a_medio_llenar_se_avisa(self):  # F1.9
        f = pd.date_range("2024-01-01", periods=40 * 10, freq="2h40min")
        df = pd.DataFrame({"f": f, "v": 1.0})
        ultimo = df.f.dt.normalize().max()
        df = pd.concat([df[df.f.dt.normalize() < ultimo], df[df.f.dt.normalize() == ultimo].head(1)])
        s = st.preparar_serie(df, "f", "v", "suma", "D")
        self.assertTrue(any("último día" in a for a in s.advertencias), s.advertencias)
    def test_proyeccion_expone_rmse_con_nombre_honesto(self):
        p = st.proyectar(serie(gen("estacional", 100, np.random.default_rng(0))), 7, 0.8)
        self.assertTrue(hasattr(p, "rmse_modelo") and not hasattr(p, "mae_modelo"))


class RuidoNoEsPatron(unittest.TestCase):
    """Hallazgo nuevo: el ruido puro recibía confianza «alta» porque «repetir lo último» es mala referencia para ruido."""
    def test_ruido_puro_no_tiene_confianza_alta(self):
        confs, mejoras = [], []
        for s in range(10):
            y = 100 + np.random.default_rng(s).normal(0, 10, 200)
            p = st.proyectar(serie(y), 14, 0.8)
            confs.append(st._confianza_proy(p, 200)[0]); mejoras.append(p.mejora)
        self.assertNotIn("alta", confs); self.assertLess(float(np.mean(mejoras)), 0.05)
    def test_con_tendencia_o_patron_sigue_habiendo_confianza_alta(self):
        confs = []
        for s in range(8):
            y = gen("tendencia", 200, np.random.default_rng(s))
            confs.append(st._confianza_proy(st.proyectar(serie(y), 14, 0.8), 200)[0])
        self.assertGreaterEqual(confs.count("alta"), 7)