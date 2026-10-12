import unittest
from tests.base import *

class DuracionCeros(unittest.TestCase):
    def _df(self, seed=0, n=300):
        rng = np.random.default_rng(seed)
        ini = pd.date_range("2024-01-01", periods=n, freq="D")
        dur = rng.choice([0, 0, 0, 1, 2, 3, 5, 8], n)
        return pd.DataFrame({"ini": ini, "fin": ini + pd.to_timedelta(dur, unit="D")}), dur
    def _cfg(self):
        return C(palabra="duracion", tabla="t", columna_inicio="ini", columna_fin="fin")
    def test_ceros_no_se_descartan_y_se_avisa(self):  # F1.6
        df, dur = self._df()
        r = ev.analizar_duracion(df, self._cfg())
        self.assertTrue(any("mismo día" in a for a in r.advertencias), r.advertencias)
        # con 0 días tratados como medio día, el típico ya no se infla (real ≈ 2)
        self.assertLess(r.metricas["mediana_dias"], 3.0, r.metricas["mediana_dias"])
        self.assertEqual(r.metricas["terminados"], len(df))
    def test_mediana_cercana_a_la_real(self):
        df, dur = self._df(seed=3, n=500)
        r = ev.analizar_duracion(df, self._cfg())
        self.assertLessEqual(abs(r.metricas["mediana_dias"] - np.median(dur)), 1.0)
    def test_columna_de_duracion_con_ceros(self):
        df = pd.DataFrame({"d": [0, 0, 1, 2, 3, 4, 5, 6, 0, 2, 3, 4]})
        r = ev.analizar_duracion(df, C(palabra="duracion", tabla="t", columna_valor="d"))
        self.assertTrue(any("mismo día" in a for a in r.advertencias))
if __name__ == "__main__":
    unittest.main()


class FinObservacion(unittest.TestCase):
    """F1.4: un elemento dado de baja no debe contarse como «sigue sin fallar» hasta el final de la tabla."""
    def _tabla(self, con_lecturas=True, seed=0):
        rng = np.random.default_rng(seed); ini = pd.Timestamp("2024-01-01"); filas = []
        for m in range(80):
            baja = rng.uniform(60, 300)            # día en que se retira del servicio (independiente de las fallas)
            t = 0.0
            while True:
                t += rng.exponential(100.0)        # vida con mediana real 69 días
                if t > baja: break
                filas.append({"maq": f"M{m:02d}", "fecha": ini + pd.Timedelta(days=float(t)), "tipo": "falla"})
            if con_lecturas:
                for d in range(0, int(baja), 15):
                    filas.append({"maq": f"M{m:02d}", "fecha": ini + pd.Timedelta(days=d), "tipo": "lectura"})
        filas.append({"maq": "M00", "fecha": ini + pd.Timedelta(days=330), "tipo": "lectura"})  # el registro más reciente de la tabla
        df = pd.DataFrame(filas); df["inst"] = ini
        return df
    def _cfg(self, **k):
        return C(palabra="vida_util", tabla="t", columna_fecha="fecha", columna_entidad="maq", columna_evento="tipo",
                 valor_evento="falla", columna_inicio="inst", **k)
    def test_con_lecturas_termina_en_el_ultimo_registro(self):
        df = self._tabla()
        med_auto = ev.analizar_vida_util(df, self._cfg()).metricas["mediana_dias"]
        med_global = ev.analizar_vida_util(df, self._cfg(fin_observacion="global")).metricas["mediana_dias"]
        print(f"\n[fin_observacion] real 69 | auto {med_auto:.0f} | global (antes) {med_global:.0f}")
        self.assertLess(abs(med_auto - 69.3), abs(med_global - 69.3))
        self.assertLess(abs(med_auto / 69.3 - 1), 0.25)
    def test_solo_eventos_avisa_del_sesgo(self):
        df = self._tabla(con_lecturas=False)
        df = df[df.tipo == "falla"]
        r = ev.analizar_vida_util(df, self._cfg())
        self.assertTrue(any("sin ningún registro" in a for a in r.advertencias), r.advertencias)
    def test_elementos_sin_inicio_se_avisan(self):
        df = self._tabla()
        df.loc[df.maq == "M05", "inst"] = pd.NaT
        r = ev.analizar_vida_util(df, self._cfg())
        self.assertTrue(any("fecha de inicio válida" in a and "M05" in a for a in r.advertencias), r.advertencias)
if __name__ == "__main__":
    unittest.main()