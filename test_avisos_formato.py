import unittest
from tests.base import *
import avisos as A
import formato as F
import prediccion as P
from prediccion import ConfigPrediccion as CP
from pronostico_base import ResultadoPronostico
from tests.test_calibracion import _datos_desbalanceados

class Niveles(unittest.TestCase):
    def test_un_string_suelto_es_informativo(self):  # F3.1: compatibilidad hacia atrás
        self.assertEqual(A.nivel_de("Se completaron los días sin datos."), A.INFORMATIVO)
        self.assertEqual(A.nivel_de(""), A.INFORMATIVO)
    def test_aviso_explicito_y_es_un_string(self):
        a = A.aviso("algo grave", A.CRITICO)
        self.assertIsInstance(a, str); self.assertEqual(a, "algo grave"); self.assertEqual(A.nivel_de(a), A.CRITICO)
        self.assertEqual(A.aviso("x", "inventado").nivel, A.INFORMATIVO)
    def test_orden_criticos_primero_y_estable(self):
        lista = ["nota 1", A.aviso("imp 1", A.IMPORTANTE), "nota 2", A.aviso("crit 1", A.CRITICO), A.aviso("imp 2", A.IMPORTANTE),
                 A.aviso("crit 2", A.CRITICO)]
        orden = [str(a) for a in A.ordenar_avisos(lista)]
        self.assertEqual(orden, ["crit 1", "crit 2", "imp 1", "imp 2", "nota 1", "nota 2"])
        self.assertEqual(lista[0], "nota 1")                         # no toca la lista original
    def test_los_avisos_reales_tienen_el_nivel_esperado(self):
        # guarda contra cambios de redacción: estos textos los produce el código de verdad
        df = _datos_desbalanceados(800)
        df["pista"] = df.y.astype(float); df["fecha"] = pd.date_range("2022-01-01", periods=len(df), freq="D")
        r = P.entrenar(df, CP(objetivo="y", tipo="clasificacion", salida="probabilidad", pliegues=1), ["x1", "x2", "pista"])
        niveles = {str(a)[:40]: a.nivel for a in A.ordenar_avisos(r.advertencias)}
        crit = [a for a in A.ordenar_avisos(r.advertencias) if a.nivel == A.CRITICO]
        self.assertTrue(crit, r.advertencias); self.assertIn("podrían ser una trampa", crit[0])
        self.assertEqual(A.ordenar_avisos(r.advertencias)[0].nivel, A.CRITICO)          # la fuga siempre arriba
        self.assertTrue(any(a.nivel == A.IMPORTANTE and "mezcló las filas al azar" in a for a in A.ordenar_avisos(r.advertencias)))
    def test_contrato_resultado_ordena_y_etiqueta(self):
        r = ResultadoPronostico("x", "X", "titular", advertencias=["nota", A.aviso("grave", A.CRITICO), "casi no mejora a la respuesta"])
        self.assertEqual([a.nivel for a in r.avisos_ordenados()], [A.CRITICO, A.IMPORTANTE, A.INFORMATIVO])
        lineas = r.lineas_resumen()
        self.assertEqual(lineas[-3:], ["Crítico: grave", "Importante: casi no mejora a la respuesta", "Nota: nota"])
    def test_avisos_de_series_y_eventos_importantes(self):
        sr = st.preparar_serie(pd.DataFrame({"f": pd.date_range("2024-01-01", periods=100, freq="D"), "v": np.arange(100.0)})
                               .drop(index=range(10, 60, 2)), "f", "v", "suma", "D")
        self.assertTrue(any(a.nivel == A.IMPORTANTE and "no tenía datos" in a for a in A.ordenar_avisos(sr.advertencias)), sr.advertencias)
        p = st.proyectar(serie(gen("rw", 150, np.random.default_rng(0))), 90, 0.8)
        self.assertTrue(any(a.nivel == A.IMPORTANTE and "extrapolación" in a for a in A.ordenar_avisos(st._avisos_proyeccion(serie(gen("rw", 150, np.random.default_rng(0))), p, 90))))

class AvisoPesosYCalibracion(unittest.TestCase):
    def test_no_dice_que_sirve_solo_para_ordenar_si_la_calibracion_lo_dejo_confiable(self):  # contradicción corregida
        df = _datos_desbalanceados()
        r = P.entrenar(df, CP(objetivo="y", tipo="clasificacion", salida="probabilidad", pliegues=3, balancear="si"), ["x1", "x2"])
        self.assertEqual(r.tipo_salida, "probabilidad")
        self.assertFalse(any("sirven para ordenar casos" in a for a in r.advertencias), r.advertencias)
        self.assertTrue(any("se corrigieron las probabilidades" in a for a in r.advertencias))
    def test_si_no_se_pudo_calibrar_si_lo_dice(self):
        df = _datos_desbalanceados().head(150)
        r = P.entrenar(df, CP(objetivo="y", tipo="clasificacion", salida="probabilidad", pliegues=3, balancear="si"), ["x1", "x2"])
        self.assertEqual(r.tipo_salida, "ranking")
        self.assertTrue(any("sirven para ordenar casos" in a for a in r.advertencias))

class FormatoCentral(unittest.TestCase):
    def tearDown(self): F.set_formato_numerico("estandar")
    def test_dec(self):
        F.set_formato_numerico("estandar")
        self.assertEqual((F.dec(1234.5, 1, True), F.dec(1234.5, 1), F.dec(0.5, 3), F.dec(12, 0, True)), ("1,234.5", "1234.5", "0.500", "12"))
        F.set_formato_numerico("chileno")
        self.assertEqual((F.dec(1234.5, 1, True), F.dec(1234.5, 1), F.dec(0.5, 3), F.dec(1234567, 0, True)), ("1.234,5", "1234,5", "0,500", "1.234.567"))
    def test_dec_con_valores_raros(self):
        for v in (None, float("nan"), float("inf"), "x"):
            self.assertEqual(F.dec(v, 2), "—")
    def test_ningun_texto_de_resultado_mezcla_formatos(self):
        F.set_formato_numerico("chileno")
        d = pd.date_range("2024-01-01", periods=120, freq="D")
        df = pd.DataFrame({"f": d, "c": np.full(120, 3.0), "s": 1234.0})
        r = pr.ejecutar(C(palabra="agotamiento", tabla="t", columna_fecha="f", columna_valor="c", stock_actual=5000.5), {"t": df})
        texto = r.texto()
        self.assertNotRegex(texto, r"\d,\d{3}(?!\d)\.\d")           # nada como 1,234.5 en formato chileno
    def test_auc_respeta_el_formato(self):
        df = _datos_desbalanceados(600)
        r = P.entrenar(df, CP(objetivo="y", tipo="clasificacion", pliegues=1), ["x1", "x2"])
        tarj = " ".join(str(t) for t in r.tarjetas())
        F.set_formato_numerico("chileno"); chile = " ".join(str(t) for t in r.tarjetas())
        F.set_formato_numerico("estandar"); estandar = " ".join(str(t) for t in r.tarjetas())
        self.assertNotEqual(chile, estandar)
if __name__ == "__main__":
    unittest.main()
