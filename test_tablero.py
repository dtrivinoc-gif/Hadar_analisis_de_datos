"""Pruebas del tablero común (predicciones por fila que otras pestañas leen).

Se ejecuta con el resto:  python -m unittest discover -s tests -t .
Las dos últimas usan el motor real (entrenar + predecir) sobre datos inventados
de respuesta conocida.
"""
import unittest

import numpy as np
import pandas as pd

try:
    from hadar.tablero import Tablero, alinear_salida
except ImportError:      # carpeta plana, sin paquete
    from tablero import Tablero, alinear_salida


def _tabla(n=6, con_id=True):
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "grupo": list("ABABAB")[:n]})
    if con_id:
        df.insert(0, "id", np.arange(100, 100 + n))
    df.index = pd.RangeIndex(10, 10 + n)        # índice que NO empieza en 0, a propósito
    return df


class TestAlinear(unittest.TestCase):
    def test_por_id_aunque_el_orden_cambie(self):
        df = _tabla()
        salida = pd.DataFrame({"id": df["id"].to_numpy()[::-1], "y": np.arange(6.0)[::-1] * 10})
        out = alinear_salida(salida, df)
        self.assertTrue(out.index.equals(df.index))
        self.assertEqual(out["y"].tolist(), [0, 10, 20, 30, 40, 50])      # cada una en su fila

    def test_por_posicion_si_el_id_es_sintetico(self):
        df = _tabla(con_id=False)
        salida = pd.DataFrame({"Id": np.arange(6), "y": np.arange(6.0)})
        out = alinear_salida(salida, df)
        self.assertEqual(out["y"].tolist(), list(range(6)))
        self.assertTrue(out.index.equals(df.index))

    def test_id_repetido_no_se_adivina(self):
        df = _tabla()
        df["id"] = 1
        salida = pd.DataFrame({"id": df["id"].to_numpy(), "y": np.arange(6.0)})
        with self.assertRaises(ValueError):
            alinear_salida(salida, df)                      # sin permiso para ir por posición
        out = alinear_salida(salida, df, posicional_seguro=True)
        self.assertEqual(out["y"].tolist(), list(range(6)))

    def test_largo_distinto_es_error_con_mensaje_simple(self):
        df = _tabla(con_id=False)
        salida = pd.DataFrame({"Id": np.arange(4), "y": np.arange(4.0)})
        with self.assertRaises(ValueError) as cm:
            alinear_salida(salida, df)
        self.assertIn("emparejar", str(cm.exception))

    def test_ids_que_no_estan(self):
        df = _tabla()
        salida = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "y": np.arange(6.0)})
        with self.assertRaises(ValueError):
            alinear_salida(salida, df)


class TestTablero(unittest.TestCase):
    def test_no_modifica_la_tabla_original(self):
        df = _tabla()
        antes = df.copy()
        t = Tablero()
        salida = pd.DataFrame({"id": df["id"], "y": np.arange(6.0)})
        t.publicar_predicciones("ventas", df, salida, "y")
        vista = t.con_predicciones(df, "ventas")
        self.assertIn("Predicción: y", vista.columns)
        self.assertNotIn("Predicción: y", df.columns)       # la tabla del usuario queda intacta
        pd.testing.assert_frame_equal(df, antes)
        self.assertEqual(vista["Predicción: y"].tolist(), list(range(6)))

    def test_otra_tabla_no_se_toca(self):
        df = _tabla()
        t = Tablero()
        t.publicar_predicciones("ventas", df, pd.DataFrame({"id": df["id"], "y": np.arange(6.0)}), "y")
        self.assertIs(t.con_predicciones(df, "otra"), df)

    def test_nombres_de_probabilidad(self):
        df = _tabla()
        t = Tablero()
        t.publicar_predicciones("t", df, pd.DataFrame({"id": df["id"], "satisfaction": np.linspace(0, 1, 6)}),
                                "satisfaction", es_probabilidad=True, clases=["no", "si"])
        self.assertEqual(t.columnas("t"), ["Probabilidad de satisfaction = si"])
        t.publicar_predicciones("m", df, pd.DataFrame({"id": df["id"], "a": np.zeros(6), "b": np.ones(6)}),
                                "clase", es_probabilidad=True, clases=["a", "b", "c"])
        self.assertEqual(t.columnas("m"), ["Probabilidad de clase = a", "Probabilidad de clase = b"])

    def test_no_pisa_una_columna_que_ya_existe(self):
        df = _tabla()
        df["Predicción: y"] = "dato del usuario"
        t = Tablero()
        t.publicar_predicciones("t", df, pd.DataFrame({"id": df["id"], "y": np.arange(6.0)}), "y")
        vista = t.con_predicciones(df, "t")
        self.assertTrue((vista["Predicción: y"] == "dato del usuario").all())
        self.assertIn("Predicción: y (Hadar)", vista.columns)

    def test_tabla_con_filas_menos_o_mas(self):
        df = _tabla()
        t = Tablero()
        t.publicar_predicciones("t", df, pd.DataFrame({"id": df["id"], "y": np.arange(6.0)}), "y")
        recortada = df.iloc[2:]
        self.assertTrue(t.desactualizada(recortada, "t"))
        vista = t.con_predicciones(recortada, "t")
        self.assertEqual(vista["Predicción: y"].tolist(), [2, 3, 4, 5])
        mas = pd.concat([df, df.iloc[:1].rename(index={10: 99})])
        vista = t.con_predicciones(mas, "t")
        self.assertTrue(np.isnan(vista["Predicción: y"].iloc[-1]))   # la fila nueva queda vacía, no inventada
        self.assertIn("cambiaron", t.descripciones(mas, "t")["Predicción: y"])

    def test_guardar_y_reabrir(self):
        df = _tabla()
        t = Tablero()
        t.publicar_predicciones("t", df, pd.DataFrame({"id": df["id"], "y": np.arange(6.0)}), "y")
        t2 = Tablero.desde_proyecto(t.a_meta(), t.frames())
        pd.testing.assert_frame_equal(t2.con_predicciones(df, "t"), t.con_predicciones(df, "t"))
        self.assertEqual(Tablero.desde_proyecto({}, {}).tablas(), [])     # proyecto antiguo
        self.assertEqual(Tablero.desde_proyecto(None, None).tablas(), [])

    def test_limpiar(self):
        df = _tabla()
        t = Tablero()
        t.publicar_predicciones("t", df, pd.DataFrame({"id": df["id"], "y": np.arange(6.0)}), "y")
        t.limpiar()
        self.assertFalse(t.hay())
        self.assertIs(t.con_predicciones(df, "t"), df)


class TestConElMotorReal(unittest.TestCase):
    """Entrena de verdad y comprueba que cada predicción cae en SU fila."""

    @classmethod
    def setUpClass(cls):
        try:
            from hadar import prediccion as P
        except ImportError:
            import prediccion as P
        cls.P = P
        rng = np.random.default_rng(0)
        n = 1500
        d = pd.DataFrame({"cliente_id": np.arange(n), "precio": rng.normal(100, 20, n),
                          "tienda": rng.choice(list("ABCDE"), n)})
        d["ventas"] = d["precio"] * 0.5 + (d["tienda"] == "A") * 30 + rng.normal(0, 3, n)
        d["alto"] = np.where(d["ventas"] > d["ventas"].median(), "si", "no")
        cls.entrenamiento = d
        prueba = d.sample(200, random_state=1).drop(columns=["ventas", "alto"]).reset_index(drop=True)
        prueba.index = prueba.index + 5000                      # índice distinto del de entrenamiento
        cls.prueba = prueba

    def _entrenar(self, objetivo, salida="clase"):
        P = self.P
        cfg = P.ConfigPrediccion(tabla_entrenamiento="t", objetivo=objetivo, salida=salida, max_iter=60)
        sug = P.sugerir_columnas(self.entrenamiento, objetivo)
        otra = "alto" if objetivo == "ventas" else "ventas"      # la otra respuesta no es un dato de entrada
        return P.entrenar(self.entrenamiento, cfg, P.columnas_a_usar(sug, {otra: "ignorar"}),
                          progreso=lambda *_: None)

    def test_regresion(self):
        res = self._entrenar("ventas")
        salida = self.P.predecir(res, self.prueba)
        t = Tablero()
        t.publicar_predicciones("prueba", self.prueba, salida, res.objetivo, tipo_salida="valor",
                                posicional_seguro=True)
        vista = t.con_predicciones(self.prueba, "prueba")
        esperado = 0.5 * self.prueba["precio"] + (self.prueba["tienda"] == "A") * 30
        self.assertLess(float(np.abs(vista["Predicción: ventas"] - esperado).mean()), 8.0)   # cada una en su fila
        self.assertTrue(vista.index.equals(self.prueba.index))

    def test_probabilidad_binaria_apunta_a_la_clase_correcta(self):
        res = self._entrenar("alto", salida="probabilidad")
        salida = self.P.predecir(res, self.prueba, "cliente_id", "probabilidad")
        t = Tablero()
        pub = t.publicar_predicciones("prueba", self.prueba, salida, res.objetivo, es_probabilidad=True,
                                      clases=res.clases, tipo_salida=res.tipo_salida)
        col = pub.datos.columns[0]
        self.assertEqual(col, f"Probabilidad de alto = {res.clases[1]}")
        real_alto = (0.5 * self.prueba["precio"] + (self.prueba["tienda"] == "A") * 30) > 80
        # la probabilidad de "si" debe ser mayor en promedio donde de verdad es "si"
        if res.clases[1] == "si":
            self.assertGreater(pub.datos[col][real_alto].mean(), pub.datos[col][~real_alto].mean() + 0.2)


class TestProyecto(unittest.TestCase):
    """Guardar y abrir un .hadarproy con y sin predicciones."""

    def setUp(self):
        import tempfile, os
        self.dir = tempfile.mkdtemp()
        self.ruta = os.path.join(self.dir, "p.hadarproy")
        try:
            from hadar import proyecto
        except ImportError:
            import proyecto
        self.proyecto = proyecto

    def _guardar(self, **extra):
        df = _tabla()
        self.proyecto.guardar_proyecto(
            self.ruta, tablas={"ventas": df}, nombre_tabla_activa="ventas", relaciones_ontologia=None,
            filtro_columna=None, filtro_valores=[], notas_manuales={}, indicadores=[], **extra)
        return df

    def test_ida_y_vuelta_con_predicciones(self):
        df = _tabla()
        t = Tablero()
        t.publicar_predicciones("ventas", df, pd.DataFrame({"id": df["id"], "y": np.arange(6.0)}), "y")
        self._guardar(tablero_meta=t.a_meta(), tablero_tablas=t.frames())
        p = self.proyecto.abrir_proyecto(self.ruta)
        t2 = Tablero.desde_proyecto(p.tablero_meta, p.tablero_tablas)
        pd.testing.assert_frame_equal(t2.con_predicciones(p.tablas["ventas"], "ventas"),
                                      t.con_predicciones(df, "ventas"))
        self.assertEqual(t2.descripciones(df, "ventas")["Predicción: y"],
                         t.descripciones(df, "ventas")["Predicción: y"])

    def test_proyecto_sin_predicciones_abre_igual(self):
        self._guardar()
        p = self.proyecto.abrir_proyecto(self.ruta)
        self.assertEqual(p.tablero_meta, {})
        self.assertEqual(Tablero.desde_proyecto(p.tablero_meta, p.tablero_tablas).tablas(), [])

    def test_proyecto_antiguo_sin_la_clave(self):
        """Un proyecto guardado antes de existir el tablero: sin clave ni carpeta."""
        import json, zipfile
        self._guardar()
        viejo = self.ruta + ".viejo"
        with zipfile.ZipFile(self.ruta) as zin, zipfile.ZipFile(viejo, "w") as zout:
            for it in zin.infolist():
                datos = zin.read(it.filename)
                if it.filename == "proyecto.json":
                    m = json.loads(datos)
                    m.pop("tablero", None)
                    datos = json.dumps(m).encode()
                zout.writestr(it.filename, datos)
        p = self.proyecto.abrir_proyecto(viejo)
        self.assertEqual(p.tablero_tablas, {})

    def test_unas_predicciones_imposibles_de_escribir_no_impiden_guardar(self):
        malo = pd.DataFrame({"Predicción: y": [object(), object()]})     # parquet no sabe guardar esto
        self._guardar(tablero_meta={"ventas": {}}, tablero_tablas={"ventas": malo})
        p = self.proyecto.abrir_proyecto(self.ruta)
        self.assertIn("ventas", p.tablas)                  # el proyecto sí se guardó
        self.assertEqual(p.tablero_tablas, {})


class TestFirma(unittest.TestCase):
    def test_cambia_al_publicar_y_al_limpiar(self):
        df = _tabla()
        t = Tablero()
        vacia = t.firma()
        t.publicar_predicciones("t", df, pd.DataFrame({"id": df["id"], "y": np.arange(6.0)}), "y",
                                calculado_en="01/01/2026 10:00")
        con = t.firma()
        self.assertNotEqual(vacia, con)
        t.publicar_predicciones("t", df, pd.DataFrame({"id": df["id"], "y": np.arange(6.0)}), "y",
                                calculado_en="02/01/2026 10:00")
        self.assertNotEqual(con, t.firma())          # volver a predecir cuenta como cambio sin guardar
        t.limpiar()
        self.assertEqual(vacia, t.firma())


class TestConAlarmas(unittest.TestCase):
    """Una alarma de tipo Columna (la de siempre) vigila una columna predicha."""

    def test_alarma_sobre_la_prediccion(self):
        try:
            from hadar.alarmas import evaluar_columna
        except ImportError:
            from alarmas import evaluar_columna
        df = _tabla()
        t = Tablero()
        t.publicar_predicciones("t", df, pd.DataFrame({"id": df["id"], "y": [5.0, 90.0, 10.0, 95.0, 20.0, 99.0]}), "y")
        vista = t.con_predicciones(df, "t")
        regla = {"objetivo": "Predicción: y", "operador": "Mayor que (>)", "umbral": 80, "columna_id": "id"}
        r = evaluar_columna(regla, vista)
        self.assertEqual((r["revisadas"], r["cumplen"], r["extremo"]), (6, 3, 99.0))
        self.assertEqual(sorted(f["id"] for f in r["filas"]), ["101", "103", "105"])   # las filas correctas
        self.assertIsNone(evaluar_columna(regla, df))      # sin la vista, la columna no existe: no inventa nada
        # y se puede acotar con otra columna, como cualquier alarma de Columna
        r2 = evaluar_columna({**regla, "filtro_columna": "grupo", "filtro_valores": ["B"]}, vista)
        self.assertEqual(r2["cumplen"], 3)


def _resultado(titular="Se agota entre el 3 y el 9 de noviembre", nivel="alerta", confianza="media", palabra="agotamiento"):
    """Un resultado de verdad (ResultadoPronostico), no un doble."""
    try:
        from hadar.pronostico_base import ResultadoPronostico
    except ImportError:
        from pronostico_base import ResultadoPronostico
    return ResultadoPronostico(palabra=palabra, titulo=palabra.capitalize(), titular=titular, nivel=nivel,
                               confianza=confianza, por_que_confianza="Hay 40 períodos de historia.")


class TestPronosticos(unittest.TestCase):
    def test_publicar_y_ordenar_por_urgencia(self):
        t = Tablero()
        t.publicar_pronostico("a", _resultado(nivel="info", palabra="tendencia"), tabla="ventas")
        t.publicar_pronostico("b", _resultado(nivel="alerta"), tabla="ventas")
        t.publicar_pronostico("c", _resultado(nivel="atencion", palabra="umbral"), tabla="stock")
        self.assertEqual([p.clave for p in t.pronosticos()], ["b", "c", "a"])        # alerta, atención, info
        self.assertEqual([p.clave for p in t.pronosticos("ventas")], ["b", "a"])
        p = t.pronosticos()[0]
        self.assertEqual((p.titular, p.confianza, p.tabla), ("Se agota entre el 3 y el 9 de noviembre", "media", "ventas"))
        self.assertTrue(p.calculado_en)

    def test_recalcular_el_mismo_analisis_lo_reemplaza(self):
        t = Tablero()
        t.publicar_pronostico("a", _resultado(titular="uno"), tabla="v")
        t.publicar_pronostico("a", _resultado(titular="dos"), tabla="v")
        self.assertEqual([p.titular for p in t.pronosticos()], ["dos"])

    def test_tope_de_pronosticos(self):
        from_ = __import__("importlib").import_module
        try:
            tablero = from_("hadar.tablero")
        except ImportError:
            tablero = from_("tablero")
        t = Tablero()
        for i in range(tablero.MAX_PRONOSTICOS):
            t.publicar_pronostico(f"k{i}", _resultado(), tabla="v")
        with self.assertRaises(ValueError):
            t.publicar_pronostico("otro", _resultado(), tabla="v")
        t.publicar_pronostico("k0", _resultado(titular="recalculado"), tabla="v")     # reemplazar sí se puede
        t.quitar_pronostico("k1")
        t.publicar_pronostico("otro", _resultado(), tabla="v")

    def test_ida_y_vuelta_y_proyecto_antiguo(self):
        t = Tablero()
        t.publicar_pronostico("a", _resultado(), tabla="ventas", aviso_critico="Faltan datos recientes.")
        t2 = Tablero.desde_proyecto({}, {}, t.pronosticos_a_lista())
        self.assertEqual(t2.pronosticos()[0].a_dict(), t.pronosticos()[0].a_dict())
        self.assertEqual(t2.pronosticos()[0].aviso_critico, "Faltan datos recientes.")
        self.assertEqual(Tablero.desde_proyecto({}, {}, None).pronosticos(), [])
        self.assertEqual(Tablero.desde_proyecto({}, {}, [{"basura": 1}, "x", None]).pronosticos(), [])  # no revienta

    def test_datos_nuevos_los_borran(self):
        t = Tablero()
        t.publicar_pronostico("a", _resultado(), tabla="ventas")
        t.publicar_pronostico("b", _resultado(), tabla="stock")
        t.quitar("ventas")                                   # esa tabla se volvió a leer
        self.assertEqual([p.clave for p in t.pronosticos()], ["b"])
        t.limpiar()                                          # se cargaron datos nuevos
        self.assertEqual(t.pronosticos(), [])

    def test_cuenta_como_cambio_sin_guardar(self):
        t = Tablero()
        antes = t.firma()
        t.publicar_pronostico("a", _resultado(), tabla="v", calculado_en="01/01/2026 10:00")
        self.assertNotEqual(antes, t.firma())

    def test_en_el_proyecto(self):
        import tempfile, os
        try:
            from hadar import proyecto
        except ImportError:
            import proyecto
        t = Tablero()
        t.publicar_pronostico("a", _resultado(), tabla="ventas")
        ruta = os.path.join(tempfile.mkdtemp(), "p.hadarproy")
        proyecto.guardar_proyecto(ruta, tablas={"ventas": _tabla()}, nombre_tabla_activa="ventas",
                                  relaciones_ontologia=None, filtro_columna=None, filtro_valores=[],
                                  notas_manuales={}, indicadores=[], tablero_pronosticos=t.pronosticos_a_lista())
        p = proyecto.abrir_proyecto(ruta)
        t2 = Tablero.desde_proyecto(p.tablero_meta, p.tablero_tablas, p.tablero_pronosticos)
        self.assertEqual(t2.pronosticos()[0].a_dict(), t.pronosticos()[0].a_dict())


if __name__ == "__main__":
    unittest.main()