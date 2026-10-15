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


class TestRetorno(unittest.TestCase):
    """Comparar lo predicho con lo que de verdad pasó, con el motor real y respuesta conocida."""

    @classmethod
    def setUpClass(cls):
        try:
            from hadar import prediccion as P
            from hadar import retorno as R
        except ImportError:
            import prediccion as P
            import retorno as R
        cls.P, cls.R = P, R
        rng = np.random.default_rng(3)
        n = 4000
        d = pd.DataFrame({"cliente_id": np.arange(n), "precio": rng.normal(100, 20, n),
                          "tienda": rng.choice(list("ABCDE"), n)})
        d["ventas"] = d["precio"] * 0.5 + (d["tienda"] == "A") * 30 + rng.normal(0, 3, n)
        z = (d["precio"] - 100) / 15 + (d["tienda"] == "A") * 1.0
        d["alto"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "si", "no")
        cls.entrena = d.iloc[:2500].reset_index(drop=True)
        cls.futuro = d.iloc[2500:].reset_index(drop=True)             # lo que «todavía no pasó»
        cls.prueba = cls.futuro.drop(columns=["ventas", "alto"]).copy()
        cls.prueba.index = cls.prueba.index + 7000                    # índice distinto, a propósito

    def _publicar(self, objetivo, salida="clase", n_prueba=None):
        P = self.P
        otra = "alto" if objetivo == "ventas" else "ventas"
        cfg = P.ConfigPrediccion(tabla_entrenamiento="t", objetivo=objetivo, salida=salida, max_iter=60)
        sug = P.sugerir_columnas(self.entrena, objetivo)
        res = P.entrenar(self.entrena, cfg, P.columnas_a_usar(sug, {otra: "ignorar"}), progreso=lambda *_: None)
        prueba = self.prueba if n_prueba is None else self.prueba.iloc[:n_prueba]
        sal = P.predecir(res, prueba, "cliente_id", salida)
        t = Tablero()
        es_prob = res.tipo != "regresion" and salida == "probabilidad"
        pub = t.publicar_predicciones("prueba", prueba, sal, res.objetivo, es_probabilidad=es_prob,
                                      clases=res.clases, tipo_salida=sal.attrs.get("tipo_salida", ""),
                                      referencia=self.R.referencia_de(res))
        return t, pub, res

    def test_regresion_que_acierta_como_prometia(self):
        t, pub, res = self._publicar("ventas")
        self.assertEqual(pub.modo, "regresion")
        self.assertIn("mae", pub.referencia)
        real = self.futuro[["cliente_id", "ventas"]].sample(frac=1, random_state=0)     # otro orden
        r = self.R.comparar(pub, real, "ventas", "cliente_id")
        self.assertEqual((r["n_comparadas"], r["emparejadas_por"]), (1500, "id"))
        self.assertEqual(r["nivel"], "bien", r["lineas"])
        self.assertLess(r["numeros"]["veces_lo_esperado"], 1.25)

    def test_regresion_cuando_la_realidad_cambio(self):
        t, pub, res = self._publicar("ventas")
        real = self.futuro[["cliente_id", "ventas"]].copy()
        real["ventas"] = real["ventas"] * 2 + 50                          # el mundo cambió
        r = self.R.comparar(pub, real, "ventas", "cliente_id")
        self.assertEqual(r["nivel"], "alerta", r["lineas"])
        self.assertTrue(any("veces ese" in l for l in r["lineas"]))
        self.assertTrue(any("de menos" in l for l in r["lineas"]))        # predijo de menos: lo dice

    def test_probabilidad_bien_calibrada_y_mal_calibrada(self):
        t, pub, res = self._publicar("alto", salida="probabilidad")
        self.assertEqual((pub.modo, pub.clase_positiva), ("probabilidad", str(res.clases[1])))
        bueno = self.R.comparar(pub, self.futuro[["cliente_id", "alto"]], "alto", "cliente_id")
        self.assertIn(bueno["nivel"], ("bien", "atencion"), bueno["lineas"])
        self.assertIsNotNone(bueno["numeros"]["brier"])
        invertido = self.futuro[["cliente_id", "alto"]].copy()
        invertido["alto"] = invertido["alto"].map({"si": "no", "no": "si"})   # la realidad contradice al modelo
        malo = self.R.comparar(pub, invertido, "alto", "cliente_id")
        self.assertEqual(malo["nivel"], "alerta", malo["lineas"])
        self.assertTrue(any("No mejora" in l or "Ojo" in l for l in malo["lineas"]))

    def test_clase_exacta(self):
        t, pub, res = self._publicar("alto", salida="clase")
        self.assertEqual(pub.modo, "clase")
        r = self.R.comparar(pub, self.futuro[["cliente_id", "alto"]], "alto", "cliente_id")
        self.assertGreater(r["numeros"]["accuracy"], 0.6)
        self.assertNotEqual(r["nivel"], "alerta", r["lineas"])

    def test_pocos_casos_no_se_pone_nota(self):
        t, pub, res = self._publicar("ventas", n_prueba=12)
        r = self.R.comparar(pub, self.futuro[["cliente_id", "ventas"]], "ventas", "cliente_id")
        self.assertEqual(r["n_comparadas"], 12)
        self.assertEqual(r["nivel"], "info")
        self.assertTrue(any("pocos casos" in l for l in r["lineas"]))

    def test_filas_sin_valor_real_todavia(self):
        t, pub, res = self._publicar("ventas")
        real = self.futuro[["cliente_id", "ventas"]].copy()
        real.loc[real.index[:100], "ventas"] = np.nan                    # esas todavía no ocurrieron
        r = self.R.comparar(pub, real, "ventas", "cliente_id")
        self.assertEqual((r["n_comparadas"], r["n_sin_real"]), (1400, 100))
        self.assertTrue(any("todavía no tienen valor real" in l for l in r["lineas"]))

    def test_emparejar_con_cuidado(self):
        t, pub, res = self._publicar("ventas")
        real = self.futuro[["cliente_id", "ventas"]]
        # sin id: solo si son las mismas filas en el mismo orden
        r = self.R.comparar(pub, real, "ventas")
        self.assertEqual(r["emparejadas_por"], "orden")
        self.assertTrue(any("por su orden" in l for l in r["lineas"]))
        with self.assertRaises(ValueError) as cm:
            self.R.comparar(pub, real.iloc[:-5], "ventas")
        self.assertIn("mismas filas", str(cm.exception))
        # id repetido en la tabla real
        dup = real.copy()
        dup["cliente_id"] = 1
        with self.assertRaises(ValueError):
            self.R.comparar(pub, dup, "ventas", "cliente_id")
        # el id llega como texto en un lado y como número en el otro
        texto = real.assign(cliente_id=real["cliente_id"].astype(str))
        self.assertEqual(self.R.comparar(pub, texto, "ventas", "cliente_id")["n_comparadas"], 1500)
        # columna que no existe
        with self.assertRaises(ValueError):
            self.R.comparar(pub, real, "no_existe", "cliente_id")
        # ninguna fila en común
        ajeno = real.assign(cliente_id=real["cliente_id"] + 10**6)
        with self.assertRaises(ValueError):
            self.R.comparar(pub, ajeno, "ventas", "cliente_id")

    def test_predicciones_sin_id_no_se_emparejan_por_id(self):
        df = _tabla(con_id=False)
        t = Tablero()
        pub = t.publicar_predicciones("t", df, pd.DataFrame({"Id": np.arange(6), "y": np.arange(6.0)}), "y",
                                      tipo_salida="valor", posicional_seguro=True)
        self.assertIsNone(pub.ids)
        real = pd.DataFrame({"k": np.arange(6), "y": np.arange(6.0)})
        with self.assertRaises(ValueError) as cm:
            self.R.comparar(pub, real, "y", "k")
        self.assertIn("identifique cada fila", str(cm.exception))
        self.assertEqual(self.R.comparar(pub, real, "y")["emparejadas_por"], "orden")

    def test_sobrevive_al_guardar_y_abrir_el_proyecto(self):
        import tempfile, os
        try:
            from hadar import proyecto
        except ImportError:
            import proyecto
        t, pub, res = self._publicar("alto", salida="probabilidad")
        real = self.futuro[["cliente_id", "alto"]]
        antes = self.R.comparar(pub, real, "alto", "cliente_id", comparado_en="x")
        t.registrar_retorno("prueba", antes)
        ruta = os.path.join(tempfile.mkdtemp(), "p.hadarproy")
        proyecto.guardar_proyecto(ruta, tablas={"prueba": self.prueba}, nombre_tabla_activa="prueba",
                                  relaciones_ontologia=None, filtro_columna=None, filtro_valores=[],
                                  notas_manuales={}, indicadores=[], tablero_meta=t.a_meta(),
                                  tablero_tablas=t.frames(), tablero_retornos=t.retornos_a_lista())
        p = proyecto.abrir_proyecto(ruta)
        t2 = Tablero.desde_proyecto(p.tablero_meta, p.tablero_tablas, p.tablero_pronosticos, p.tablero_retornos)
        pub2 = t2.publicacion("prueba")
        self.assertEqual((pub2.modo, pub2.clase_positiva, pub2.id_nombre), (pub.modo, pub.clase_positiva, "cliente_id"))
        self.assertEqual(pub2.referencia, pub.referencia)
        despues = self.R.comparar(pub2, real, "alto", "cliente_id", comparado_en="x")
        self.assertEqual(antes, despues)                                   # mismo resultado tras reabrir
        self.assertEqual(t2.retorno("prueba"), antes)
        self.assertNotIn("__id__", t2.con_predicciones(self.prueba, "prueba").columns)   # el id no se cuela en Datos

    def test_retornos_en_el_tablero(self):
        t = Tablero()
        antes = t.firma()
        t.registrar_retorno("a", {"tabla": "a", "nivel": "bien", "comparado_en": "1"})
        t.registrar_retorno("b", {"tabla": "b", "nivel": "alerta", "comparado_en": "2"})
        self.assertNotEqual(antes, t.firma())
        self.assertEqual([d["tabla"] for d in t.retornos()], ["b", "a"])    # lo peor primero
        t.quitar("b")                                                      # esa tabla cambió
        self.assertEqual([d["tabla"] for d in t.retornos()], ["a"])
        t.limpiar()
        self.assertEqual(t.retornos(), [])
        self.assertEqual(Tablero.desde_proyecto({}, {}, None, [{"basura": 1}, "x"]).retornos(), [])


class TestFechasFuturasDelMotor(unittest.TestCase):
    """Las preguntas que responden con fechas las entregan también como dato (rango_fechas), no solo en el titular.
    Datos inventados de respuesta conocida; el motor es el de verdad."""

    @classmethod
    def setUpClass(cls):
        try:
            from hadar import pronosticos as PR
            from hadar.pronostico_base import fecha_txt, ventana_fechas
        except ImportError:
            import pronosticos as PR
            from pronostico_base import fecha_txt, ventana_fechas
        cls.PR, cls.fecha_txt, cls.ventana_fechas = PR, staticmethod(fecha_txt), staticmethod(ventana_fechas)
        rng = np.random.default_rng(1)
        n = 240
        f = pd.date_range("2026-01-01", periods=n, freq="D")
        cls.ultimo = f[-1]
        v = 100 + 0.8 * np.arange(n) + 6 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 4, n)
        cls.serie = pd.DataFrame({"fecha": f, "valor": v})
        filas = []
        for prod, (stock0, cons) in {"A": (400, 8), "B": (900, 6), "C": (300, 2)}.items():
            st = stock0
            for d in f[-60:]:
                st -= max(0, rng.normal(cons, 1.5))
                filas.append((d, prod, max(st, 0)))
        cls.stock = pd.DataFrame(filas, columns=["fecha", "producto", "stock"])
        ev = []
        for m in range(12):
            t = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 20)))
            while t < pd.Timestamp("2026-08-30"):
                ev.append((t, f"M{m}"))
                t += pd.Timedelta(days=int(rng.normal(30, 5)))
        cls.eventos = pd.DataFrame(ev, columns=["fecha", "maquina"])

    def _correr(self, palabra, df, **kw):
        return self.PR.ejecutar(self.PR.ConfigPronostico(palabra=palabra, tabla="t", **kw), {"t": df})

    def _t(self, texto):
        return pd.Timestamp(texto)

    def test_umbral_que_se_cruza(self):
        r = self._correr("umbral", self.serie, columna_fecha="fecha", columna_valor="valor", umbral=330.0,
                         agregacion="promedio")
        self.assertEqual(len(r.rango_fechas), 1)
        w = r.rango_fechas[0]
        desde, probable, hasta = self._t(w["desde"]), self._t(w["probable"]), self._t(w["hasta"])
        self.assertTrue(self.ultimo < desde <= probable <= hasta <= self._t(w["horizonte_hasta"]))
        for fecha in (desde, probable, hasta):                       # el dato dice lo mismo que el titular
            self.assertIn(self.fecha_txt(fecha), r.titular)

    def test_umbral_que_no_se_cruza_no_inventa_fechas(self):
        r = self._correr("umbral", self.serie, columna_fecha="fecha", columna_valor="valor", umbral=5000.0,
                         agregacion="promedio")
        self.assertEqual(r.rango_fechas, [])

    def test_saturacion(self):
        r = self._correr("saturacion", self.serie, columna_fecha="fecha", columna_valor="valor", capacidad=330.0,
                         agregacion="max")
        w = r.rango_fechas[0]
        self.assertIsNone(w["hasta"])                                # «desde»: no hay fecha tardía
        self.assertTrue(self.ultimo < self._t(w["desde"]) <= self._t(w["probable"]) <= self._t(w["horizonte_hasta"]))
        self.assertIn(self.fecha_txt(self._t(w["probable"])), r.titular)
        ya = self._correr("saturacion", self.serie, columna_fecha="fecha", columna_valor="valor", capacidad=100.0,
                          agregacion="max")
        self.assertEqual(ya.rango_fechas, [])                        # ya está saturado: no hay «cuándo»

    def test_agotamiento_por_elemento(self):
        r = self._correr("agotamiento", self.stock, columna_fecha="fecha", columna_stock="stock",
                         columna_entidad="producto")
        etiquetas = [w["etiqueta"] for w in r.rango_fechas]
        self.assertEqual(etiquetas[0], "A")                          # el más urgente primero
        self.assertLessEqual(len(etiquetas), 5)
        self.assertIn("«A»", r.titular)
        for w in r.rango_fechas:
            self.assertGreaterEqual(self._t(w["desde"]), self.ultimo)
            if w["hasta"]:
                self.assertTrue(self._t(w["desde"]) <= self._t(w["probable"]) <= self._t(w["hasta"]))
        desde_c = self._t(next(w for w in r.rango_fechas if w["etiqueta"] == "C")["desde"])
        desde_b = self._t(next(w for w in r.rango_fechas if w["etiqueta"] == "B")["desde"])
        self.assertLess(desde_c, desde_b)                            # C tiene menos stock y gasta poco: igual antes que B

    def test_vida_util_coincide_con_la_tabla(self):
        r = self._correr("vida_util", self.eventos, columna_fecha="fecha", columna_entidad="maquina")
        self.assertTrue(0 < len(r.rango_fechas) <= 5)
        w = r.rango_fechas[0]
        self.assertEqual(w["etiqueta"], str(r.tabla.iloc[0]["Elemento"]))
        self.assertEqual(self._t(w["probable"]), r.tabla.iloc[0]["Fecha central"])
        self.assertTrue(self._t(w["desde"]) <= self._t(w["probable"]) <= self._t(w["hasta"]))

    def test_las_demas_preguntas_siguen_sin_fechas(self):
        r = self._correr("tendencia", self.serie, columna_fecha="fecha", columna_valor="valor", agregacion="promedio")
        self.assertEqual(r.rango_fechas, [])

    def test_ventana_fechas_a_prueba_de_datos_malos(self):
        vf = self.ventana_fechas
        self.assertIsNone(vf("x", None))
        self.assertIsNone(vf("x", pd.NaT))
        self.assertIsNone(vf("x", "no es una fecha"))
        w = vf("x", pd.Timestamp("2026-10-01"), pd.NaT, "2026-10-03")
        self.assertEqual((w["desde"][:10], w["hasta"], w["probable"][:10]), ("2026-10-01", None, "2026-10-03"))
        import json
        json.dumps(w)                                                # se puede guardar en el proyecto


class TestFechasFuturasEnElTablero(unittest.TestCase):
    def _r(self, ventanas, nivel="alerta"):
        r = _resultado(nivel=nivel)
        r.rango_fechas = ventanas
        return r

    def test_se_guardan_y_se_listan_con_su_contexto(self):
        try:
            from hadar.pronostico_base import ventana_fechas
        except ImportError:
            from pronostico_base import ventana_fechas
        t = Tablero()
        t.publicar_pronostico("a", self._r([ventana_fechas("A", "2026-11-01", "2026-11-10", "2026-11-05"),
                                            ventana_fechas("B", "2026-12-01")]), tabla="stock")
        t.publicar_pronostico("b", self._r([ventana_fechas("v", "2026-12-05", None, None, "2027-01-01")], "info"),
                              tabla="ventas")
        v = t.ventanas_futuras()
        self.assertEqual([x["etiqueta"] for x in v], ["A", "B", "v"])           # la alerta primero
        self.assertEqual((v[0]["tabla"], v[0]["nivel"], v[0]["titulo"]), ("stock", "alerta", "Agotamiento"))
        self.assertEqual([x["etiqueta"] for x in t.ventanas_futuras("ventas")], ["v"])
        t.quitar_pronostico("a")
        self.assertEqual([x["etiqueta"] for x in t.ventanas_futuras()], ["v"])

    def test_sobreviven_al_proyecto_y_los_antiguos_abren_sin_ventanas(self):
        try:
            from hadar.pronostico_base import ventana_fechas
        except ImportError:
            from pronostico_base import ventana_fechas
        t = Tablero()
        t.publicar_pronostico("a", self._r([ventana_fechas("A", "2026-11-01", "2026-11-10", "2026-11-05")]), tabla="stock")
        t2 = Tablero.desde_proyecto({}, {}, t.pronosticos_a_lista())
        self.assertEqual(t2.ventanas_futuras(), t.ventanas_futuras())
        lista = t.pronosticos_a_lista()
        for d in lista:
            d.pop("ventanas")                                       # como se guardaba antes de existir
        self.assertEqual(Tablero.desde_proyecto({}, {}, lista).ventanas_futuras(), [])
        self.assertEqual(len(Tablero.desde_proyecto({}, {}, lista).pronosticos()), 1)

    def test_ventanas_malas_o_ausentes_no_rompen(self):
        t = Tablero()
        t.publicar_pronostico("a", self._r([{"etiqueta": "sin desde"}, "x", None]), tabla="t")
        self.assertEqual(t.ventanas_futuras(), [])
        sin_campo = _resultado()
        del_attr = getattr(sin_campo, "rango_fechas", None)
        t.publicar_pronostico("b", sin_campo, tabla="t")                 # un resultado que no trae el campo
        self.assertEqual(t.ventanas_futuras(), [])
        class Antiguo:                                                     # un objeto de una versión sin el campo
            palabra, titulo, titular, nivel, confianza, por_que_confianza = "umbral", "Umbral", "x", "info", "media", ""
        t.publicar_pronostico("c", Antiguo(), tabla="t")
        self.assertEqual(t.ventanas_futuras(), [])


if __name__ == "__main__":
    unittest.main()