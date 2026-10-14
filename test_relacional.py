import unittest
from tests.base import *
import relacional as R
from avisos import ordenar_avisos, IMPORTANTE

def _tablas(seed=0, n_cli=40, n_com=500, mismo_dia=True):
    rng = np.random.default_rng(seed)
    f0 = pd.Timestamp("2024-06-01")
    cli = pd.DataFrame({"id": range(1, n_cli + 1), "fecha": f0 + pd.to_timedelta(rng.integers(0, 60, n_cli), unit="D"),
                        "region": rng.choice(list("ab"), n_cli)})
    com = pd.DataFrame({"cliente_id": rng.integers(1, n_cli + 1, n_com),
                        "fecha": pd.Timestamp("2024-04-01") + pd.to_timedelta(rng.integers(0, 120, n_com), unit="D"),
                        "monto": rng.uniform(10, 100, n_com).round(2)})
    return cli, com

E = lambda: R.Enlace("clientes", "id", "compras", "cliente_id", True)

class Relacional(unittest.TestCase):
    def test_mismas_filas_en_el_mismo_orden(self):  # F4: revisión de relacional.py
        cli, com = _tablas(); cli = cli.sample(frac=1, random_state=3).reset_index(drop=True)
        a = R.construir_tabla_ampliada({"clientes": cli, "compras": com}, "clientes", [E()], 2, "fecha")
        self.assertEqual(len(a.df), len(cli)); self.assertEqual(list(a.df["id"]), list(cli["id"]))
    def test_el_resumen_usa_solo_lo_estrictamente_anterior(self):
        cli, com = _tablas()
        a = R.construir_tabla_ampliada({"clientes": cli, "compras": com}, "clientes", [E()], 2, "fecha")
        for _, f in a.df.iterrows():
            previas = com[(com.cliente_id == f["id"]) & (com.fecha < f["fecha"])]          # estrictamente antes: lo del mismo día NO entra
            esperado_cant = len(previas)
            real = f["compras (cantidad anterior)"]
            self.assertEqual(0 if pd.isna(real) else int(real), esperado_cant, f"cliente {f['id']}")
            if esperado_cant:
                self.assertAlmostEqual(f["compras.monto (suma anterior)"], previas.monto.sum(), places=6)
    def test_cambiar_el_futuro_no_cambia_el_resumen(self):
        cli, com = _tablas()
        antes = R.construir_tabla_ampliada({"clientes": cli, "compras": com}, "clientes", [E()], 2, "fecha").df
        futuro = com.copy(); idx = futuro.index[futuro.fecha >= cli.fecha.max()]
        futuro.loc[idx, "monto"] = 1e9; futuro.loc[idx, "fecha"] = futuro.loc[idx, "fecha"] + pd.Timedelta(days=500)
        despues = R.construir_tabla_ampliada({"clientes": cli, "compras": futuro}, "clientes", [E()], 2, "fecha").df
        pd.testing.assert_frame_equal(antes.fillna(-1), despues.fillna(-1))
    def test_el_objetivo_no_entra_por_otra_tabla(self):
        cli, com = _tablas(); com["churn"] = np.random.default_rng(9).integers(0, 2, len(com))
        a = R.construir_tabla_ampliada({"clientes": cli, "compras": com}, "clientes", [E()], 2, "fecha", columnas_excluir={"churn"})
        self.assertFalse([c for c in a.df.columns if "churn" in c])
        b = R.construir_tabla_ampliada({"clientes": cli, "compras": com}, "clientes", [E()], 2, "fecha")
        self.assertTrue([c for c in b.df.columns if "churn" in c])                       # sin excluir sí entraría
    def test_sin_fecha_avisa_del_riesgo_de_ver_el_futuro(self):
        cli, com = _tablas()
        a = R.construir_tabla_ampliada({"clientes": cli, "compras": com}, "clientes", [E()], 2, None)
        self.assertTrue(any("ver el futuro" in x for x in a.advertencias), a.advertencias)
        self.assertTrue(any(av.nivel == IMPORTANTE for av in ordenar_avisos(a.advertencias)))
    def test_claves_sin_pareja_se_avisan(self):
        cli, com = _tablas(); com["cliente_id"] = com["cliente_id"] + 1000              # ninguna compra apunta a un cliente
        a = R.construir_tabla_ampliada({"clientes": cli, "compras": com}, "clientes", [E()], 2, "fecha")
        self.assertTrue(any("no tiene dato en «compras»" in x for x in a.advertencias), a.advertencias)
    def test_claves_sin_pareja_tambien_en_tablas_intermedias(self):  # hallazgo nuevo
        rng = np.random.default_rng(0)
        base = pd.DataFrame({"id": range(1, 101), "producto_id": rng.integers(1, 21, 100)})
        prod = pd.DataFrame({"producto_id": range(1, 21), "proveedor_id": rng.integers(500, 600, 20), "precio": rng.uniform(1, 9, 20)})
        prov = pd.DataFrame({"proveedor_id": range(1, 11), "pais": list("abcdefghij")})       # ningún proveedor de «prod» existe aquí
        enl = [R.Enlace("base", "producto_id", "prod", "producto_id", True), R.Enlace("prod", "proveedor_id", "prov", "proveedor_id", True)]
        a = R.construir_tabla_ampliada({"base": base, "prod": prod, "prov": prov}, "base", enl, 2, None)
        self.assertTrue(any("tabla intermedia" in x and "«prov»" in x for x in a.advertencias), a.advertencias)
    def test_fechas_posteriores_en_una_union_uno_a_uno_se_avisan(self):  # hallazgo nuevo
        rng = np.random.default_rng(1); n = 200
        tick = pd.DataFrame({"id": range(n), "cliente_id": rng.integers(1, 51, n),
                             "fecha": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")})
        cli = pd.DataFrame({"cliente_id": range(1, 51), "plan": rng.choice(["a", "b"], 50),
                            "fecha_baja": pd.Timestamp("2024-09-01") + pd.to_timedelta(rng.integers(0, 90, 50), unit="D")})
        a = R.construir_tabla_ampliada({"tick": tick, "cli": cli}, "tick", [R.Enlace("tick", "cliente_id", "cli", "cliente_id", True)], 2, "fecha")
        self.assertTrue(any("«fecha_baja» de «cli» tiene fechas posteriores" in x for x in a.advertencias), a.advertencias)
        self.assertTrue(any(av.nivel == IMPORTANTE for av in ordenar_avisos(a.advertencias) if "fechas posteriores" in av))
    def test_fechas_anteriores_no_dan_falsa_alarma(self):
        rng = np.random.default_rng(1); n = 200
        tick = pd.DataFrame({"id": range(n), "cliente_id": rng.integers(1, 51, n),
                             "fecha": pd.Timestamp("2024-06-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")})
        cli = pd.DataFrame({"cliente_id": range(1, 51), "plan": rng.choice(["a", "b"], 50),
                            "fecha_alta": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 300, 50), unit="D")})
        a = R.construir_tabla_ampliada({"tick": tick, "cli": cli}, "tick", [R.Enlace("tick", "cliente_id", "cli", "cliente_id", True)], 2, "fecha")
        self.assertFalse(any("fechas posteriores" in x for x in a.advertencias), a.advertencias)
    def test_clave_casi_unica_con_repetidos_se_avisa(self):  # hallazgo nuevo
        rng = np.random.default_rng(2)
        base = pd.DataFrame({"id": range(300), "producto_id": rng.integers(1, 101, 300)})
        prod = pd.DataFrame({"producto_id": list(range(1, 101)) + [5, 6], "precio": rng.uniform(1, 9, 102)})   # 2 ids repetidos
        a = R.construir_tabla_ampliada({"base": base, "prod": prod}, "base", [R.Enlace("base", "producto_id", "prod", "producto_id", True)], 2, None)
        self.assertTrue(any("fila(s) repetida(s)" in x and "«producto_id» de «prod»" in x for x in a.advertencias), a.advertencias)
    def test_clave_unica_limpia_no_avisa_repetidos(self):
        rng = np.random.default_rng(2)
        base = pd.DataFrame({"id": range(300), "producto_id": rng.integers(1, 101, 300)})
        prod = pd.DataFrame({"producto_id": range(1, 101), "precio": rng.uniform(1, 9, 100)})
        a = R.construir_tabla_ampliada({"base": base, "prod": prod}, "base", [R.Enlace("base", "producto_id", "prod", "producto_id", True)], 2, None)
        self.assertFalse(any("repetida" in x for x in a.advertencias)); self.assertIn("prod.precio", a.df.columns)
    def test_no_se_recorren_ciclos(self):
        cli, com = _tablas()
        enl = [E(), R.Enlace("compras", "cliente_id", "clientes", "id", True)]
        a = R.construir_tabla_ampliada({"clientes": cli, "compras": com}, "clientes", enl, 3, "fecha")
        self.assertEqual(len(a.df), len(cli))
if __name__ == "__main__":
    unittest.main()
