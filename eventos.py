"""
eventos.py — Palabras clave sobre EVENTOS en el tiempo (sin Qt):

  vida útil · falla · duración · prioridad · clasificación

Idea central (sirve igual para una máquina, un cliente o un trámite):
  «¿cuánto tiempo pasa hasta que ocurre algo?»

Se cuenta el tiempo entre eventos de cada elemento. Lo más importante: los
elementos a los que todavía NO les ha pasado nada también aportan información
("duró al menos X días sin fallar"). Ignorarlos haría creer que todo falla
mucho antes de lo real. Para tratarlos bien se usa el método de Kaplan-Meier
(estándar en confiabilidad y supervivencia), hecho aquí con numpy:

  - De la curva se saca: tiempo típico, y para cada elemento con su edad actual,
    la probabilidad de que le ocurra en los próximos N días, condicionada a que
    ya sobrevivió hasta hoy.
  - El rango sale de repetir el cálculo 200 veces con elementos remuestreados.
  - Si la pregunta cae fuera de lo que los datos han visto, se marca.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from . import pistas as _pistas
    from .prediccion import (ConfigPrediccion, _a_datetime, columnas_a_usar, columnas_fecha, entrenar, es_columna_id,
                             normalizar_clave, predecir, sugerir_columnas, tipo_columna)
    from .pronostico_base import (ResultadoPronostico, confianza_por_datos, dias_txt, fecha_txt, leer_fechas, num_txt, pct_txt)
except ImportError:
    import pistas as _pistas
    from prediccion import (ConfigPrediccion, _a_datetime, columnas_a_usar, columnas_fecha, entrenar, es_columna_id,
                            normalizar_clave, predecir, sugerir_columnas, tipo_columna)
    from pronostico_base import (ResultadoPronostico, confianza_por_datos, dias_txt, fecha_txt, leer_fechas, num_txt, pct_txt)

MIN_EVENTOS = 3
MIN_EVENTOS_GRUPO = 10
N_REMUESTREOS = 200
DIA = np.timedelta64(1, "D")
_NEGATIVOS = {"", "0", "no", "false", "falso", "none", "nan", "null", "ok", "-", "n/a", "na"}


def _con_tipo(tipo_salida: str, resultado):
    resultado.tipo_salida = tipo_salida or ""
    return resultado


def _g(cfg, nombre, defecto=None):
    v = getattr(cfg, nombre, None)
    return defecto if v is None else v


# ----------------------------------------------------------------------
# Kaplan-Meier con pesos (los pesos permiten remuestrear elementos enteros)
# ----------------------------------------------------------------------
def _km_pesos(inv: np.ndarray, ev: np.ndarray, w: np.ndarray, n_u: int):
    d = np.bincount(inv, weights=w * ev, minlength=n_u)
    c = np.bincount(inv, weights=w * (~ev), minlength=n_u)
    salidas = np.cumsum(d + c)
    en_riesgo = w.sum() - np.concatenate(([0.0], salidas[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(en_riesgo > 0, d / en_riesgo, 0.0)
    return np.cumprod(1.0 - h), en_riesgo


class ModeloSupervivencia:
    """Curva S(t) = probabilidad de seguir «sin evento» pasado un tiempo t (en días).

    Más allá del punto donde quedan muy pocos elementos en observación (`t_sup`), la curva de
    Kaplan-Meier es inestable (puede caer de golpe a 0 por un solo caso). Ahí se continúa con una
    caída exponencial cuyo ritmo se toma de la segunda mitad de la zona fiable. Así un elemento
    muy viejo nunca recibe un 100% falso, y se marca como «fuera de lo observado»."""

    def __init__(self, U, S0, Sb, n_eventos: int, n_tramos: int, t_sup: float, lam_min: float):
        self.U, self.S, self.Sb = U, S0, Sb
        self.n_eventos, self.n_tramos = n_eventos, n_tramos
        self.t_sup = float(t_sup)
        self.tmax = self.t_sup
        SS = np.vstack([S0[None, :], Sb])
        self._SSe = np.concatenate([np.ones((SS.shape[0], 1)), SS], axis=1)
        i_sup, i_half = self._idx(self.t_sup), self._idx(self.t_sup / 2)
        self._n_valido = int(i_sup)
        s_sup, s_half = self._SSe[:, i_sup], self._SSe[:, i_half]
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where((s_sup > 0) & (s_half > s_sup), -np.log(s_sup / s_half) / max(self.t_sup / 2, 1e-9), lam_min)
        self._S_sup = s_sup
        self._lam = np.maximum(lam, lam_min)

    def _idx(self, t):
        return np.searchsorted(self.U, np.asarray(t, dtype=float), side="right")

    def S_at(self, t) -> np.ndarray:
        """(B+1, len(t)): fila 0 = curva estimada; filas 1.. = repeticiones con elementos remuestreados."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        base = self._SSe[:, self._idx(np.minimum(t, self.t_sup))]
        cola = self._S_sup[:, None] * np.exp(-self._lam[:, None] * np.maximum(t - self.t_sup, 0.0)[None, :])
        return np.where(t[None, :] > self.t_sup, cola, base)

    def S_en(self, t):
        return self.S_at(t)[0]

    def prob_ventana(self, edad, w):
        """P(ocurre en (edad, edad+w] | no ha ocurrido hasta `edad`), con rango 10-90%."""
        edad = np.atleast_1d(np.asarray(edad, dtype=float))
        Sa, Sw = self.S_at(edad), self.S_at(edad + w)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(Sa > 0, 1.0 - Sw / Sa, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo = np.nanpercentile(P[1:], 10, axis=0)
            hi = np.nanpercentile(P[1:], 90, axis=0)
        return P[0], lo, hi

    def plazos(self, edad, qs=(0.2, 0.5, 0.8)):
        """Días que faltan desde `edad` para que ya haya ocurrido en el q% de los casos."""
        edad = np.atleast_1d(np.asarray(edad, dtype=float))
        Sa = self.S_at(edad)[0]
        S0v = self._SSe[0, 1:1 + self._n_valido]
        s_sup, lam = float(self._S_sup[0]), float(self._lam[0])
        out = []
        for q in qs:
            objetivo = (1.0 - q) * Sa
            ii = np.searchsorted(-S0v, -objetivo, side="left")
            hallado = ii < len(S0v)
            t_h = self.U[np.minimum(ii, max(len(S0v) - 1, 0))]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_cola = np.where((objetivo > 0) & (objetivo < s_sup) & (lam > 0),
                                  self.t_sup + np.log(s_sup / objetivo) / lam, np.nan)
            t = np.where(Sa > 0, np.where(hallado, t_h, t_cola), np.nan) - edad
            out.append(np.maximum(t, 0.0))
        return out

    def cuantil_vida(self, q: float) -> float:
        S0v = self._SSe[0, 1:1 + self._n_valido]
        ii = int(np.searchsorted(-S0v, -(1.0 - q), side="left"))
        return float(self.U[ii]) if ii < len(S0v) else float("nan")

    def cuantil_vida_rango(self, q: float):
        vals = []
        for fila in self._SSe[1:, 1:1 + self._n_valido]:
            ii = int(np.searchsorted(-fila, -(1.0 - q), side="left"))
            if ii < len(fila):
                vals.append(self.U[ii])
        if len(vals) < 0.8 * (self._SSe.shape[0] - 1):
            return float("nan"), float("nan")
        return float(np.percentile(vals, 10)), float(np.percentile(vals, 90))


def ajustar_km(tiempos, eventos, ent_idx, B: int = N_REMUESTREOS, semilla: int = 0) -> ModeloSupervivencia:
    t = np.asarray(tiempos, dtype=float)
    ev = np.asarray(eventos, dtype=bool)
    U, inv = np.unique(t, return_inverse=True)
    S0, riesgo = _km_pesos(inv, ev, np.ones(len(t)), len(U))
    ents = np.unique(ent_idx)
    remap = np.searchsorted(ents, ent_idx)
    E = len(ents)
    rng = np.random.default_rng(semilla)
    Sb = np.empty((B, len(U)))
    for b in range(B):
        cnt = np.bincount(rng.integers(0, E, size=E), minlength=E).astype(float)
        Sb[b] = _km_pesos(inv, ev, cnt[remap], len(U))[0]
    minimo = max(5.0, 0.05 * len(t))
    ok = np.flatnonzero(riesgo >= minimo)
    t_sup = float(U[ok[-1]]) if len(ok) else float(U[0])
    lam_min = 0.5 / max(float(t.sum()), 1e-9)
    return ModeloSupervivencia(U, S0, Sb, int(ev.sum()), len(t), t_sup, lam_min)


# ----------------------------------------------------------------------
# Construir los «tramos» (tiempo entre eventos) desde la tabla
# ----------------------------------------------------------------------
def mascara_evento(df: pd.DataFrame, col_evento, valor_evento=None) -> np.ndarray:
    """Qué filas son un evento. Sin columna: todas las filas lo son (la tabla es un registro de eventos).
    Con valor: solo las filas donde la columna vale eso. Sin valor: números distintos de 0, verdadero,
    o texto que no sea vacío/«no»/«ok»."""
    if not col_evento:
        return np.ones(len(df), dtype=bool)
    if col_evento not in df.columns:
        raise ValueError(f"No existe la columna «{col_evento}».")
    s = df[col_evento]
    if valor_evento not in (None, ""):
        return (normalizar_clave(s) == str(valor_evento)).fillna(False).to_numpy(dtype=bool)
    if s.dtype == bool:
        return s.to_numpy(dtype=bool)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.8:
        return (num.fillna(0) != 0).to_numpy(dtype=bool)
    txt = s.astype("string").str.strip().str.lower().fillna("")
    return (~txt.isin(_NEGATIVOS)).to_numpy(dtype=bool)


@dataclass
class Tramos:
    tiempos: np.ndarray       # días
    eventos: np.ndarray       # True = el evento ocurrió; False = sigue sin ocurrir (observación cortada)
    ent_idx: np.ndarray       # a qué elemento pertenece cada tramo
    entidades: list
    edades: np.ndarray        # días desde el último evento (o desde el inicio)
    n_eventos_ent: np.ndarray
    vivos: np.ndarray         # elementos a los que todavía les puede ocurrir (en modo «único»: los que aún no)
    grupos: object            # array de grupo por elemento, o None
    ultimo_evento: list
    corte: pd.Timestamp
    avisos: list
    modo: str


def _dias(a, b) -> float:
    return float((b - a) / DIA)


def construir_tramos(df: pd.DataFrame, col_entidad, col_fecha, col_evento=None, valor_evento=None,
                     modo: str = "repetido", col_inicio=None, col_grupo=None, fecha_corte=None,
                     fin_observacion: str = "auto") -> Tramos:
    if not col_entidad or col_entidad not in df.columns:
        raise ValueError("Elige la columna que identifica a cada elemento (máquina, cliente, equipo…).")
    if not col_fecha or col_fecha not in df.columns:
        raise ValueError("Elige la columna de fecha.")
    dt = leer_fechas(df[col_fecha], col_fecha)
    ent = normalizar_clave(df[col_entidad])
    ev = mascara_evento(df, col_evento, valor_evento)
    ok = dt.notna().to_numpy() & ent.notna().to_numpy()
    if not ok.any():
        raise ValueError("No pude leer fechas o identificadores válidos.")
    t = pd.DataFrame({"ent": ent.astype(object).to_numpy()[ok], "t": dt.to_numpy()[ok], "ev": ev[ok]})
    corte = pd.Timestamp(fecha_corte) if fecha_corte is not None else pd.Timestamp(t["t"].max())
    avisos = []
    if col_inicio and col_inicio in df.columns:
        di = leer_fechas(df[col_inicio], col_inicio, exigir_validas=False)
        ini = pd.DataFrame({"ent": ent.astype(object).to_numpy(), "i": di.to_numpy()}).dropna()
        ini = ini.groupby("ent")["i"].min()
    else:
        ini = t.groupby("ent")["t"].min()
        if modo == "repetido":
            avisos.append("No hay fecha de inicio de cada elemento: el tiempo hasta el PRIMER evento no se usó "
                          "(no se sabe cuánto llevaba funcionando antes de empezar el registro).")
    # Fin de la observación de cada elemento. «global»: todos se observan hasta la fecha de corte (si un elemento se dio
    # de baja, cuenta como «sigue sin fallar» e infla la vida útil). «ultimo_registro»: cada uno hasta su último
    # registro. «auto» usa el segundo solo si la tabla trae filas que NO son eventos (lecturas, estados): ahí el
    # último registro sí indica hasta cuándo se le siguió.
    ultimo_reg = t.groupby("ent")["t"].max()
    hay_no_eventos = bool((~t["ev"]).any())
    usar_ultimo = fin_observacion == "ultimo_registro" or (fin_observacion == "auto" and hay_no_eventos)
    evs = {k: np.sort(np.unique(g["t"].to_numpy())) for k, g in t[t["ev"]].groupby("ent")}
    grupos = None
    if col_grupo and col_grupo in df.columns:
        g = pd.DataFrame({"ent": ent.astype(object).to_numpy(), "g": normalizar_clave(df[col_grupo]).astype(object).to_numpy()}).dropna()
        gmap = g.groupby("ent")["g"].first()
    else:
        gmap = None
    entidades = list(ini.index)
    tiempos, eventos, idx = [], [], []
    edades = np.zeros(len(entidades))
    n_ev = np.zeros(len(entidades), dtype=int)
    vivos = np.ones(len(entidades), dtype=bool)
    ult = []
    ceros = 0
    for i, e in enumerate(entidades):
        s0 = ini[e]
        corte_e = min(corte, pd.Timestamp(ultimo_reg[e])) if usar_ultimo and e in ultimo_reg.index else corte
        v = evs.get(e, np.array([], dtype="datetime64[ns]"))
        v = v[v <= np.datetime64(corte)]
        n_ev[i] = len(v)
        ult.append(pd.Timestamp(v[-1]) if len(v) else pd.NaT)
        if modo == "unico":
            if len(v):
                d = _dias(np.datetime64(s0), v[0])
                if d > 0:
                    tiempos.append(d), eventos.append(True), idx.append(i)
                else:
                    ceros += 1
                vivos[i] = False
                edades[i] = max(d, 0.0)
            else:
                d = _dias(np.datetime64(s0), np.datetime64(corte_e))
                if d > 0:
                    tiempos.append(d), eventos.append(False), idx.append(i)
                edades[i] = max(d, 0.0)
        else:
            if col_inicio and len(v):
                d = _dias(np.datetime64(s0), v[0])
                if d > 0:
                    tiempos.append(d), eventos.append(True), idx.append(i)
            for a, b in zip(v[:-1], v[1:]):
                d = _dias(a, b)
                if d > 0:
                    tiempos.append(d), eventos.append(True), idx.append(i)
                else:
                    ceros += 1
            ref = v[-1] if len(v) else np.datetime64(s0)
            d = _dias(ref, np.datetime64(corte_e))
            if d > 0:
                tiempos.append(d), eventos.append(False), idx.append(i)
            edades[i] = max(d, 0.0)
    if col_inicio and col_inicio in df.columns:
        sin_inicio = [e for e in pd.unique(t["ent"]) if e not in ini.index]
        if sin_inicio:
            avisos.append(f"{len(sin_inicio)} elemento(s) no tienen una fecha de inicio válida y no se analizaron: "
                          + ", ".join(str(x) for x in sin_inicio[:5]) + (" y otros." if len(sin_inicio) > 5 else "."))
    if usar_ultimo:
        avisos.append("Cada elemento se observó hasta su último registro (no hasta la fecha más reciente de la tabla): "
                      "así los que se dieron de baja no cuentan como «siguen sin fallar».")
    else:
        ev_t = np.array(tiempos)[np.array(eventos, dtype=bool)] if len(tiempos) else np.array([])
        if len(ev_t) >= 5 and len(entidades):
            inactivos = int((edades[vivos] > 2 * float(np.median(ev_t))).sum()) if vivos.any() else 0
            if inactivos:
                avisos.append(f"{inactivos} elemento(s) llevan más del doble del tiempo típico sin ningún registro. La tabla "
                              "solo trae eventos, así que no se sabe si siguen activos o se dieron de baja: si algunos se "
                              "dieron de baja, la vida útil estimada sale más larga de lo real.")
    if ceros:
        avisos.append(f"{ceros} eventos ocurrieron el mismo día/instante que el anterior y no se contaron como intervalos.")
    if gmap is not None:
        grupos = np.array([gmap.get(e, None) for e in entidades], dtype=object)
    tiempos, eventos, idx = np.array(tiempos, dtype=float), np.array(eventos, dtype=bool), np.array(idx, dtype=int)
    return Tramos(tiempos, eventos, idx, entidades, edades, n_ev, vivos, grupos, ult, corte, avisos, modo)


# ----------------------------------------------------------------------
# Estimación por grupos y tabla por elemento
# ----------------------------------------------------------------------
def _modelos(tr: Tramos, semilla: int = 0):
    if int(tr.eventos.sum()) < MIN_EVENTOS:
        raise ValueError(f"Hay solo {int(tr.eventos.sum())} intervalos completos entre eventos; se necesitan al "
                         f"menos {MIN_EVENTOS} para estimar tiempos. Con más historial (o más elementos) se podrá.")
    modelos = {None: ajustar_km(tr.tiempos, tr.eventos, tr.ent_idx, semilla=semilla)}
    if tr.grupos is not None:
        g_tramo = tr.grupos[tr.ent_idx]
        for g in {x for x in tr.grupos if x is not None}:
            sel = np.array([x == g for x in g_tramo])
            if sel.sum() >= 2 * MIN_EVENTOS_GRUPO and int(tr.eventos[sel].sum()) >= MIN_EVENTOS_GRUPO:
                modelos[g] = ajustar_km(tr.tiempos[sel], tr.eventos[sel], tr.ent_idx[sel], semilla=semilla)
    return modelos


def _tabla_elementos(tr: Tramos, modelos: dict, w: float) -> pd.DataFrame:
    sel = np.flatnonzero(tr.vivos)
    n = len(sel)
    if n == 0:
        return pd.DataFrame()
    claves = []
    for i in sel:
        g = tr.grupos[i] if tr.grupos is not None else None
        claves.append(g if g in modelos else None)
    p0, lo, hi = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    q20, q50, q80 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    extrap = np.zeros(n, dtype=bool)
    edades = tr.edades[sel]
    for clave in set(claves):
        pos = np.array([k for k, c in enumerate(claves) if c == clave])
        m = modelos[clave]
        a, b, c = m.prob_ventana(edades[pos], w)
        p0[pos], lo[pos], hi[pos] = a, b, c
        t20, t50, t80 = m.plazos(edades[pos])
        q20[pos], q50[pos], q80[pos] = t20, t50, t80
        extrap[pos] = (edades[pos] + w) > m.tmax
    corte = tr.corte
    fecha_c = [(corte + pd.Timedelta(days=float(x))).normalize() if np.isfinite(x) else pd.NaT for x in q50]
    out = pd.DataFrame({
        "Elemento": [tr.entidades[i] for i in sel],
        "Días desde el último evento" if tr.modo == "repetido" else "Días de vida": edades,
        "Eventos registrados": tr.n_eventos_ent[sel],
        f"Probabilidad en {num_txt(w)} días": p0,
        "Probabilidad (mínimo probable)": lo,
        "Probabilidad (máximo probable)": hi,
        "Plazo pronto (días)": q20, "Plazo central (días)": q50, "Plazo tarde (días)": q80,
        "Fecha central": fecha_c,
        "Fuera de lo observado": extrap,
    })
    if tr.grupos is not None:
        out.insert(1, "Grupo", [tr.grupos[i] for i in sel])
    return out


def _resumen_vida(modelo: ModeloSupervivencia, modo: str):
    med = modelo.cuantil_vida(0.5)
    lo, hi = modelo.cuantil_vida_rango(0.5)
    q25, q75 = modelo.cuantil_vida(0.25), modelo.cuantil_vida(0.75)
    return med, lo, hi, q25, q75


def _conf_eventos(n_ev: int):
    c = confianza_por_datos(n_ev, 10, 30)
    motivo = {"baja": f"hay solo {n_ev} eventos completos: menos de 10 da estimaciones muy inestables.",
              "media": f"hay {n_ev} eventos completos; con 30 o más las estimaciones se vuelven sólidas.",
              "alta": f"hay {n_ev} eventos completos, suficientes para estimar con estabilidad."}[c]
    return c, motivo


def _supuestos_eventos(modo: str, con_grupo: bool):
    s = ["Los elementos se comportan parecido a los del historial" + (" de su mismo grupo." if con_grupo else ".")]
    if modo == "repetido":
        s.append("Tras cada evento el elemento vuelve a empezar de cero (por ejemplo, queda reparado).")
    return s


def _analisis_supervivencia(df, cfg, w_defecto=30.0):
    tr = construir_tramos(df, _g(cfg, "columna_entidad"), _g(cfg, "columna_fecha"), _g(cfg, "columna_evento"),
                          _g(cfg, "valor_evento"), _g(cfg, "modo_evento", "repetido"), _g(cfg, "columna_inicio"),
                          _g(cfg, "columna_grupo"), _g(cfg, "fecha_corte"), _g(cfg, "fin_observacion", "auto"))
    modelos = _modelos(tr, int(_g(cfg, "semilla", 0)))
    w = float(_g(cfg, "ventana_dias", w_defecto))
    tabla = _tabla_elementos(tr, modelos, w)
    return tr, modelos, tabla, w


def _avisos_eventos(tr: Tramos, modelos: dict, tabla: pd.DataFrame) -> list:
    av = list(tr.avisos)
    if len(tabla) and tabla["Fuera de lo observado"].any():
        k = int(tabla["Fuera de lo observado"].sum())
        av.append(f"En {k} elemento(s) la pregunta llega más lejos que el tiempo más largo visto en los datos: "
                  f"la probabilidad se calculó extendiendo lo observado y puede estar subestimada.")
    if tr.grupos is not None and len(modelos) == 1:
        av.append("Ningún grupo tiene suficientes eventos (mínimo 10) para estimarse por separado: se usó el promedio de todos.")
    elif tr.grupos is not None:
        sin = {g for g in tr.grupos if g is not None and g not in modelos}
        if sin:
            av.append(f"Los grupos {', '.join(map(str, sorted(map(str, sin))))} tienen pocos eventos: se les aplicó el promedio general.")
    return av


def analizar_vida_util(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    tr, modelos, tabla, w = _analisis_supervivencia(df, cfg)
    glob = modelos[None]
    med, lo, hi, q25, q75 = _resumen_vida(glob, tr.modo)
    unidad = "entre un evento y el siguiente pasan" if tr.modo == "repetido" else "un elemento tarda"
    if np.isfinite(med):
        base = (f"Típicamente {unidad} {dias_txt(med)} (la mitad de los casos, entre {dias_txt(q25)} y {dias_txt(q75)})"
                if np.isfinite(q25) and np.isfinite(q75) else f"Típicamente {unidad} {dias_txt(med)}")
    else:
        base = (f"Menos de la mitad de los casos llegó a tener el evento dentro de lo observado "
                f"({dias_txt(glob.tmax)}): el tiempo típico es mayor que eso")
    tabla = tabla.sort_values("Plazo central (días)", kind="stable", na_position="last").reset_index(drop=True) if len(tabla) else tabla
    conf, motivo = _conf_eventos(glob.n_eventos)
    avisos = _avisos_eventos(tr, modelos, tabla)
    tarj = [(dias_txt(med), "Tiempo típico", f"mediana; rango probable {dias_txt(lo)} a {dias_txt(hi)}"),
            (str(glob.n_eventos), "Eventos usados", f"intervalos completos entre eventos, de {len(tr.entidades)} elementos"),
            (str(int(tr.vivos.sum())), "Elementos en seguimiento", "con una estimación individual en la tabla")]
    if len(tabla) and np.isfinite(tabla.iloc[0]["Plazo central (días)"]):
        r = tabla.iloc[0]
        pr, ta = r["Plazo pronto (días)"], r["Plazo tarde (días)"]
        rango = (f"en {dias_txt(pr)} a {dias_txt(ta)}" if np.isfinite(pr) and np.isfinite(ta)
                 else f"en {dias_txt(pr)} o más" if np.isfinite(pr) else "más adelante")
        titular = (f"{base}. El más cercano a ocurrir es «{r['Elemento']}»: probablemente {rango} desde hoy "
                   f"(lo más probable: {dias_txt(r['Plazo central (días)'])}, ≈{fecha_txt(r['Fecha central'])}).")
        nivel = "alerta" if (tabla["Plazo central (días)"] <= w).any() else "atencion" if (tabla["Plazo pronto (días)"] <= w).any() else "info"
    else:
        titular = base + "."
        nivel = "info"
    det = ["El tiempo se mide entre eventos consecutivos de cada elemento; los que aún no tienen evento aportan que "
           "«duraron al menos» lo observado (así no se subestima la vida real).",
           f"El rango es del 10% al 90% de {glob.Sb.shape[0]} repeticiones del cálculo con elementos remuestreados.",
           "«Pronto/central/tarde» = en el 20% / 50% / 80% de los casos parecidos ya habría ocurrido."]
    return ResultadoPronostico(
        "vida_util", "Vida útil", titular, nivel, conf, motivo, tarj, det,
        _supuestos_eventos(tr.modo, tr.grupos is not None), avisos, tabla, "Cuánto falta, elemento por elemento",
        None, {"mediana_dias": med, "eventos": glob.n_eventos, "tramos": glob.n_tramos})


def analizar_falla(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    if str(_g(cfg, "metodo_evento", "estadistico")) == "modelo":
        return analizar_falla_modelo(df, cfg)
    tr, modelos, tabla, w = _analisis_supervivencia(df, cfg)
    glob = modelos[None]
    col_p = f"Probabilidad en {num_txt(w)} días"
    if len(tabla) == 0:
        raise ValueError("No hay elementos en seguimiento (todos tuvieron ya el evento).")
    tabla = tabla.sort_values(col_p, ascending=False, kind="stable", na_position="last").reset_index(drop=True)
    umbral = 0.3
    n_riesgo = int((tabla[col_p] >= umbral).sum())
    n_alto = int((tabla[col_p] >= 0.5).sum())
    r = tabla.iloc[0]
    conf, motivo = _conf_eventos(glob.n_eventos)
    avisos = _avisos_eventos(tr, modelos, tabla)
    probs_all = tabla[col_p].to_numpy()
    esperados = float(np.nansum(probs_all))
    titular = (f"En los próximos {num_txt(w)} días se esperan unos {num_txt(esperados, 1)} eventos entre {len(tabla)} elementos. "
               f"El mayor riesgo es «{r['Elemento']}»: {pct_txt(r[col_p])} "
               f"(entre {pct_txt(r['Probabilidad (mínimo probable)'])} y {pct_txt(r['Probabilidad (máximo probable)'])})"
               + (f"; {n_alto} superan el 50%." if n_alto else "."))
    nivel = "alerta" if n_alto else "atencion" if n_riesgo else "info"
    tarj = [(str(n_riesgo), f"Con ≥{umbral * 100:.0f}% de riesgo", f"de {len(tabla)} elementos, en {num_txt(w)} días"),
            (num_txt(esperados, 1), "Eventos esperados", f"suma de probabilidades en {num_txt(w)} días"),
            (pct_txt(r[col_p]), "Mayor riesgo", str(r["Elemento"]))]
    det = ["La probabilidad de cada elemento depende de cuánto tiempo lleva sin evento: si ya sobrevivió mucho, "
           "se calcula la chance de que ocurra a partir de ahora (no desde cero).",
           f"El rango sale de {glob.Sb.shape[0]} repeticiones del cálculo con elementos remuestreados.",
           "Este método solo mira cuándo ocurren los eventos; no usa mediciones ni condiciones de uso. "
           "Si tienes esos datos, el método «modelo» puede distinguir mejor cuáles se acercan a fallar."]
    return ResultadoPronostico(
        "falla", "Falla / evento", titular, nivel, conf, motivo, tarj, det,
        _supuestos_eventos(tr.modo, tr.grupos is not None), avisos, tabla,
        f"Probabilidad de que ocurra en {num_txt(w)} días", None,
        {"en_riesgo": n_riesgo, "eventos": glob.n_eventos, "esperados": esperados})


# ----------------------------------------------------------------------
# FALLA con modelo (usa condiciones: sensores, uso, tipo…) — reutiliza el motor de Predicción
# ----------------------------------------------------------------------
def preparar_dataset_evento(df: pd.DataFrame, col_entidad, col_fecha, mask_evento: np.ndarray, ventana_dias: float,
                            fecha_corte=None):
    """Agrega a cada fila: ¿ocurre un evento del mismo elemento en los próximos N días?, días desde el último
    evento anterior y cuántos eventos hubo antes (ambos mirando SOLO lo anterior a esa fila)."""
    d = df.reset_index(drop=True).copy()
    dt = leer_fechas(d[col_fecha], col_fecha)
    ent = normalizar_clave(d[col_entidad])
    ok = (dt.notna() & ent.notna()).to_numpy()
    d, dt, ent, mask_evento = d[ok].reset_index(drop=True), dt[ok].reset_index(drop=True), ent[ok].reset_index(drop=True), mask_evento[ok]
    t_ns = dt.to_numpy("datetime64[ns]").astype("int64")
    corte = pd.Timestamp(fecha_corte) if fecha_corte is not None else pd.Timestamp(dt.max())
    corte_ns = int(np.datetime64(corte, "ns").astype("int64"))
    w_ns = int(ventana_dias * 86400e9)
    n = len(d)
    y = np.zeros(n, dtype=bool)
    desde = np.full(n, np.nan)
    previos = np.zeros(n, dtype=int)
    claves = ent.astype(object).to_numpy()
    filas_por_ent = pd.Series(np.arange(n)).groupby(claves).indices
    ev_por_ent = pd.Series(t_ns[mask_evento]).groupby(claves[mask_evento]).apply(lambda s: np.sort(np.unique(s.to_numpy()))).to_dict()
    for e, filas in filas_por_ent.items():
        ev = ev_por_ent.get(e)
        if ev is None or len(ev) == 0:
            continue
        t = t_ns[filas]
        nxt = np.searchsorted(ev, t, side="right")
        hay = nxt < len(ev)
        sig = ev[np.minimum(nxt, len(ev) - 1)]
        y[filas] = hay & ((sig - t) <= w_ns)
        prv = np.searchsorted(ev, t, side="left") - 1
        hp = prv >= 0
        desde[filas] = np.where(hp, (t - ev[np.maximum(prv, 0)]) / 86400e9, np.nan)
        previos[filas] = prv + 1
    nombre = f"Ocurre en los próximos {num_txt(ventana_dias)} días"
    d["Días desde el último evento"] = desde
    d["Eventos anteriores"] = previos
    d[nombre] = np.where(y, "sí", "no")
    d["__observable__"] = (t_ns + w_ns) <= corte_ns
    return d, nombre


def analizar_falla_modelo(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    col_ent, col_fecha = _g(cfg, "columna_entidad"), _g(cfg, "columna_fecha")
    if not col_ent or not col_fecha:
        raise ValueError("Elige la columna del elemento y la de fecha.")
    w = float(_g(cfg, "ventana_dias", 30))
    mask = mascara_evento(df, _g(cfg, "columna_evento"), _g(cfg, "valor_evento"))
    if not mask.any():
        raise ValueError("No encontré filas marcadas como evento: revisa la columna y el valor del evento.")
    d, objetivo = preparar_dataset_evento(df, col_ent, col_fecha, mask, w, _g(cfg, "fecha_corte"))
    entrenable = d[d["__observable__"]].drop(columns="__observable__").reset_index(drop=True)
    if len(entrenable) < 100:
        raise ValueError(f"Solo hay {len(entrenable)} filas con el futuro ya conocido ({num_txt(w)} días); "
                         f"se necesitan al menos 100. Prueba con una ventana más corta o más historial.")
    excluir = {col_ent, objetivo, "__observable__"}
    if _g(cfg, "columna_evento"):
        excluir.add(_g(cfg, "columna_evento"))
    extra = list(_g(cfg, "columnas_explicativas", []) or [])
    sug = sugerir_columnas(entrenable, objetivo)
    manual = {c: True for c in extra}
    cols = [c for c in columnas_a_usar(sug, manual if extra else {}) if c not in excluir]
    if extra:
        cols = [c for c in extra if c in entrenable.columns and c not in excluir]
    if not cols:
        raise ValueError("No quedaron columnas para predecir. Elige columnas explicativas (mediciones, uso, tipo…).")
    cfgp = ConfigPrediccion(objetivo=objetivo, validacion="temporal", columna_fecha=col_fecha, salida="probabilidad")
    res = entrenar(entrenable, cfgp, cols)
    ultimo = d.sort_values(col_fecha, kind="stable").groupby(col_ent, sort=False).tail(1).drop(columns="__observable__")
    pred = predecir(res, ultimo, columna_id=col_ent)
    pred = pred.sort_values(pred.columns[1], ascending=False, kind="stable").reset_index(drop=True)
    es_prob = res.tipo_salida == "probabilidad"
    pred.columns = ["Elemento", f"Probabilidad en {num_txt(w)} días" if es_prob else "Puntaje de riesgo (0 a 1)"] + list(pred.columns[2:])
    cp = pred.columns[1]
    nivel_v, titulo_v, frase_v = res.veredicto()
    top = pred.iloc[0] if len(pred) else None
    if not es_prob:
        titular = (f"El modelo ordena los {len(pred)} elementos por riesgo de que ocurra en {num_txt(w)} días; "
                   f"el primero es «{top['Elemento']}». El número sirve para ORDENAR, no es un porcentaje real.")
    else:
        titular = (f"El modelo estima que «{top['Elemento']}» tiene {pct_txt(top[cp])} de probabilidad de que ocurra en "
                   f"{num_txt(w)} días ({int((pred[cp] >= 0.3).sum())} de {len(pred)} superan 30%).")
    conf = {"muy_bueno": "alta", "bueno": "alta", "regular": "media"}.get(str(nivel_v), "baja")
    avisos = list(res.advertencias)
    if nivel_v == "sospechoso":
        avisos.insert(0, frase_v)
    return _con_tipo(res.tipo_salida, ResultadoPronostico(
        "falla", "Falla / evento (con modelo)", titular, "atencion" if es_prob and len(pred) and float(top[cp]) >= 0.5 else "info",
        conf, f"{titulo_v}. {frase_v} (probado entrenando con lo antiguo y evaluando con lo más reciente).",
        res.tarjetas(), res.lineas_resumen(),
        ["Las condiciones recientes de cada elemento seguirán pareciéndose a las que tenían antes de fallar los demás."],
        avisos, pred, "Riesgo actual de cada elemento", None,
        {"metricas": dict(res.metricas), "mejora": float(res.mejora())}))


# ----------------------------------------------------------------------
# DURACIÓN (cuánto tarda un trámite, entrega, reparación…)
# ----------------------------------------------------------------------
_FACTOR_UNIDAD = {"dias": 1.0, "horas": 1 / 24, "minutos": 1 / 1440, "semanas": 7.0}


def analizar_duracion(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    col_ini, col_fin, col_val = _g(cfg, "columna_inicio"), _g(cfg, "columna_fin"), _g(cfg, "columna_valor")
    col_g, col_id = _g(cfg, "columna_grupo"), _g(cfg, "columna_entidad")
    w = float(_g(cfg, "ventana_dias", 7))
    avisos = []
    if col_ini and col_fin:
        ini, fin = leer_fechas(df[col_ini], col_ini), leer_fechas(df[col_fin], col_fin, exigir_validas=False)
        buenos = ini.notna().to_numpy()
        corte = pd.Timestamp(_g(cfg, "fecha_corte") or max(ini.max(), fin.max() if fin.notna().any() else ini.max()))
        dur = ((fin - ini) / pd.Timedelta(days=1)).to_numpy(dtype="float64")
        abierto = (fin.isna() & ini.notna()).to_numpy()
        dur = np.where(abierto, (corte - ini) / pd.Timedelta(days=1), dur).astype("float64")
        invalido = ~np.isfinite(dur) | (dur < 0) | ~buenos
        if invalido.any():
            avisos.append(f"{int(invalido.sum())} filas con fechas faltantes o con el fin antes del inicio no se usaron.")
        solo_fecha = bool((ini[buenos] == ini[buenos].dt.normalize()).all()
                          and (fin.dropna() == fin.dropna().dt.normalize()).all())
        medio = 0.5 if solo_fecha else None      # «0 días» con fechas sin hora = menos de un día
    elif col_val:
        f = _FACTOR_UNIDAD.get(str(_g(cfg, "unidad_duracion", "dias")), 1.0)
        dur = pd.to_numeric(df[col_val], errors="coerce").to_numpy(dtype="float64") * f
        abierto = np.zeros(len(df), dtype=bool)
        invalido = ~np.isfinite(dur) | (dur < 0)
        corte = pd.Timestamp.now().normalize()
        medio = 0.5 * f                           # «0» en la unidad de la columna = menos de una unidad
    else:
        raise ValueError("Elige las columnas de inicio y fin, o una columna con la duración.")
    cero_cerrado = ~invalido & ~abierto & (dur == 0)
    if cero_cerrado.any():
        n0 = int(cero_cerrado.sum())
        if medio is not None and _g(cfg, "ceros_mismo_dia", "medio_dia") != "descartar":
            dur = np.where(cero_cerrado, medio, dur)
            avisos.append(f"{n0} casos ({n0 / max(int((~invalido).sum()), 1) * 100:.0f}%) terminaron el mismo día (0 días): "
                          f"se contaron como medio día porque no hay hora. Descartarlos habría alargado el tiempo típico.")
        elif medio is not None:
            avisos.append(f"{n0} casos terminaron el mismo día (0 días) y no se usaron, como pediste en Ajustes: el tiempo "
                          f"típico que se ve corresponde solo a los casos que tardaron más de un día.")
        else:
            avisos.append(f"{n0} casos con duración exactamente 0 no se usaron (con hora, 0 no es una duración medible).")
    n_abiertos_hoy = int((~invalido & abierto & (dur <= 0)).sum())
    if n_abiertos_hoy:
        avisos.append(f"{n_abiertos_hoy} casos abiertos se iniciaron en la fecha de corte; todavía no aportan información y no se usaron.")
    ok = ~invalido & (dur > 0)
    if ok.sum() < 10:
        raise ValueError(f"Solo hay {int(ok.sum())} casos con duración válida; se necesitan al menos 10.")
    t = dur[ok]
    ev = ~abierto[ok]
    if int(ev.sum()) < 5:
        raise ValueError("Hay muy pocos casos terminados para estimar cuánto tarda (mínimo 5).")
    idx = np.arange(len(t))
    grupos = None
    if col_g and col_g in df.columns:
        grupos = normalizar_clave(df[col_g]).astype(object).to_numpy()[ok]
    mod = ajustar_km(t, ev, idx, semilla=int(_g(cfg, "semilla", 0)))
    med = mod.cuantil_vida(0.5)
    lo_m, hi_m = mod.cuantil_vida_rango(0.5)
    p10, p90 = mod.cuantil_vida(0.10), mod.cuantil_vida(0.90)
    n_ab = int((~ev).sum())
    # tabla por grupo
    filas_g = []
    if grupos is not None:
        for g in sorted({x for x in grupos if x is not None}, key=str):
            sel = np.array([x == g for x in grupos])
            if int(ev[sel].sum()) >= MIN_EVENTOS_GRUPO:
                mg = ajustar_km(t[sel], ev[sel], idx[sel], B=100, semilla=int(_g(cfg, "semilla", 0)))
                filas_g.append({"Grupo": g, "Casos": int(sel.sum()), "Terminados": int(ev[sel].sum()),
                                "Tiempo típico (días)": mg.cuantil_vida(0.5),
                                "Rápido (10%)": mg.cuantil_vida(0.10), "Lento (90%)": mg.cuantil_vida(0.90)})
            else:
                filas_g.append({"Grupo": g, "Casos": int(sel.sum()), "Terminados": int(ev[sel].sum()),
                                "Tiempo típico (días)": np.nan, "Rápido (10%)": np.nan, "Lento (90%)": np.nan})
    tabla = pd.DataFrame(filas_g) if filas_g else None
    # casos abiertos
    detalle_abiertos = None
    n_atras = 0
    if n_ab:
        pos = np.flatnonzero(ok)[~ev]
        edad = t[~ev]
        F = 1.0 - mod.S_en(edad)
        p0, lo, hi = mod.prob_ventana(edad, w)
        q50 = mod.plazos(edad, (0.5,))[0]
        etiquetas = (normalizar_clave(df[col_id]).astype(object).to_numpy()[pos] if col_id and col_id in df.columns
                     else np.array([f"fila {i + 1}" for i in pos], dtype=object))
        detalle_abiertos = pd.DataFrame({
            "Caso": etiquetas, "Lleva (días)": edad, "Ya pasó más que el % de los casos": F,
            f"Prob. de terminar en {num_txt(w)} días": p0, "Faltan, lo más probable (días)": q50})
        if grupos is not None:
            detalle_abiertos.insert(1, "Grupo", grupos[~ev])
        detalle_abiertos = detalle_abiertos.sort_values("Ya pasó más que el % de los casos", ascending=False, kind="stable").reset_index(drop=True)
        n_atras = int((F >= 0.9).sum())
        if tabla is None:
            tabla = detalle_abiertos
    conf, motivo = _conf_eventos(int(ev.sum()))
    texto_t = (f"Un caso típico tarda {dias_txt(med)} (el 80% de los casos entre {dias_txt(p10)} y {dias_txt(p90)})"
               if np.isfinite(med) and np.isfinite(p10) and np.isfinite(p90)
               else f"Un caso típico tarda {dias_txt(med)}" if np.isfinite(med) else "La mayoría de los casos tarda más de lo observado")
    titular = texto_t + (f". Hay {n_ab} casos abiertos y {n_atras} ya llevan más tiempo que el 90% de los casos." if n_ab else ".")
    nivel = "atencion" if n_atras else "info"
    tarj = [(dias_txt(med), "Tiempo típico", f"mediana; rango probable {dias_txt(lo_m)} a {dias_txt(hi_m)}"),
            (f"{dias_txt(p10)} – {dias_txt(p90)}", "Rango habitual", "del 10% más rápido al 10% más lento"),
            (str(n_atras), "Casos atrasados", "abiertos y más lentos que el 90% de los casos terminados")]
    det = ["Los casos que siguen abiertos se tratan como «tardan al menos lo que llevan»: así no se subestima el tiempo real."]
    return ResultadoPronostico(
        "duracion", "Duración", titular, nivel, conf, motivo, tarj, det,
        ["Los casos nuevos se parecen a los del historial (mismo proceso, misma carga de trabajo)."], avisos,
        tabla, "Por grupo" if filas_g else "Casos abiertos", None,
        {"mediana_dias": med, "terminados": int(ev.sum()), "abiertos": n_ab, "detalle_abiertos": detalle_abiertos})


# ----------------------------------------------------------------------
# PRIORIDAD
# ----------------------------------------------------------------------
def analizar_prioridad(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    col_r, col_i, col_id = _g(cfg, "columna_riesgo"), _g(cfg, "columna_impacto"), _g(cfg, "columna_entidad")
    if not col_r or col_r not in df.columns:
        raise ValueError("Elige la columna con el riesgo (por ejemplo, una probabilidad de falla o de abandono).")
    riesgo = pd.to_numeric(df[col_r], errors="coerce").to_numpy(dtype="float64")
    ok = np.isfinite(riesgo) & (riesgo >= 0)
    if col_i:
        imp = pd.to_numeric(df[col_i], errors="coerce").to_numpy(dtype="float64")
        ok &= np.isfinite(imp) & (imp >= 0)
    if ok.sum() == 0:
        raise ValueError("No hay filas con números válidos para priorizar.")
    r = riesgo[ok]
    # ¿el «riesgo» es una probabilidad o un puntaje? Si es un puntaje (mayor que 1, o con nombre de puntaje/orden), riesgo × impacto
    # NO es una pérdida esperada: sirve para ordenar, nada más.
    es_puntaje = bool(np.nanmax(r) > 1.0) or bool(set(_pistas.tokens(col_r)) & _pistas.RANKING)
    puntaje = r * imp[ok] if col_i else r
    ids = (df[col_id].astype(str).to_numpy()[ok] if col_id and col_id in df.columns
           else np.array([f"fila {i + 1}" for i in np.flatnonzero(ok)], dtype=object))
    orden = np.argsort(-puntaje, kind="stable")
    top = int(_g(cfg, "top", 20))
    tab = pd.DataFrame({"Prioridad": np.arange(1, len(orden) + 1), "Elemento": ids[orden], "Riesgo": r[orden]})
    if col_i:
        tab["Impacto"] = imp[ok][orden]
        tab["Puntaje (riesgo × impacto)" if es_puntaje else "Pérdida esperada"] = puntaje[orden]
    total = float(puntaje.sum())
    acum = np.cumsum(puntaje[orden]) / total if total > 0 else np.zeros(len(orden))
    tab["% acumulado del total"] = acum
    k = min(top, len(tab))
    conc = float(acum[k - 1]) if total > 0 else float("nan")
    primero = tab.iloc[0]
    etiqueta = ("puntaje (riesgo × impacto)" if es_puntaje else "pérdida esperada (riesgo × impacto)") if col_i else "riesgo"
    titular = (f"Empieza por «{primero['Elemento']}». Los primeros {k} de {len(tab)} concentran el {pct_txt(conc)} del total de {etiqueta}.")
    n80 = int(np.argmax(acum >= 0.8) + 1) if total > 0 and (acum >= 0.8).any() else len(tab)
    tarj = [(str(k), "Para revisar primero", f"concentran {pct_txt(conc)} del {etiqueta}"),
            (str(n80), "Para cubrir el 80%", f"de {len(tab)} elementos"),
            (num_txt(float(puntaje[orden][0])), "Puntaje del primero", str(primero["Elemento"]))]
    avisos = []
    if col_i is None:
        avisos.append("Sin columna de impacto se ordena solo por riesgo: un riesgo alto sobre algo poco importante pasa por delante de uno medio sobre algo crítico.")
    if es_puntaje:
        avisos.append(f"«{col_r}» parece un puntaje u orden, no una probabilidad: sirve para ORDENAR, pero "
                      + ("«riesgo × impacto» no es una pérdida esperada real ni el % acumulado es un % real del total."
                         if col_i else "sus valores no son porcentajes reales."))
    return ResultadoPronostico(
        "prioridad", "Prioridad", titular, "info", "media",
        "el orden es tan bueno como el riesgo y el impacto que le entregues.", tarj,
        [f"El puntaje de cada elemento es " + ("riesgo × impacto." if col_i else "su riesgo.")],
        ["El riesgo y el impacto de la tabla son confiables y comparables entre elementos.",
         "El riesgo se toma tal como viene: si es una probabilidad calibrada, riesgo × impacto es una pérdida esperada; "
         "si es un puntaje (por ejemplo, de un modelo sin calibrar), solo sirve para ordenar."], avisos,
        tab.head(max(top, 50)), "Orden de revisión", None, {"concentracion": conc})


# ----------------------------------------------------------------------
# CLASIFICACIÓN (envuelve el motor de Predicción)
# ----------------------------------------------------------------------
def analizar_clasificacion(df: pd.DataFrame, cfg) -> ResultadoPronostico:
    obj = _g(cfg, "columna_objetivo")
    if not obj or obj not in df.columns:
        raise ValueError("Elige la columna que quieres clasificar (la categoría a predecir).")
    # Validación: por fecha cuando la tabla tiene fechas (la prueba se parece a lo que pasará de verdad); al azar solo
    # si no hay fechas o si así se pidió en Ajustes. «auto» = por fecha si hay UNA columna de fecha clara.
    pref = _g(cfg, "validacion", "auto")
    fecha = _g(cfg, "columna_fecha") or None
    if fecha and fecha not in df.columns:
        raise ValueError(f"La columna de fecha «{fecha}» no está en la tabla.")
    notas_val = []
    if pref == "temporal" and not fecha:
        raise ValueError("Para probar «por fecha» elige la columna de fecha.")
    if pref in ("auto", "temporal") and not fecha:
        cand = columnas_fecha(df, excluir=[obj])
        if len(cand) == 1:
            fecha = cand[0]
            notas_val.append(f"Se probó con los casos más nuevos usando la fecha «{fecha}».")
        elif len(cand) > 1:
            notas_val.append("Hay varias columnas de fecha (" + ", ".join(f"«{c}»" for c in cand[:4])
                             + "): elige una en «Fecha» para probar con los casos más nuevos.")
    modo = "temporal" if (fecha and pref != "aleatoria") else "aleatoria"
    if fecha and pref != "aleatoria" and not notas_val:
        notas_val.append(f"Se probó con los casos más nuevos usando la fecha «{fecha}».")
    cfgp = ConfigPrediccion(objetivo=obj, validacion=modo, columna_fecha=fecha if modo == "temporal" else None,
                            salida="probabilidad")
    extra = list(_g(cfg, "columnas_explicativas", []) or [])
    if extra:
        cols = [c for c in extra if c in df.columns and c != obj]
    else:
        cols = columnas_a_usar(sugerir_columnas(df, obj), {})
    if not cols:
        raise ValueError("No quedaron columnas para predecir.")
    res = entrenar(df, cfgp, cols)
    nivel_v, titulo_v, frase_v = res.veredicto()
    conf = {"muy_bueno": "alta", "bueno": "alta", "regular": "media"}.get(str(nivel_v), "baja")
    imp = getattr(res, "importancias", None)
    tabla = pd.DataFrame(imp, columns=["Variable", "Importancia (%)"]) if imp else None
    avisos = list(res.advertencias)
    if nivel_v == "sospechoso":
        avisos.insert(0, frase_v)
    return _con_tipo(res.tipo_salida, ResultadoPronostico(
        "clasificacion", "Clasificación", f"{titulo_v}. {frase_v}",
        "info", conf, "calidad medida sobre filas que el modelo no vio al entrenar.", res.tarjetas(),
        res.lineas_resumen(), ["Los casos futuros se parecen a los del historial."] + [n for n in notas_val if n.startswith("Se probó")],
        avisos + [n for n in notas_val if not n.startswith("Se probó")], tabla,
        "Qué variables pesan más", None, {"metricas": dict(res.metricas), "mejora": float(res.mejora())}))