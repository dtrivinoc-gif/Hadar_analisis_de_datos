"""
Configuración global de Hadar Data Analytics: rutas, colores/temas y
constantes compartidas entre módulos. No depende de ningún otro módulo
propio de Hadar -- todos los demás pueden importar de acá sin riesgo de
importación circular.
"""
import os
import sys


# ----------------------------------------------------------------------------
# Paletas de colores
# ----------------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg": "#000020",        # Azul profundo (secondary)
        "card": "#0A0A0A",      # Negro puro (dark)
        "text": "#FFFFFF",      # Blanco (primary)
        "border": "#1E293B",
        "muted": "#94A3B8",
    },
    "light": {
        "bg": "#FFFFFF",        # Blanco (primary)
        "card": "#F3F4F9",      # Blanco con un leve tinte del azul profundo
        "text": "#0A0A0A",      # Negro puro (dark)
        "border": "#E2E8F0",
        "muted": "#64748B",
    },
    "gray": {
        "bg": "#D4D4D8",        # Gris claro (zinc-300)
        "card": "#FFFFFF",      # Blanco
        "text": "#18181B",      # Casi negro, buena legibilidad sobre gris
        "border": "#A1A1AA",
        "muted": "#52525B",
    },
    "tactical": {
        "bg": "#000000",        # Negro puro
        "card": "#27272A",      # Gris oscuro (zinc-800)
        "text": "#FFFFFF",      # Blanco
        "border": "#3F3F46",
        "muted": "#A1A1AA",
    },
}
COLOR_ACCENT = "#6366F1"    # Índigo (accent_1) — acciones principales
COLOR_ACCENT_2 = "#06B6D4"  # Cian vibrante (accent_2) — acciones secundarias/actualizar
COLOR_ACCENT_3 = "#F59E0B"  # Ámbar dorado (accent_3) — estados activos/interruptores
COLOR_HOVER = "#4F46E5"
COLOR_DANGER = "#EF4444"

# ----------------------------------------------------------------------------
# Umbral para decidir el motor de carga: Polars para datasets masivos,
# Pandas para el resto (evita el "costo" de Polars en archivos chicos y
# mantiene el resto de la app -- filtros, tabla, gráficos -- sin cambios,
# porque siempre reciben un pandas.DataFrame al final).
# ----------------------------------------------------------------------------
FILAS_UMBRAL_MASIVO = 300_000        # filas: sobre esto, se prefiere Polars
TAMANO_UMBRAL_MASIVO_BYTES = 30 * 1024 * 1024  # 30 MB: heurística para CSV/Excel,
                                                 # donde no se puede saber el N° de
                                                 # filas sin leer el archivo completo

CHART_TYPES = ["Barras", "Líneas", "Dispersión", "Dona / Torta", "Histograma", "Mapa de Calor (Correlación)"]
DONUT_PALETTE = [
    "#6366F1", "#22D3EE", "#F472B6", "#FBBF24", "#34D399",
    "#F87171", "#A78BFA", "#60A5FA", "#FB923C", "#4ADE80",
]


def resolve_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        path = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(path):
            return path
    base_dir = os.path.dirname(
        os.path.abspath(sys.argv[0] if getattr(sys, "frozen", False) else __file__)
    )
    path = os.path.join(base_dir, relative_path)
    if os.path.exists(path):
        return path
    return os.path.abspath(relative_path)


ICON_PATH = resolve_path("logo.ico")
LOGO_PNG_PATH = resolve_path("logo.png")


# ----------------------------------------------------------------------------
# Conexión al POS (fuente de datos "api_pos", ver io_datos.py) -- para
# sincronizar en la misma red wifi con el PC donde corre POS by Hadar.
#
# Igual que las credenciales de Turso en el POS, la API key NUNCA se
# escribe acá adentro ni se sube a GitHub -- se define como variable de
# entorno en ESTE PC (el de Analytics). La URL sí tiene un valor por
# defecto editable, porque no es secreta, solo hay que ajustarla si la
# IP o el nombre de red del PC del POS cambia.
# ----------------------------------------------------------------------------
URL_POS_SINCRONIZACION = os.environ.get(
    "ANALYTICS_URL_POS", "http://192.168.1.100:8000"
)
API_KEY_POS_SINCRONIZACION = os.environ.get("ANALYTICS_API_KEY_POS", "")