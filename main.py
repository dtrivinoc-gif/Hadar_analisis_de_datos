"""
Punto de entrada de la aplicación. Ejecutar con: python -m hadar.main
(o crear un lanzador run_hadar.py en la raíz del proyecto, ver más abajo).
"""
import ctypes
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog

from .main_window import HadarApp
from .pantalla_inicio import PantallaInicio

# El ícono debe estar en la misma carpeta que este archivo (main.py).
RUTA_ICONO = Path(__file__).parent / "hadar.ico"


def main():
    # En Windows, sin esto la barra de tareas muestra el ícono de python.exe
    # en vez del de Hadar. Debe ejecutarse antes de crear la aplicación.
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("hadar.analytics")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(RUTA_ICONO)))

    pantalla = PantallaInicio()
    if pantalla.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)  # cerró la pantalla de inicio sin elegir nada

    window = HadarApp(modo=pantalla.modo)
    if pantalla.modo == "continuar" and pantalla.ruta_proyecto:
        window.abrir_proyecto_desde_ruta(pantalla.ruta_proyecto)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()