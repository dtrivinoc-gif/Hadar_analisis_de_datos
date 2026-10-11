"""
Envío de avisos por correo cuando se activa una Alarma.

Usa una cuenta de Gmail dedicada como remitente (por defecto,
hadar.notificaciones@gmail.com). Gmail exige una "contraseña de
aplicación" para enviar por este método -- no es la contraseña normal de
la cuenta, se genera aparte en la configuración de seguridad de Google
(con la verificación en 2 pasos activada) y se pega en el diálogo de acá.

El envío corre en un hilo aparte (mismo patrón que UfWorker en
graficos.py) para no trabar la ventana mientras se conecta a Gmail.
"""
import json
import os
import smtplib
from email.mime.text import MIMEText

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit, QCheckBox,
    QPushButton, QMessageBox,
)

from .memoria import ruta_directorio_datos_usuario

CORREO_REMITENTE_POR_DEFECTO = "hadar.notificaciones@gmail.com"
SERVIDOR_SMTP = "smtp.gmail.com"
PUERTO_SMTP = 587


def ruta_config_correo():
    return os.path.join(ruta_directorio_datos_usuario(), "hadar_correo_config.json")


def cargar_configuracion():
    ruta = ruta_config_correo()
    config_por_defecto = {
        "habilitado": False,
        "correo_remitente": CORREO_REMITENTE_POR_DEFECTO,
        "contrasena_app": "",
        "correo_destino_predeterminado": "",
    }
    if not os.path.exists(ruta):
        return config_por_defecto
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            guardado = json.load(f)
        config_por_defecto.update(guardado)
        return config_por_defecto
    except (json.JSONDecodeError, OSError):
        return config_por_defecto


def guardar_configuracion(config):
    with open(ruta_config_correo(), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _armar_mensaje(regla, valor_actual, detalle=None):
    asunto = f"ALARMA Hadar: {regla['nombre']}"
    if detalle:
        # Alarmas de tipo Columna: el detalle ya viene armado (columna,
        # condición, filtro, cuántas filas cumplen).
        cuerpo = f"¡ALARMA! {regla['nombre']}\n\n{detalle}"
    else:
        valor_texto = (
            "sin datos" if valor_actual is None
            else f"{valor_actual:,.2f}".replace(",", ".")
        )
        cuerpo = (
            f"¡ALARMA! {regla['nombre']}\n\n"
            f"Condición: {regla['tipo']} · {regla['objetivo']} {regla['operador']} {regla['umbral']}\n"
            f"Valor actual: {valor_texto}\n"
        )
    if regla.get("mensaje"):
        cuerpo += f"\nNota: {regla['mensaje']}\n"
    return asunto, cuerpo


def _enviar_smtp(config, destinatario, asunto, cuerpo):
    mensaje = MIMEText(cuerpo, "plain", "utf-8")
    mensaje["Subject"] = asunto
    mensaje["From"] = config["correo_remitente"]
    mensaje["To"] = destinatario

    with smtplib.SMTP(SERVIDOR_SMTP, PUERTO_SMTP, timeout=15) as servidor:
        servidor.starttls()
        servidor.login(config["correo_remitente"], config["contrasena_app"])
        servidor.sendmail(config["correo_remitente"], [destinatario], mensaje.as_string())


class CorreoWorker(QObject):
    """Manda un único correo. Vive en su propio QThread -- ver
    disparar_envio_correo() más abajo, que arma el hilo con el mismo
    patrón que UfWorker en graficos.py."""
    resultado = Signal(bool, str)

    def __init__(self, config, destinatario, asunto, cuerpo):
        super().__init__()
        self.config = config
        self.destinatario = destinatario
        self.asunto = asunto
        self.cuerpo = cuerpo

    def run(self):
        try:
            _enviar_smtp(self.config, self.destinatario, self.asunto, self.cuerpo)
            self.resultado.emit(True, "")
        except Exception as exc:
            self.resultado.emit(False, str(exc))


class _PuenteResultado(QObject):
    """Puente para que el callback de resultado se ejecute siempre en el
    hilo principal. Sin esto, conectar la señal directamente a una
    función común (no un método de un QObject) hace que Qt la ejecute en
    el hilo de envío -- territorio inestable para tocar la interfaz."""
    listo = Signal(bool, str)

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self.listo.connect(self._ejecutar)

    def _ejecutar(self, ok, error):
        if self._callback:
            self._callback(ok, error)


def disparar_envio_correo(host, destinatario, regla, valor_actual, on_resultado=None,
                           verificar_habilitado=True, detalle=None):
    """Manda el correo de una alarma en un hilo aparte.

    `host` es la ventana principal (HadarApp): necesita una lista
    `host._hilos_correo` donde guardamos una referencia al hilo (y al
    puente) mientras dura el envío -- si no se guarda en algún lado,
    Python los destruye a mitad de camino y el correo nunca sale.

    `verificar_habilitado=False` se usa para el botón "correo de prueba":
    ahí queremos poder probar la contraseña ANTES de decidir activar el
    interruptor de verdad, así que no debe bloquearse por eso."""
    if not hasattr(host, "_hilos_correo"):
        host._hilos_correo = []

    config = cargar_configuracion()
    if (verificar_habilitado and not config.get("habilitado")) or not config.get("contrasena_app"):
        if on_resultado:
            on_resultado(False, "Notificaciones por correo no configuradas.")
        return

    asunto, cuerpo = _armar_mensaje(regla, valor_actual, detalle)

    hilo = QThread()
    worker = CorreoWorker(config, destinatario, asunto, cuerpo)
    worker.moveToThread(hilo)
    hilo.started.connect(worker.run)

    puente = _PuenteResultado(on_resultado) if on_resultado else None

    def _al_terminar(ok, error):
        if puente is not None:
            puente.listo.emit(ok, error)
        hilo.quit()

    worker.resultado.connect(_al_terminar)
    worker.resultado.connect(worker.deleteLater)
    hilo.finished.connect(hilo.deleteLater)

    referencia = (hilo, worker, puente)

    def _liberar():
        if referencia in host._hilos_correo:
            host._hilos_correo.remove(referencia)

    hilo.finished.connect(_liberar)

    host._hilos_correo.append(referencia)
    hilo.start()


# ----------------------------------------------------------------------------
# Diálogo de configuración
# ----------------------------------------------------------------------------

class DialogoConfiguracionCorreo(QDialog):
    """Configurar la cuenta que manda los avisos y el correo de destino
    predeterminado. Se guarda en un archivo local (hadar_correo_config.json),
    no en el código -- así la contraseña de aplicación no queda pegada en
    ningún archivo que se comparta."""

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.setWindowTitle("Configurar avisos por correo")
        self.setMinimumWidth(420)
        config = cargar_configuracion()

        layout = QVBoxLayout(self)

        self.checkbox_habilitado = QCheckBox(
            "Enviar avisos por correo cuando se active una alarma"
        )
        self.checkbox_habilitado.setChecked(config.get("habilitado", False))
        layout.addWidget(self.checkbox_habilitado)

        layout.addWidget(QLabel("Correo remitente (la cuenta que manda los avisos):"))
        self.remitente_edit = QLineEdit(config.get("correo_remitente", CORREO_REMITENTE_POR_DEFECTO))
        layout.addWidget(self.remitente_edit)

        layout.addWidget(QLabel(
            "Contraseña de aplicación de Gmail\n"
            "(no es tu contraseña normal -- se genera en myaccount.google.com,\n"
            "con la verificación en 2 pasos activada primero):"
        ))
        self.contrasena_edit = QLineEdit(config.get("contrasena_app", ""))
        self.contrasena_edit.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.contrasena_edit)

        layout.addWidget(QLabel("Correo de destino predeterminado:"))
        self.destino_edit = QLineEdit(config.get("correo_destino_predeterminado", ""))
        self.destino_edit.setPlaceholderText("tu-correo@ejemplo.com")
        layout.addWidget(self.destino_edit)

        self.btn_probar = QPushButton("Guardar y enviar correo de prueba")
        self.btn_probar.clicked.connect(self._enviar_prueba)
        layout.addWidget(self.btn_probar)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self._guardar_y_aceptar)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _config_actual(self):
        return {
            "habilitado": self.checkbox_habilitado.isChecked(),
            "correo_remitente": self.remitente_edit.text().strip() or CORREO_REMITENTE_POR_DEFECTO,
            "contrasena_app": self.contrasena_edit.text().strip(),
            "correo_destino_predeterminado": self.destino_edit.text().strip(),
        }

    def _enviar_prueba(self):
        destino = self.destino_edit.text().strip()
        if not destino:
            QMessageBox.information(self, "Falta el correo", "Completá el correo de destino primero.")
            return
        if not self.contrasena_edit.text().strip():
            QMessageBox.information(self, "Falta la contraseña", "Completá la contraseña de aplicación primero.")
            return

        guardar_configuracion(self._config_actual())
        self.btn_probar.setEnabled(False)
        self.btn_probar.setText("Enviando...")

        regla_prueba = {
            "nombre": "Prueba de configuración", "tipo": "Indicador", "objetivo": "-",
            "operador": "-", "umbral": 0,
            "mensaje": "Si recibiste esto, la configuración de correo de Hadar funciona.",
        }

        def _resultado(ok, error):
            self.btn_probar.setEnabled(True)
            self.btn_probar.setText("Guardar y enviar correo de prueba")
            if ok:
                QMessageBox.information(self, "Listo", "Correo de prueba enviado. Revisá tu bandeja de entrada.")
            else:
                QMessageBox.warning(self, "No se pudo enviar", error or "Error desconocido.")

        disparar_envio_correo(self.host, destino, regla_prueba, None, on_resultado=_resultado,
                               verificar_habilitado=False)

    def _guardar_y_aceptar(self):
        guardar_configuracion(self._config_actual())
        self.accept()