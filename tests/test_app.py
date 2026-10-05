"""Pruebas sin navegador para la interfaz Streamlit."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class PruebasInterfaz(unittest.TestCase):
    def test_app_renderiza_graficas_metricas_y_comparacion(self):
        ruta_app = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(ruta_app), default_timeout=45).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        textos = [elemento.value for elemento in app.subheader]
        self.assertIn("Diagrama de ojo en recepción", textos)
        self.assertIn("Constelación en recepción", textos)
        self.assertIn("Comparación teórica: 4-PAM, 4-QAM y 4-PSK", textos)
        self.assertGreaterEqual(len(app.metric), 6)
        # AppTest expone los componentes Plotly como elementos desconocidos.
        self.assertGreaterEqual(len(app.get("plotly_chart")), 6)

    def test_cambio_de_esquema_recalcula_sin_boton(self):
        ruta_app = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(ruta_app), default_timeout=45).run()
        app.selectbox[0].select("M-PAM").run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertTrue(any(elemento.value.startswith("Simulación actual: M-PAM") for elemento in app.caption))
        self.assertGreaterEqual(len(app.metric), 6)

    def test_cualquier_parametro_actualiza_automaticamente_la_vista(self):
        ruta_app = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(ruta_app), default_timeout=45).run()
        app.slider[0].set_value(0.5)
        app.slider[1].set_value(10.0)
        app.select_slider[0].set_value(2048)
        app.number_input[0].set_value(2027)
        app.run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        captions = [elemento.value for elemento in app.caption]
        self.assertTrue(any("roll-off=0.50, Eb/N₀=10.0 dB, 2,048 símbolos, semilla=2027" in c for c in captions))
        self.assertGreaterEqual(len(app.get("plotly_chart")), 6)


if __name__ == "__main__":
    unittest.main()
