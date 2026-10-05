"""Pruebas automáticas para los bloques matemáticos del simulador."""

import unittest

import numpy as np

from simulador import (
    ber_teorica,
    canal_awgn,
    decidir,
    ebno_requerido,
    mapear,
    modulador_pasabanda,
    receptor,
    rrc,
    simular,
    transmisor,
)


class PruebasSimulador(unittest.TestCase):
    """Comprueba mapeo, bloques, cálculos y repetibilidad del sistema."""

    def test_mapeo_y_decision_sin_ruido(self):
        # Todo mensaje válido debe sobrevivir al recorrido de ida y decisión.
        rng = np.random.default_rng(10)
        for esquema, orden in [("M-PAM", 4), ("M-PSK", 8), ("M-QAM", 16)]:
            bits = rng.integers(0, 2, 240, dtype=np.uint8)
            np.testing.assert_array_equal(decidir(mapear(bits, esquema, orden), esquema, orden), bits)

    def test_vecinos_qam_gray_difieren_en_un_bit(self):
        # En una cuadrícula QAM, toda pareja de vecinos horizontales/verticales
        # debe cambiar exactamente uno de los bits del símbolo.
        orden = 16
        etiquetas = np.arange(orden, dtype=np.uint8)
        bits = ((etiquetas[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8).ravel()
        puntos = mapear(bits, "M-QAM", orden)
        distancias = np.abs(puntos[:, None] - puntos[None, :]) ** 2
        distancia_vecino = np.min(distancias[distancias > 1e-12])
        for a in range(orden):
            for b in range(a + 1, orden):
                if np.isclose(distancias[a, b], distancia_vecino):
                    self.assertEqual((a ^ b).bit_count(), 1)

    def test_pulso_rrc_tiene_energia_unitaria(self):
        pulso = rrc(0.25, 8, 8)
        self.assertAlmostEqual(float(np.sum(pulso**2)), 1.0, places=12)

    def test_bloques_transmisor_canal_receptor(self):
        bits = np.tile(np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8), 32)
        simbolos, basebanda, pulso, retardo = transmisor(bits, "M-QAM", 4, 0.25, 8, 8)
        pasabanda, portadora = modulador_pasabanda(basebanda, 8)
        sin_ruido, varianza = canal_awgn(pasabanda, 100, 2, np.random.default_rng(2))
        bits_rx, muestras, salida = receptor(sin_ruido, portadora, pulso, len(simbolos), 8, retardo, "M-QAM", 4)
        np.testing.assert_array_equal(bits_rx, bits)
        self.assertEqual(len(muestras), len(simbolos))
        self.assertGreater(varianza, 0)
        self.assertEqual(len(salida), len(sin_ruido) + len(pulso) - 1)

    def test_simulacion_reproducible_en_tres_esquemas(self):
        for esquema in ("M-PAM", "M-PSK", "M-QAM"):
            a = simular(esquema, 4, 0.25, 8, 256, 73)
            b = simular(esquema, 4, 0.25, 8, 256, 73)
            np.testing.assert_array_equal(a.bits_tx, b.bits_tx)
            np.testing.assert_array_equal(a.bits_rx, b.bits_rx)
            self.assertAlmostEqual(a.energia_simbolo, 1.0)
            self.assertAlmostEqual(a.energia_bit, 0.5)

    def test_curvas_y_comparacion_teorica(self):
        # 4-QAM y 4-PSK son constelaciones equivalentes salvo una rotación.
        ebno = np.array([0.0, 4.0, 8.0])
        np.testing.assert_allclose(ber_teorica("M-QAM", 4, ebno), ber_teorica("M-PSK", 4, ebno))
        self.assertTrue(np.all(np.diff(ber_teorica("M-PAM", 4, ebno)) < 0))
        requerido = ebno_requerido("M-QAM", 4, 1e-3)
        self.assertAlmostEqual(float(ber_teorica("M-QAM", 4, requerido)), 1e-3, places=10)

    def test_ber_simulada_converge_hacia_teoria_a_6_db(self):
        # Compara BER medida y aproximada donde se espera un número suficiente
        # de errores para que la estimación Monte Carlo tenga sentido.
        numero_simbolos = 20_000
        ebno_db = 6.0
        for esquema in ("M-PAM", "M-QAM", "M-PSK"):
            resultado = simular(esquema, 4, 0.25, ebno_db, numero_simbolos, 314159)
            referencia = float(ber_teorica(esquema, 4, ebno_db))
            numero_bits = numero_simbolos * 2
            desviacion = np.sqrt(referencia * (1 - referencia) / numero_bits)
            self.assertLessEqual(abs(resultado.ber - referencia), 6 * desviacion + 1 / numero_bits)

    def test_rechazo_de_parametros_invalidos(self):
        casos = [
            {"rolloff": 1.1},
            {"esquema": "M-QAM", "orden": 8},
            {"orden": 3},
            {"numero_simbolos": 32},
            {"semilla": -1},
        ]
        for cambios in casos:
            with self.subTest(cambios=cambios), self.assertRaises(ValueError):
                simular(**cambios)
        with self.assertRaises(ValueError):
            ebno_requerido("M-QAM", 4, 0.0)


if __name__ == "__main__":
    unittest.main()
