"""Bloques de simulación para un sistema digital M-PAM, M-PSK y M-QAM.

El mapeo, el conformador RRC, la modulación pasabanda, el canal AWGN,
el filtro adaptado y la decisión se implementan explícitamente con NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, log2, pi, sqrt

import numpy as np


@dataclass
class Resultado:
    """Agrupa las señales, parámetros y medidas producidas por una ejecución."""

    esquema: str
    orden: int
    rolloff: float
    ebno_db: float
    bits_tx: np.ndarray
    bits_rx: np.ndarray
    simbolos_tx: np.ndarray
    simbolos_rx: np.ndarray
    senal_basebanda: np.ndarray
    senal_pasabanda_rx: np.ndarray
    salida_filtro: np.ndarray
    muestras_por_simbolo: int
    retardo_total: int
    energia_simbolo: float
    energia_bit: float
    ancho_banda: float
    potencia_pasabanda: float

    @property
    def ber(self) -> float:
        return float(np.mean(self.bits_tx != self.bits_rx))


def _gray(n: int) -> int:
    # Convierte una etiqueta binaria en código Gray: puntos vecinos difieren
    # en un solo bit, lo que reduce errores de bit cuando falla un símbolo.
    return n ^ (n >> 1)


def _bits_a_enteros(bits: np.ndarray, k: int) -> np.ndarray:
    # Separa el flujo en grupos de k bits y lee cada grupo como un entero
    # binario, comenzando por el bit más significativo.
    return bits.reshape(-1, k).dot(1 << np.arange(k - 1, -1, -1))


def _enteros_a_bits(valores: np.ndarray, k: int) -> np.ndarray:
    # Operación inversa: representa cada entero con exactamente k bits.
    return ((valores[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def mapear(bits: np.ndarray, esquema: str, orden: int) -> np.ndarray:
    """Mapea grupos binarios a una constelación normalizada de energía media 1."""
    k = int(log2(orden))
    etiquetas = _bits_a_enteros(bits, k)
    grises = np.array([_gray(int(x)) for x in etiquetas])
    if esquema == "M-PAM":
        # M-PAM dispone sus M niveles en un eje real y normaliza su energía.
        niveles = 2 * grises - (orden - 1)
        puntos = niveles.astype(float)
        puntos /= sqrt((orden**2 - 1) / 3)
        return puntos.astype(complex)
    if esquema == "M-PSK":
        # En PSK, cada punto está sobre el círculo unidad; Gray define su fase.
        return np.exp(2j * pi * grises / orden)
    # En QAM cuadrada, las coordenadas I y Q forman una cuadrícula uniforme.
    raiz = int(sqrt(orden))
    if raiz * raiz != orden:
        raise ValueError("M-QAM requiere un orden cuadrado, por ejemplo 4, 16 o 64.")
    escala = sqrt(2 * (raiz**2 - 1) / 3)
    bits_por_eje = k // 2
    mascara_eje = raiz - 1
    # Se aplica Gray por separado a los bits de I y a los de Q. Aplicarlo
    # al entero completo produciría vecinos verticales con dos bits distintos.
    etiqueta_i = etiquetas >> bits_por_eje
    etiqueta_q = etiquetas & mascara_eje
    gray_i = np.array([_gray(int(x)) for x in etiqueta_i])
    gray_q = np.array([_gray(int(x)) for x in etiqueta_q])
    i = 2 * gray_i - (raiz - 1)
    q = 2 * gray_q - (raiz - 1)
    return (i + 1j * q) / escala


def decidir(muestras: np.ndarray, esquema: str, orden: int) -> np.ndarray:
    """Decisión de vecino más cercano y conversión inversa de etiqueta Gray."""
    k = int(log2(orden))
    etiquetas = np.arange(orden)
    grises = np.array([_gray(int(x)) for x in etiquetas])
    if esquema == "M-PAM":
        niveles = (2 * grises - (orden - 1)) / sqrt((orden**2 - 1) / 3)
        puntos = niveles.astype(complex)
    elif esquema == "M-PSK":
        puntos = np.exp(2j * pi * grises / orden)
    else:
        raiz = int(sqrt(orden))
        escala = sqrt(2 * (raiz**2 - 1) / 3)
        bits_por_eje = k // 2
        mascara_eje = raiz - 1
        etiqueta_i = etiquetas >> bits_por_eje
        etiqueta_q = etiquetas & mascara_eje
        gray_i = np.array([_gray(int(x)) for x in etiqueta_i])
        gray_q = np.array([_gray(int(x)) for x in etiqueta_q])
        puntos = (2 * gray_i - (raiz - 1) + 1j * (2 * gray_q - (raiz - 1))) / escala
    # Para cada muestra recibida, selecciona el punto de constelación más
    # cercano en distancia euclidiana y recupera su etiqueta binaria.
    indices_binarios = np.argmin(np.abs(muestras[:, None] - puntos[None, :]) ** 2, axis=1)
    return _enteros_a_bits(indices_binarios, k)


def rrc(rolloff: float, sps: int, span: int = 8) -> np.ndarray:
    """Pulso raíz de coseno alzado con energía discreta unitaria."""
    # El eje temporal se expresa en periodos de símbolo. Se tratan por
    # separado los puntos donde la fórmula tiene límites removibles.
    t = np.arange(-span * sps // 2, span * sps // 2 + 1, dtype=float) / sps
    h = np.empty_like(t)
    for n, x in enumerate(t):
        if abs(x) < 1e-12:
            h[n] = 1 + rolloff * (4 / pi - 1)
        elif rolloff > 0 and abs(abs(4 * rolloff * x) - 1) < 1e-10:
            h[n] = (rolloff / sqrt(2)) * (
                (1 + 2 / pi) * np.sin(pi / (4 * rolloff))
                + (1 - 2 / pi) * np.cos(pi / (4 * rolloff))
            )
        else:
            h[n] = (np.sin(pi * x * (1 - rolloff)) + 4 * rolloff * x * np.cos(pi * x * (1 + rolloff))) / (
                pi * x * (1 - (4 * rolloff * x) ** 2)
            )
    return h / np.linalg.norm(h)


def transmisor(
    bits: np.ndarray,
    esquema: str,
    orden: int,
    rolloff: float,
    sps: int,
    span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Bloque transmisor: bits → símbolos mapeados → pulsos RRC en banda base."""
    k = int(log2(orden))
    simbolos = mapear(bits, esquema, orden)
    pulso = rrc(rolloff, sps, span)
    retardo_pulso = (len(pulso) - 1) // 2

    # Cada símbolo se coloca cada sps muestras y el pulso limita el ancho de banda.
    tren_impulsos = np.zeros(len(simbolos) * sps, dtype=complex)
    tren_impulsos[::sps] = simbolos
    senal_basebanda = np.convolve(tren_impulsos, pulso, mode="full")
    return simbolos, senal_basebanda, pulso, retardo_pulso


def modulador_pasabanda(
    senal_basebanda: np.ndarray,
    sps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Traslada la señal compleja en banda base a una portadora real coherente."""
    muestras = np.arange(len(senal_basebanda))
    portadora = np.exp(2j * pi * 1.5 * muestras / sps)
    senal_pasabanda = sqrt(2) * np.real(senal_basebanda * portadora)
    return senal_pasabanda, portadora


def canal_awgn(
    senal_pasabanda: np.ndarray,
    ebno_db: float,
    bits_por_simbolo: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Canal AWGN real: suma ruido gaussiano con la varianza derivada de Eb/N0."""
    gamma_b = 10 ** (ebno_db / 10)
    energia_bit = 1 / bits_por_simbolo
    varianza_ruido = energia_bit / (2 * gamma_b)
    ruido = rng.normal(0, sqrt(varianza_ruido), len(senal_pasabanda))
    return senal_pasabanda + ruido, varianza_ruido


def receptor(
    senal_pasabanda_rx: np.ndarray,
    portadora: np.ndarray,
    pulso: np.ndarray,
    numero_simbolos: int,
    sps: int,
    retardo_pulso: int,
    esquema: str,
    orden: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bloque receptor: demodula, filtra, sincroniza, decide y recupera bits."""
    # La mezcla coherente vuelve a banda base; el filtro adaptado acumula la energía
    # del pulso. Se descartan los dos retardos de grupo antes de muestrear símbolos.
    senal_bb_rx = sqrt(2) * senal_pasabanda_rx * np.conj(portadora)
    salida_filtro = np.convolve(senal_bb_rx, pulso, mode="full")
    retardo_total = 2 * retardo_pulso
    indices = retardo_total + np.arange(numero_simbolos) * sps
    muestras_simbolo = salida_filtro[indices]
    bits_rx = decidir(muestras_simbolo, esquema, orden)
    return bits_rx, muestras_simbolo, salida_filtro


def simular(
    esquema: str = "M-QAM",
    orden: int = 4,
    rolloff: float = 0.25,
    ebno_db: float = 8.0,
    numero_simbolos: int = 1024,
    semilla: int = 2026,
    sps: int = 8,
    span: int = 8,
) -> Resultado:
    """Ejecuta una simulación pasabanda coherente con canal AWGN."""
    # Validación temprana: evita errores de dimensiones o constelaciones
    # indefinidas y permite que la interfaz muestre mensajes comprensibles.
    if esquema not in {"M-PAM", "M-PSK", "M-QAM"}:
        raise ValueError("El esquema debe ser M-PAM, M-PSK o M-QAM.")
    if not isinstance(orden, (int, np.integer)) or orden < 2 or orden & (orden - 1):
        raise ValueError("El orden M debe ser potencia de dos y mayor o igual a 2.")
    if esquema == "M-QAM" and int(sqrt(orden)) ** 2 != orden:
        raise ValueError("M-QAM requiere un orden cuadrado (4, 16, 64, ...).")
    if (
        not isinstance(rolloff, (int, float, np.integer, np.floating))
        or not np.isfinite(rolloff)
        or not 0 <= rolloff <= 1
    ):
        raise ValueError("El factor roll-off debe estar entre 0 y 1.")
    if not isinstance(ebno_db, (int, float, np.integer, np.floating)) or not np.isfinite(ebno_db):
        raise ValueError("Eb/N₀ debe ser un número finito.")
    if not isinstance(numero_simbolos, (int, np.integer)) or numero_simbolos < 64:
        raise ValueError("Use al menos 64 símbolos para la simulación.")
    if not isinstance(sps, (int, np.integer)) or sps < 4 or sps % 2:
        raise ValueError("Use un número par de muestras por símbolo, mínimo 4.")
    if not isinstance(span, (int, np.integer)) or span < 2:
        raise ValueError("La duración del filtro (span) debe ser de al menos 2 símbolos.")
    if not isinstance(semilla, (int, np.integer)) or semilla < 0:
        raise ValueError("La semilla debe ser un entero no negativo.")

    # Orquestador de los bloques. La semilla controla los bits y el ruido
    # para poder repetir una misma simulación.
    rng = np.random.default_rng(semilla)
    k = int(log2(orden))
    bits_tx = rng.integers(0, 2, numero_simbolos * k, dtype=np.uint8)
    simbolos, basebanda, pulso, retardo = transmisor(
        bits_tx, esquema, orden, rolloff, sps, span
    )
    pasabanda, portadora = modulador_pasabanda(basebanda, sps)
    rx_pb, _ = canal_awgn(pasabanda, ebno_db, k, rng)
    bits_rx, muestras, salida = receptor(
        rx_pb, portadora, pulso, numero_simbolos, sps, retardo, esquema, orden
    )
    retardo_total = 2 * retardo
    # Con la constelación normalizada Es=1; por tanto Eb=Es/log2(M).
    # El ancho pasabanda nulo a nulo de coseno alzado es (1+alpha)*Rs.
    energia_s = 1.0
    energia_b = energia_s / k
    # Ancho nulo a nulo del espectro pasabanda real, con Rs normalizada a 1.
    ancho_banda = 1 + rolloff
    potencia = float(np.mean(pasabanda**2) * sps)
    return Resultado(
        esquema, orden, rolloff, ebno_db, bits_tx, bits_rx, simbolos, muestras,
        basebanda, rx_pb, salida, sps, retardo_total, energia_s, energia_b,
        ancho_banda, potencia,
    )


def qfunc(x: float | np.ndarray) -> float | np.ndarray:
    """Función Q vectorizada, sin depender de librerías especiales."""
    return 0.5 * np.vectorize(erfc, otypes=[float])(np.asarray(x) / sqrt(2))


def ber_teorica(esquema: str, orden: int, ebno_db: np.ndarray | float) -> np.ndarray:
    """BER teórica Gray: expresiones estándar (aprox. para M>2 en PSK/QAM)."""
    if esquema not in {"M-PAM", "M-PSK", "M-QAM"}:
        raise ValueError("El esquema debe ser M-PAM, M-PSK o M-QAM.")
    if not isinstance(orden, (int, np.integer)) or orden < 2 or orden & (orden - 1):
        raise ValueError("El orden M debe ser potencia de dos y mayor o igual a 2.")
    if esquema == "M-QAM" and int(sqrt(orden)) ** 2 != orden:
        raise ValueError("M-QAM requiere un orden cuadrado (4, 16, 64, ...).")
    valores_db = np.asarray(ebno_db, dtype=float)
    if not np.all(np.isfinite(valores_db)):
        raise ValueError("Eb/N₀ teórico debe contener valores finitos.")
    gamma = 10 ** (valores_db / 10)
    k = log2(orden)
    # Las expresiones relacionan Eb/N0 con la probabilidad de error de bit.
    # Para BPSK (2-PSK) se usa su resultado exacto; la expresión genérica de
    # M-PSK sería una cota aproximada y duplicaría la BER en ese caso.
    if esquema == "M-PAM":
        x = sqrt(6 * k / (orden**2 - 1)) * np.sqrt(gamma)
        return 2 * (orden - 1) / (orden * k) * qfunc(x)
    if esquema == "M-PSK":
        if orden == 2:
            return qfunc(np.sqrt(2 * gamma))
        x = np.sqrt(2 * k * gamma) * np.sin(pi / orden)
        return 2 / k * qfunc(x)
    raiz = sqrt(orden)
    x = np.sqrt(3 * k * gamma / (orden - 1))
    return (4 / k) * (1 - 1 / raiz) * qfunc(x)


def ebno_requerido(esquema: str, orden: int, ber_objetivo: float) -> float:
    """Búsqueda binaria de Eb/N0 (dB) para una BER teórica objetivo."""
    # La BER decrece al aumentar Eb/N0. La búsqueda acota el valor en dB
    # hasta encontrar el punto donde la teoría alcanza la BER solicitada.
    if not 0 < ber_objetivo < 0.5:
        raise ValueError("La BER objetivo debe estar entre 0 y 0.5, sin incluir los extremos.")
    bajo, alto = -10.0, 50.0
    for _ in range(80):
        medio = (bajo + alto) / 2
        if float(ber_teorica(esquema, orden, medio)) > ber_objetivo:
            bajo = medio
        else:
            alto = medio
    return (bajo + alto) / 2
