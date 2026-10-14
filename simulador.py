"""Motor de simulación de un enlace digital M-PAM, M-PSK y M-QAM.

El archivo se organiza por etapas del enlace: representación de bits, mapeo,
formación de pulsos, modulación pasabanda, canal AWGN y recepción. Las
operaciones de señal se escriben con NumPy para que cada bloque sea visible
y explicable; no se llaman moduladores de alto nivel.

Todas las energías usan una constelación normalizada a Es=1 y una tasa de
símbolo Rs=1. Por eso las métricas se reportan en unidades normalizadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, log2, pi, sqrt

import numpy as np


@dataclass
class Resultado:
    """Paquete con todas las señales y métricas de una ejecución.

    Guardar las señales intermedias permite que la interfaz dibuje espectros,
    constelación y diagrama de ojo sin volver a ejecutar el receptor.
    """

    # Parámetros escogidos en la interfaz.
    esquema: str
    orden: int
    rolloff: float
    ebno_db: float
    # Flujo de información comparado para medir los errores.
    bits_tx: np.ndarray
    bits_rx: np.ndarray
    # Símbolos antes y después del canal y sus filtros.
    simbolos_tx: np.ndarray
    simbolos_rx: np.ndarray
    # Formas de onda para construir las gráficas.
    senal_basebanda: np.ndarray
    senal_pasabanda_rx: np.ndarray
    salida_filtro: np.ndarray
    # Constantes temporales y medidas calculadas del sistema.
    muestras_por_simbolo: int
    retardo_total: int
    energia_simbolo: float
    energia_bit: float
    ancho_banda: float
    potencia_pasabanda: float

    @property
    def ber(self) -> float:
        """Proporción de posiciones de bit donde transmisión y recepción difieren."""
        return float(np.mean(self.bits_tx != self.bits_rx))


def _gray(n: int) -> int:
    # Convierte una etiqueta binaria en código Gray: puntos vecinos difieren
    # en un solo bit, lo que reduce errores de bit cuando falla un símbolo.
    return n ^ (n >> 1)


def _bits_a_enteros(bits: np.ndarray, k: int) -> np.ndarray:
    # Separa el flujo en grupos de k bits y lee cada grupo como un entero
    # binario, comenzando por el bit más significativo.
    grupos = bits.reshape(-1, k)
    pesos_binarios = 1 << np.arange(k - 1, -1, -1)
    return grupos.dot(pesos_binarios)


def _enteros_a_bits(valores: np.ndarray, k: int) -> np.ndarray:
    # Operación inversa: extrae cada posición binaria y deja exactamente k bits.
    return ((valores[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def mapear(bits: np.ndarray, esquema: str, orden: int) -> np.ndarray:
    """Convierte cada grupo de bits en un punto de constelación de energía media 1.

    `orden` es M, la cantidad de puntos posibles, y cada símbolo lleva
    `k=log2(M)` bits. Las etiquetas Gray minimizan errores de bit entre vecinos.
    """
    k = int(log2(orden))
    etiquetas = _bits_a_enteros(bits, k)
    grises = np.array([_gray(int(x)) for x in etiquetas])
    if esquema == "M-PAM":
        # M-PAM dispone sus M niveles equiespaciados en el eje real.
        niveles = 2 * grises - (orden - 1)
        puntos = niveles.astype(float)
        # La división hace que el promedio de |símbolo|^2 sea uno.
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
    """Decide el símbolo más cercano y devuelve los bits asociados.

    El arreglo `puntos` se construye en el mismo orden que `mapear`; por ello,
    el índice del punto elegido vuelve a ser la etiqueta binaria transmitida.
    """
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
    diferencias = muestras[:, None] - puntos[None, :]
    distancias_cuadradas = np.abs(diferencias) ** 2
    indices_binarios = np.argmin(distancias_cuadradas, axis=1)
    return _enteros_a_bits(indices_binarios, k)


def rrc(rolloff: float, sps: int, span: int = 8) -> np.ndarray:
    """Calcula el pulso raíz de coseno alzado (RRC) muestreado.

    `rolloff` controla el exceso de ancho de banda, `sps` es el número de
    muestras por símbolo y `span` la longitud del pulso en símbolos.
    """
    # El eje temporal se expresa en periodos de símbolo. Se tratan por
    # separado los puntos donde la fórmula tiene límites removibles.
    t = np.arange(-span * sps // 2, span * sps // 2 + 1, dtype=float) / sps
    h = np.empty_like(t)
    for n, x in enumerate(t):
        if abs(x) < 1e-12:
            # Límite de la fórmula RRC cuando t=0; evita dividir por cero.
            h[n] = 1 + rolloff * (4 / pi - 1)
        elif rolloff > 0 and abs(abs(4 * rolloff * x) - 1) < 1e-10:
            # Límite en t=±1/(4α), donde la expresión general también es 0/0.
            h[n] = (rolloff / sqrt(2)) * (
                (1 + 2 / pi) * np.sin(pi / (4 * rolloff))
                + (1 - 2 / pi) * np.cos(pi / (4 * rolloff))
            )
        else:
            h[n] = (np.sin(pi * x * (1 - rolloff)) + 4 * rolloff * x * np.cos(pi * x * (1 + rolloff))) / (
                pi * x * (1 - (4 * rolloff * x) ** 2)
            )
    # Normalización: el filtro adaptado tendrá energía discreta unitaria.
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
    # Mapeo: agrupa los bits y selecciona el punto de la constelación.
    simbolos = mapear(bits, esquema, orden)
    # Conformación: un tren de impulsos produce una copia escalada del pulso
    # en cada instante de símbolo. El RRC limita el espectro y reduce ISI.
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
    # La portadora elegida es fc=1.5 Rs; el tiempo está en muestras discretas.
    muestras = np.arange(len(senal_basebanda))
    portadora = np.exp(2j * pi * 1.5 * muestras / sps)
    # Re{u(t)e^(j2πfct)} crea la señal pasabanda real. sqrt(2) compensa que
    # el promedio de cos² sea 1/2 con la convención de energía usada.
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
    # Es=1, entonces Eb=Es/k. Para ruido real AWGN, σ²=N0/2=Eb/(2γb).
    energia_bit = 1 / bits_por_simbolo
    varianza_ruido = energia_bit / (2 * gamma_b)
    ruido = rng.normal(0, sqrt(varianza_ruido), len(senal_pasabanda))
    # El canal suma ruido independiente muestra a muestra.
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
    # Mezcla con el oscilador local sincronizado; conjugado elimina la fase
    # de portadora de la señal útil y desplaza el contenido a banda base.
    senal_bb_rx = sqrt(2) * senal_pasabanda_rx * np.conj(portadora)
    salida_filtro = np.convolve(senal_bb_rx, pulso, mode="full")
    retardo_total = 2 * retardo_pulso
    # La señal pasó por un RRC al transmitir y otro al recibir. Se muestrea
    # solo en los máximos de cada símbolo, después de compensar ambos retardos.
    indices = retardo_total + np.arange(numero_simbolos) * sps
    muestras_simbolo = salida_filtro[indices]
    # Decisor + demapeador: convierte cada punto estimado de nuevo en bits.
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
    """Ejecuta una trama completa y devuelve señales, bits y métricas.

    Esta función coordina los bloques; la lógica de cada etapa vive en una
    función independiente para facilitar su lectura, prueba y sustentación.
    """
    # Validación temprana: evita errores de dimensiones o constelaciones
    # indefinidas y permite que la interfaz muestre mensajes comprensibles.
    # Las validaciones protegen al motor aunque se invoque fuera de la GUI.
    if esquema not in {"M-PAM", "M-PSK", "M-QAM"}:
        raise ValueError("El esquema debe ser M-PAM, M-PSK o M-QAM.")
    # Los grupos binarios requieren una cantidad M potencia de dos.
    if not isinstance(orden, (int, np.integer)) or orden < 2 or orden & (orden - 1):
        raise ValueError("El orden M debe ser potencia de dos y mayor o igual a 2.")
    if esquema == "M-QAM" and int(sqrt(orden)) ** 2 != orden:
        raise ValueError("M-QAM requiere un orden cuadrado (4, 16, 64, ...).")
    # Parámetros del filtro, tamaño de trama y semilla deben ser válidos para
    # evitar divisiones imposibles, arreglos mal dimensionados o ejecución no reproducible.
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
    # Una sola secuencia aleatoria produce primero los bits y luego el ruido.
    # Así, una misma semilla y los mismos parámetros repiten exactamente la trama.
    rng = np.random.default_rng(semilla)
    k = int(log2(orden))
    bits_tx = rng.integers(0, 2, numero_simbolos * k, dtype=np.uint8)
    # Cadena física del enlace: transmisor → modulador → canal → receptor.
    simbolos, basebanda, pulso, retardo = transmisor(
        bits_tx, esquema, orden, rolloff, sps, span
    )
    pasabanda, portadora = modulador_pasabanda(basebanda, sps)
    rx_pb, _ = canal_awgn(pasabanda, ebno_db, k, rng)
    bits_rx, muestras, salida = receptor(
        rx_pb, portadora, pulso, numero_simbolos, sps, retardo, esquema, orden
    )
    retardo_total = 2 * retardo
    # Energía de referencia de la constelación normalizada: Es=1 y Eb=Es/k.
    energia_s = 1.0
    energia_b = energia_s / k
    # Ancho nulo a nulo pasabanda B=(1+α)Rs; aquí Rs=1, así que B=1+α.
    ancho_banda = 1 + rolloff
    # La señal real se muestrea a sps muestras/símbolo; multiplicar su
    # potencia media por sps da la potencia por símbolo en unidades normalizadas.
    potencia = float(np.mean(pasabanda**2) * sps)
    return Resultado(
        esquema, orden, rolloff, ebno_db, bits_tx, bits_rx, simbolos, muestras,
        basebanda, rx_pb, salida, sps, retardo_total, energia_s, energia_b,
        ancho_banda, potencia,
    )


def qfunc(x: float | np.ndarray) -> float | np.ndarray:
    """Evalúa Q(x)=0.5 erfc(x/√2) para escalar o arreglo, sin SciPy."""
    return 0.5 * np.vectorize(erfc, otypes=[float])(np.asarray(x) / sqrt(2))


def ber_teorica(esquema: str, orden: int, ebno_db: np.ndarray | float) -> np.ndarray:
    """BER teórica Gray: expresiones estándar (aprox. para M>2 en PSK/QAM)."""
    # La BER se deriva bajo AWGN y mapeo Gray. Estas expresiones permiten
    # contrastar Monte Carlo con teoría; las de PSK/QAM son aproximadas salvo BPSK.
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
        # Aproximación para M-PAM Gray, expresada directamente en Eb/N0.
        x = sqrt(6 * k / (orden**2 - 1)) * np.sqrt(gamma)
        return 2 * (orden - 1) / (orden * k) * qfunc(x)
    if esquema == "M-PSK":
        if orden == 2:
            # BPSK tiene BER exacta Q(sqrt(2γb)).
            return qfunc(np.sqrt(2 * gamma))
        # Aproximación de Gray para M-PSK con M>2.
        x = np.sqrt(2 * k * gamma) * np.sin(pi / orden)
        return 2 / k * qfunc(x)
    # Aproximación usual para constelaciones QAM cuadradas con Gray por eje.
    raiz = sqrt(orden)
    x = np.sqrt(3 * k * gamma / (orden - 1))
    return (4 / k) * (1 - 1 / raiz) * qfunc(x)


def ebno_requerido(esquema: str, orden: int, ber_objetivo: float) -> float:
    """Invierte la BER teórica para hallar Eb/N0 que alcanza el objetivo."""
    # La BER decrece al aumentar Eb/N0. La búsqueda acota el valor en dB
    # hasta encontrar el punto donde la teoría alcanza la BER solicitada.
    if not 0 < ber_objetivo < 0.5:
        raise ValueError("La BER objetivo debe estar entre 0 y 0.5, sin incluir los extremos.")
    bajo, alto = -10.0, 50.0
    for _ in range(80):
        # Si la BER calculada es demasiado alta, se aumenta Eb/N0; de lo
        # contrario se reduce el extremo superior del intervalo de búsqueda.
        medio = (bajo + alto) / 2
        if float(ber_teorica(esquema, orden, medio)) > ber_objetivo:
            bajo = medio
        else:
            alto = medio
    return (bajo + alto) / 2
