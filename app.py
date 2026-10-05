"""Interfaz interactiva del proyecto de comunicaciones digitales."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulador import ber_teorica, ebno_requerido, simular


# Configuración general y título de la aplicación Streamlit.
st.set_page_config(page_title="Sistema de comunicación digital", page_icon="📡", layout="wide")
st.title("Simulación de un sistema de comunicación digital")
st.caption("M-PAM · M-PSK · M-QAM | Transmisor, canal AWGN y receptor coherente")
st.markdown("**Flujo de señal:** Bits → mapeo Gray → pulso RRC → portadora → canal AWGN → demodulación coherente → filtro adaptado → decisión y demapeo")

# Los controles laterales exponen los parámetros funcionales del requerimiento.
with st.sidebar:
    st.header("Parámetros")
    esquema = st.selectbox("Esquema de modulación", ["M-PAM", "M-QAM", "M-PSK"])
    opciones = {"M-PAM": [2, 4, 8, 16], "M-QAM": [4, 16, 64], "M-PSK": [2, 4, 8, 16]}
    orden = st.selectbox("Orden M", opciones[esquema], index=opciones[esquema].index(4) if 4 in opciones[esquema] else 0)
    rolloff = st.slider("Factor roll-off", 0.0, 1.0, 0.25, 0.05)
    ebno_db = st.slider("Eb/N₀ de la simulación (dB)", -2.0, 20.0, 8.0, 0.5)
    numero_simbolos = st.select_slider("Número de símbolos", options=[256, 512, 1024, 2048, 4096, 8192], value=1024)
    semilla = st.number_input("Semilla aleatoria", min_value=0, max_value=2**31 - 1, value=2026, step=1)


@st.cache_data(show_spinner=False, max_entries=32)
def correr(esquema, orden, rolloff, ebno_db, numero_simbolos, semilla):
    # Streamlit guarda resultados para no repetir un cálculo idéntico.
    return simular(esquema, orden, rolloff, ebno_db, numero_simbolos, semilla)


@st.cache_data(show_spinner=False, max_entries=32)
def barrer_ebno(esquema, orden, rolloff, numero_simbolos, semilla):
    # Ejecuta simulaciones Monte Carlo independientes para trazar la BER
    # medida a varios Eb/N0 y compararla con la curva teórica.
    puntos = np.arange(-2.0, 20.01, 4.0)
    bers = []
    for indice, ebno in enumerate(puntos):
        resultado = simular(esquema, orden, rolloff, ebno, numero_simbolos, semilla + indice + 1)
        bers.append(resultado.ber)
    return puntos, np.asarray(bers)


# Streamlit vuelve a ejecutar el script al cambiar un control. El caché evita
# repetir una simulación cuando los parámetros no cambiaron, así la interacción
# actualiza la vista sin un botón de ejecución ni resultados obsoletos.
try:
    with st.spinner("Actualizando la simulación y las gráficas…"):
        resultado = correr(esquema, orden, rolloff, ebno_db, numero_simbolos, int(semilla))
except ValueError as error:
    st.error(f"No se pudo ejecutar la simulación: {error}")
    resultado = None

if resultado is not None:
    st.caption(f"Simulación actual: {resultado.esquema}, M={resultado.orden}, roll-off={resultado.rolloff:.2f}, Eb/N₀={resultado.ebno_db:.1f} dB, {numero_simbolos:,} símbolos, semilla={int(semilla)}.")
    met = st.columns(6)
    met[0].metric("BER simulada", f"{resultado.ber:.3e}")
    met[1].metric("Energía por símbolo", f"{resultado.energia_simbolo:.3f}")
    met[2].metric("Energía por bit", f"{resultado.energia_bit:.3f}")
    met[3].metric("Ancho pasabanda", f"{resultado.ancho_banda:.3f} Rs")
    met[4].metric("Eficiencia espectral", f"{np.log2(resultado.orden) / (1 + resultado.rolloff):.3f} bit/s/Hz")
    met[5].metric("Potencia pasabanda", f"{resultado.potencia_pasabanda:.3f}")

    # Curva teórica y curva simulada; el piso visual evita ceros en escala log.
    st.subheader("Desempeño BER frente a Eb/N₀")
    ebno_axis = np.arange(-2.0, 20.01, 0.25)
    ber_axis = ber_teorica(resultado.esquema, resultado.orden, ebno_axis)
    with st.spinner("Calculando puntos BER simulados…"):
        simbolos_barrido = resultado.bits_tx.size // int(np.log2(resultado.orden))
        x_sim, y_sim = barrer_ebno(resultado.esquema, resultado.orden, resultado.rolloff, simbolos_barrido, int(semilla))
    fig_ber = go.Figure()
    fig_ber.add_trace(go.Scatter(x=ebno_axis, y=np.maximum(ber_axis, 1e-8), mode="lines", name="BER teórica (aproximación)", line=dict(color="#1565C0", width=3)))
    bits_barrido = simbolos_barrido * int(np.log2(resultado.orden))
    y_sim_vis = np.maximum(y_sim, 0.5 / bits_barrido)
    fig_ber.add_trace(go.Scatter(x=x_sim, y=y_sim_vis, mode="lines+markers", name="BER simulada", line=dict(color="#E65100", width=2, dash="dash"), marker=dict(size=8)))
    fig_ber.add_trace(go.Scatter(x=[resultado.ebno_db], y=[max(resultado.ber, 0.5 / len(resultado.bits_tx))], mode="markers", name=f"Simulación actual ({resultado.ebno_db:.1f} dB)", marker=dict(color="#111111", size=12, symbol="diamond")))
    fig_ber.update_layout(xaxis_title="Eb/N₀ (dB)", yaxis_title="BER", yaxis_type="log", height=400, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig_ber, width="stretch")
    st.caption(f"La curva teórica usa aproximaciones estándar para constelaciones Gray. Los puntos simulados usan la semilla seleccionada y {simbolos_barrido:,} símbolos por Eb/N₀; si no hubo errores, se muestra el límite visual de 0.5/Nbits ({0.5 / bits_barrido:.2e}).")
    st.caption("El roll-off modifica el pulso, el espectro, el ojo y el ancho de banda. En un sistema ideal con filtro adaptado no cambia la BER teórica; la BER simulada puede variar ligeramente por el filtrado finito y el ruido aleatorio.")
    st.caption("La resolución Monte Carlo está limitada por la cantidad finita de bits. Para BER objetivo menores que ese límite, usa la comparación teórica de abajo; no se interpreta un punto sin errores como BER cero.")

    # Diagrama de ojo: superpone ventanas de dos símbolos de la salida del
    # filtro adaptado para mostrar interferencia entre símbolos y apertura.
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Diagrama de ojo en recepción")
        sps = resultado.muestras_por_simbolo
        salida = resultado.salida_filtro
        trazas = []
        for i in range(10, min(100, len(resultado.simbolos_tx) - 1)):
            centro = resultado.retardo_total + i * sps
            tramo = salida[centro - sps : centro + sps + 1]
            if len(tramo) == 2 * sps + 1:
                trazas.append(go.Scatter(x=np.linspace(-1, 1, len(tramo)), y=np.real(tramo), mode="lines", line=dict(color="#1565C0", width=1), opacity=0.25, showlegend=False))
        ojo = go.Figure(trazas)
        ojo.update_layout(xaxis_title="Tiempo normalizado (Tₛ)", yaxis_title="Amplitud del filtro adaptado", height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(ojo, width="stretch")
    # Constelación: dibuja las muestras recibidas alrededor de sus regiones
    # de decisión (la distancia más cercana determina el símbolo estimado).
    with c2:
        st.subheader("Constelación en recepción")
        const = go.Figure()
        const.add_trace(go.Scatter(x=np.real(resultado.simbolos_rx), y=np.imag(resultado.simbolos_rx), mode="markers", marker=dict(size=6, opacity=0.55, color="#00897B"), name="Muestras recibidas"))
        const.update_layout(xaxis_title="Componente en fase (I)", yaxis_title="Componente en cuadratura (Q)", yaxis=dict(scaleanchor="x", scaleratio=1), height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(const, width="stretch")

    # FFT de señales discretas: frecuencia expresada en múltiplos de Rs;
    # los niveles se muestran relativos para comparar la forma del espectro.
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Espectro de transmisión en banda base")
        señal = resultado.senal_basebanda
        nfft = 2 ** int(np.ceil(np.log2(max(len(señal), 256))))
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / resultado.muestras_por_simbolo))
        potencia = np.fft.fftshift(np.abs(np.fft.fft(señal, nfft)) ** 2 / nfft)
        base_fig = go.Figure(go.Scatter(x=freqs, y=10 * np.log10(np.maximum(potencia, 1e-12)), mode="lines", line=dict(color="#5E35B1")))
        base_fig.update_layout(xaxis_title="Frecuencia (Rs)", yaxis_title="Potencia (dB, escala relativa)", height=330, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(base_fig, width="stretch")
    with c4:
        st.subheader("Espectro de recepción en pasabanda")
        señal = resultado.senal_pasabanda_rx
        nfft = 2 ** int(np.ceil(np.log2(max(len(señal), 256))))
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / resultado.muestras_por_simbolo))
        potencia = np.fft.fftshift(np.abs(np.fft.fft(señal, nfft)) ** 2 / nfft)
        pb_fig = go.Figure(go.Scatter(x=freqs, y=10 * np.log10(np.maximum(potencia, 1e-12)), mode="lines", line=dict(color="#EF6C00")))
        pb_fig.update_layout(xaxis_title="Frecuencia (Rs)", yaxis_title="Potencia (dB, escala relativa)", height=330, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(pb_fig, width="stretch")

# Comparación exigida en el documento: Eb/N0 para seis BER objetivo y sus
# diferencias por pares, primero en tabla y luego en dos gráficas de análisis.
st.divider()
st.subheader("Comparación teórica: 4-PAM, 4-QAM y 4-PSK")
st.write("Eb/N₀ requerido (dB) para alcanzar cada BER objetivo, calculado con las aproximaciones teóricas de cada esquema. La gráfica permite comparar los valores y sus diferencias.")
objetivos = [10.0**(-i) for i in range(2, 8)]
filas = []
for objetivo in objetivos:
    pam = ebno_requerido("M-PAM", 4, objetivo)
    qam = ebno_requerido("M-QAM", 4, objetivo)
    psk = ebno_requerido("M-PSK", 4, objetivo)
    filas.append({"BER objetivo": f"{objetivo:.0e}", "4-PAM (dB)": pam, "4-QAM (dB)": qam, "4-PSK (dB)": psk, "4-PAM − 4-QAM (dB)": pam - qam, "4-PSK − 4-QAM (dB)": psk - qam, "4-QAM − 4-PAM (dB)": qam - pam})
tabla = pd.DataFrame(filas)
st.dataframe(tabla.style.format({c: "{:.2f}" for c in tabla.columns if c != "BER objetivo"}), width="stretch", hide_index=True)
st.caption("Los diferenciales comparan el Eb/N₀ requerido para la misma BER. Un valor positivo indica que el esquema de la izquierda requiere más Eb/N₀.")

# Gráfica izquierda: Eb/N0 requerido. Gráfica derecha: brecha con signo entre
# esquemas; arriba de cero el primer esquema necesita más Eb/N0.
comparacion, diferencias = st.columns(2)
with comparacion:
    fig_req = go.Figure()
    for columna, nombre, color in [
        ("4-PAM (dB)", "4-PAM", "#E65100"),
        ("4-QAM (dB)", "4-QAM", "#1565C0"),
        ("4-PSK (dB)", "4-PSK", "#00897B"),
    ]:
        fig_req.add_trace(go.Scatter(x=tabla["BER objetivo"], y=tabla[columna], mode="lines+markers", name=nombre, line=dict(color=color, width=2)))
    fig_req.update_layout(title="Eb/N₀ requerido por BER", xaxis_title="BER objetivo", yaxis_title="Eb/N₀ (dB)", height=360, margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig_req, width="stretch")
with diferencias:
    fig_dif = go.Figure()
    for columna, nombre, color in [
        ("4-PAM − 4-QAM (dB)", "4-PAM − 4-QAM", "#E65100"),
        ("4-PSK − 4-QAM (dB)", "4-PSK − 4-QAM", "#00897B"),
        ("4-QAM − 4-PAM (dB)", "4-QAM − 4-PAM", "#1565C0"),
    ]:
        fig_dif.add_trace(go.Scatter(x=tabla["BER objetivo"], y=tabla[columna], mode="lines+markers", name=nombre, line=dict(color=color, width=2)))
    fig_dif.add_hline(y=0, line_dash="dot", line_color="#616161")
    fig_dif.update_layout(title="Diferencia de Eb/N₀ entre esquemas", xaxis_title="BER objetivo", yaxis_title="Diferencia (dB)", height=360, margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig_dif, width="stretch")

st.markdown("**Lectura del resultado:** 4-QAM y 4-PSK tienen la misma BER teórica porque sus constelaciones de cuatro puntos son equivalentes salvo una rotación. La 4-PAM es unidimensional, por lo que su separación entre símbolos con energía media normalizada difiere; la brecha se aprecia en la gráfica y en la tabla.")

with st.expander("Modelo matemático y supuestos"):
    st.markdown(r"""
    - Bits por símbolo: $k=\log_2(M)$.
    - La constelación se normaliza a energía media $E_s=1$; por ello $E_b=E_s/k$.
    - Se expresa la relación de energía en escala lineal como $\gamma_b=10^{(E_b/N_0)_{dB}/10}$. En el modelo discreto normalizado, la varianza de cada muestra real de ruido es $\sigma_n^2=E_b/(2\gamma_b)$.
    - La BER simulada se calcula como $N_{errores}/N_{bits}$. La línea teórica usa la expresión Gray de M-PAM y las aproximaciones usuales de M-PSK/M-QAM; para BPSK se usa $Q(\sqrt{2\gamma_b})$.
    - Con pulso coseno alzado, $B=(1+\alpha)R_s$ y la eficiencia espectral es $R_b/B=k/(1+\alpha)$ bit/s/Hz. Aquí $R_s=1$ se usa como unidad normalizada.
    - La potencia de pasabanda se estima promediando la señal modulada y se expresa en unidades normalizadas, no en vatios.
    """)
