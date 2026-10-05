# Simulador de comunicaciones digitales

Aplicación educativa para analizar M-PAM, M-PSK y M-QAM con un transmisor, canal AWGN y receptor coherente implementados en Python/NumPy. No se usan funciones de modulación de alto nivel.

## Instalación y ejecución (Windows)

Desde esta carpeta, crea el entorno, instala las dependencias e inicia Streamlit:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

También puedes usar directamente `\.venv\Scripts\python.exe -m pip install -r requirements.txt` y `\.venv\Scripts\python.exe -m streamlit run app.py` sin activar el entorno. Si el puerto 8501 está ocupado, ejecuta `python -m streamlit run app.py --server.port 8765`.

## Pruebas

```powershell
\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas cubren la interfaz, el cambio de esquema, la cadena transmisor/canal/receptor, mapeo y decisión Gray, parámetros inválidos, reproducibilidad, energía del pulso RRC y cercanía entre BER teórica y Monte Carlo en una condición con suficientes errores observables.

## Cobertura de requisitos

| Requisito | Implementación |
|---|---|
| Interfaz para modificar parámetros | Controles para esquema, orden M, roll-off, Eb/N₀, cantidad de símbolos y semilla; la app recalcula al cambiar un control. |
| M-PAM, M-QAM, M-PSK | Mapeo/demapeo Gray y decisión de vecino más cercano en `simulador.py`. |
| Conformación de pulso | Pulso raíz de coseno alzado (RRC) calculado muestra a muestra. |
| Modulación/demodulación analógica | Portadora pasabanda real y demodulación coherente por mezcla. |
| Canal | Ruido AWGN real con varianza calculada a partir de Eb/N₀. |
| Gráficas | BER simulada y teórica, diagrama de ojo, constelación, espectro transmitido en banda base y espectro recibido en pasabanda. |
| Métricas | BER, eficiencia espectral, energía por símbolo, energía por bit, ancho pasabanda y potencia pasabanda. |
| Errores de entrada | Los controles limitan valores inválidos y el simulador valida esquema, orden, roll-off, Eb/N₀, tamaño, sobremuestreo, duración del pulso y semilla. |
| Simulación sin funciones de modulación de alto nivel | Se implementan explícitamente mapeo, pulso, portadora, AWGN, filtro adaptado y decisión con NumPy. |
| Reproducibilidad | La semilla controla bits y ruido; hay pruebas que comparan ejecuciones repetidas. |
| Comparación de entrega | Tabla y gráficas de Eb/N₀ requerido y diferencias para 4-PAM, 4-QAM y 4-PSK en BER de 10⁻² a 10⁻⁷. |

## Modelo, ecuaciones y convenciones

- Se normaliza la tasa de símbolo como `Rs = 1 símbolo/s` y la energía media de constelación como `Es = 1`. Para `k = log2(M)` bits por símbolo, `Eb = Es/k`.
- La relación lineal es `γb = 10^(Eb/N0 en dB / 10)`. El ruido blanco gaussiano tiene densidad espectral bilateral `N0/2`. Como el pulso adaptado tiene energía unitaria, la varianza de ruido a su salida es `N0/2 = Eb/(2γb)`; esa es la varianza usada para cada muestra real del modelo normalizado.
- La BER medida es `número de bits errados / número total de bits`. Para M-PAM Gray se usa la aproximación `Pb ≈ 2(M−1)/(M log2 M) Q(√(6 log2(M)/(M²−1) γb))`. Para M-PSK, salvo BPSK, se usa `Pb ≈ 2/log2(M) Q(√(2 log2(M) γb) sin(π/M))`. Para QAM cuadrada Gray se usa `Pb ≈ (4/log2(M))(1−1/√M) Q(√(3 log2(M)/(M−1) γb))`. BPSK usa el resultado exacto `Q(√(2γb))`.
- `Q(x) = 1/2 erfc(x/√2)`. Las expresiones de M-PSK y M-QAM para órdenes mayores son aproximaciones teóricas, no valores Monte Carlo.
- Con coseno alzado, el ancho pasabanda nulo a nulo es `B = (1+α)Rs`; la eficiencia espectral es `Rb/B = log2(M)/(1+α)` bit/s/Hz.
- La potencia pasabanda se estima promediando la señal modulada y se presenta en unidades normalizadas, no en vatios.
- La curva Monte Carlo tiene resolución finita. Si no se observan errores, la app muestra `0.5/Nbits` como límite visual, nunca como BER medida igual a cero. La tabla para objetivos hasta `10⁻⁷` invierte las aproximaciones teóricas, por lo que los valores bajos deben reportarse como teóricos salvo que se simule un número suficiente de bits.
- El roll-off modifica el pulso, el ancho de banda, el espectro y el diagrama de ojo; en un canal AWGN ideal con filtro adaptado no altera la BER teórica. Cambiarlo actualiza la simulación Monte Carlo, aunque la BER estimada puede coincidir por azar con la corrida previa.

## Organización del código

- `simulador.py`: funciones separadas para el transmisor, modulador pasabanda, canal AWGN y receptor, además de mapeo, pulso, BER y métricas.
- `app.py`: interfaz y visualizaciones.
- `tests/test_simulador.py`: pruebas de los bloques de comunicaciones y los cálculos.
- `tests/test_app.py`: pruebas headless con `streamlit.testing.v1.AppTest`.

La ejecución automática comprueba el código y la interfaz; la sustentación aún requiere que el equipo explique los bloques, las aproximaciones y la incertidumbre Monte Carlo, y analice las gráficas.
