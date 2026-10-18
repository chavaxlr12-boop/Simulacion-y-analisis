# Análisis de señales con la Transformada de Fourier (Python)

**Autor:** Salvador Muñoz Hernández

Este repositorio analiza señales en el **dominio del tiempo** y en el **dominio de la frecuencia** con la Transformada de Fourier, calculada con `np.fft.fft()`. Hace cuatro cosas:

1. Crea tres señales elementales (pulso rectangular, escalón unitario y senoidal) y grafica la **magnitud y la fase** de su espectro.
2. Compara el espectro calculado con la FFT contra la **transformada teórica** de cada señal.
3. Verifica numéricamente las propiedades de **linealidad**, **desplazamiento en el tiempo** y **escalamiento**, más dos extras: modulación y teorema de Parseval.
4. Aplica la transformada a dos **problemas reales**: quitar el ruido de 60 Hz de un electrocardiograma (ECG) y decodificar los tonos de un teléfono (DTMF).

## Estructura

```
analisis-fourier/
├── main.py                 # Ejecuta todo y genera resultados.txt
├── utilidades.py           # Definición de señales, FFT y funciones de graficación
├── senales_elementales.py  # Parte 1: pulso, escalón y senoidal
├── propiedades.py          # Parte 2: linealidad, desplazamiento, escalamiento (+ modulación, Parseval)
├── aplicaciones.py         # Parte 3: filtrado de 60 Hz en ECG y decodificación DTMF
├── figuras/                # Gráficas generadas (PNG)
├── resultados.txt          # Resumen numérico de todas las verificaciones
└── requirements.txt
```

Cada script sigue la misma estructura, tal como pide la actividad:
**definición de la señal → cálculo de la transformada → visualización → análisis de propiedades**.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py
```

También se puede correr cada parte por separado, por ejemplo con `python propiedades.py`.

## Cómo se calcula la transformada

La FFT calcula la DFT: `X[k] = Σ x[n]·e^(−j2πkn/N)`. Para que se parezca a la transformada **continua** `X(f) = ∫ x(t)·e^(−j2πft) dt`, el código hace tres ajustes:

- Multiplica por `Δt = 1/fs`, lo que convierte la suma en una aproximación de la integral.
- Multiplica por `e^(−j2πf·t₀)` porque la ventana empieza en `t₀ = −1 s` y no en 0. Esta corrección es la propia propiedad de desplazamiento.
- Aplica `np.fft.fftshift` para que la frecuencia 0 quede al centro de la gráfica.

En las senoidales usa en cambio `|X[k]|/N`. Así cada pico mide `A/2` y la gráfica se lee directamente como un espectro de amplitud.

Parámetros principales: `fs = 1000 Hz` y una ventana de 2 s, con lo que `N = 2000` y la resolución en frecuencia es `Δf = 0.5 Hz`.

## Resultados principales

| Señal / propiedad | Teoría | Resultado numérico |
|---|---|---|
| Pulso rect. T = 0.2 s | `T·sinc(fT)`, X(0) = 0.2, ceros cada 5 Hz | X(0) = 0.2; error vs teoría 0.3 % |
| Escalón u(t) | `½δ(f) + 1/(j2πf)` | Fase −π/2 en f > 0 y +π/2 en f < 0; la magnitud decae como 1/f |
| Senoidal 50 Hz | picos de altura A/2 en ±50 Hz | Pico en 50 Hz con altura 0.5 y fase −π/2 |
| Linealidad | `aX₁ + bX₂` | error 1.7·10⁻¹⁶ |
| Desplazamiento t₀ = 0.3 s | `X(f)·e^(−j2πft₀)` | error 6.2·10⁻¹⁶; la magnitud no cambia |
| Escalamiento a = 0.5, 2 | `(1/|a|)·X(f/a)` | error < 0.5 % |
| Modulación fc = 20 Hz | `½[X(f−fc) + X(f+fc)]` | error 0.03 % |
| Parseval | `Σ|x|² = (1/N)·Σ|X|²` | 449.5 = 449.5 |
| ECG + ruido 60 Hz | quitar los bins de 60 y 120 Hz | SNR: −3.8 dB → 16.2 dB |
| DTMF «7391» | dos picos por tecla | decodificado: «7391» |

Un error del orden de 10⁻¹⁶ significa que la propiedad se cumple exactamente, al límite de precisión de la computadora. El error de ~0.3 % en el escalamiento y en la comparación con la teoría viene de muestrear los bordes del pulso.

## Figuras

| | |
|---|---|
| ![](figuras/01_pulso_rectangular.png) | ![](figuras/02_escalon_unitario.png) |
| ![](figuras/03_senoidal.png) | ![](figuras/04_linealidad.png) |
| ![](figuras/05_desplazamiento_tiempo.png) | ![](figuras/06_escalamiento.png) |
| ![](figuras/07_modulacion.png) | ![](figuras/08_aplicacion_ecg_60hz.png) |
| ![](figuras/09_aplicacion_dtmf.png) | |

## Requisitos

Python 3.10 o más reciente con NumPy, SciPy y Matplotlib. Ver `requirements.txt`.
