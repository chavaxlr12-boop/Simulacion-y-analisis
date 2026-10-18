"""
senales_elementales.py
======================
Parte 1 de la actividad: señales elementales y su espectro.

Para cada señal (pulso rectangular, escalón unitario y senoidal):
    1. Se define la señal en el dominio del tiempo.
    2. Se calcula su Transformada de Fourier con np.fft.fft().
    3. Se grafican la señal, la magnitud y la fase del espectro.
    4. Se compara el resultado numérico con la transformada teórica.

Ejecutar:  python senales_elementales.py
"""

import numpy as np

from utilidades import (eje_tiempo, pulso_rectangular, escalon_unitario,
                        senoidal, transformada_fourier,
                        graficar_tiempo_y_espectro, error_relativo)

FS = 1000          # frecuencia de muestreo [Hz]
T_INICIO = -1.0    # la ventana de análisis va de -1 s a 1 s
T_FIN = 1.0


def analizar_pulso(ancho=0.2):
    """Pulso rectangular de ancho T centrado en t = 0.

    Teoría:  rect(t/T)  <->  T · sinc(f T),   con sinc(u) = sin(πu)/(πu)
      - Valor máximo en f = 0: X(0) = T (el área del pulso).
      - Ceros en f = ±1/T, ±2/T, ...  (cada 5 Hz para T = 0.2 s).
      - Al ser real y par, X(f) es real: la fase solo vale 0 o ±π
        (cambia de signo en cada lóbulo del sinc).
    """
    # --- Definición de la señal en el dominio del tiempo
    t = eje_tiempo(FS, T_INICIO, T_FIN)
    x = pulso_rectangular(t, ancho=ancho)

    # --- Cálculo de la Transformada de Fourier
    f, X = transformada_fourier(x, FS, t_inicio=t[0])
    X_teo = ancho * np.sinc(f * ancho)

    # --- Visualización
    ruta = graficar_tiempo_y_espectro(
        t, x, f, X,
        titulo=f"Pulso rectangular (T = {ancho} s)  ↔  T·sinc(fT)",
        archivo="01_pulso_rectangular.png",
        lim_t=(-0.5, 0.5), lim_f=(-40, 40), teorico=(f, X_teo),
        etiqueta_mag="|X(f)|  [s]")

    return {
        "ruta": ruta,
        "X(0) numérico": np.abs(X[f == 0][0]),
        "X(0) teórico": ancho,
        "primer cero teórico [Hz]": 1 / ancho,
        "error relativo vs teoría": error_relativo(X, X_teo),
    }


def analizar_escalon():
    """Escalón unitario u(t).

    Teoría:  u(t)  <->  (1/2)·δ(f) + 1 / (j2πf)
      - Tiene un impulso en f = 0 (su valor medio no es cero).
      - Para f ≠ 0 la magnitud decae como 1/(2π|f|).
      - Fase de -π/2 para f > 0 y +π/2 para f < 0 (término 1/(j2πf)).

    Ojo: en la computadora la ventana es finita (de -1 a 1 s), así que en
    realidad se transforma u(t)·rect ventana = un pulso de 0 a 1 s. Por eso
    la FFT muestra un pico finito en f = 0 (no un impulso infinito) y
    oscilaciones de tipo sinc cuya envolvente sí sigue 1/(2π|f|).
    """
    t = eje_tiempo(FS, T_INICIO, T_FIN)
    x = escalon_unitario(t)

    f, X = transformada_fourier(x, FS, t_inicio=t[0])

    # Envolvente teórica |1/(j2πf)| (se evita dividir entre cero en f = 0)
    with np.errstate(divide="ignore"):
        envolvente = np.where(f != 0, 1 / (2 * np.pi * np.abs(f)), np.nan)

    ruta = graficar_tiempo_y_espectro(
        t, x, f, X,
        titulo="Escalón unitario u(t)  ↔  ½δ(f) + 1/(j2πf)",
        archivo="02_escalon_unitario.png",
        lim_t=(-1, 1), lim_f=(-10, 10), teorico=(f, envolvente),
        etiqueta_mag="|X(f)|  [s]")

    # Se mide la fase en f = 0.5 Hz (primer bin positivo): debe acercarse a
    # -π/2 en promedio; con ventana finita aparece un término adicional.
    return {
        "ruta": ruta,
        "|X(0)| (área de la ventana)": np.abs(X[f == 0][0]),
        "valor medio de x(t)": x.mean(),
    }


def analizar_senoidal(f0=50.0, amplitud=1.0):
    """Senoidal de frecuencia f0.

    Teoría:  A·sin(2πf0 t)  <->  (A / 2j)·[δ(f - f0) - δ(f + f0)]
      - Toda la energía está en dos frecuencias: ±f0.
      - Con normalización |FFT|/N cada pico mide A/2.
      - Fase: -π/2 en +f0 y +π/2 en -f0 (el seno es un coseno retrasado).
    Se eligen 2 s de ventana -> 100 ciclos exactos de 50 Hz, así no hay
    fuga espectral (leakage) y los picos caen exactamente en un bin.
    """
    t = eje_tiempo(FS, T_INICIO, T_FIN)
    x = senoidal(t, frecuencia=f0, amplitud=amplitud)

    f, X = transformada_fourier(x, FS, normalizacion="amplitud")
    # La normalización 'amplitud' no corrige el origen de tiempo; como la
    # ventana empieza en t = -1 s (múltiplo entero del periodo de 50 Hz),
    # la fase medida coincide con la teórica.

    ruta = graficar_tiempo_y_espectro(
        t, x, f, X,
        titulo=f"Senoidal (f0 = {f0:.0f} Hz, A = {amplitud})  ↔  "
               "(A/2j)[δ(f−f0) − δ(f+f0)]",
        archivo="03_senoidal.png",
        lim_t=(-0.05, 0.05), lim_f=(-100, 100),
        etiqueta_mag="|X[k]| / N")

    k_pico = np.argmax(np.abs(X) * (f > 0))
    return {
        "ruta": ruta,
        "frecuencia del pico [Hz]": f[k_pico],
        "altura del pico (teórica A/2)": np.abs(X[k_pico]),
        "fase en +f0 [rad] (teórica -π/2)": np.angle(X[k_pico]),
    }


def ejecutar():
    resultados = {
        "Pulso rectangular": analizar_pulso(),
        "Escalón unitario": analizar_escalon(),
        "Senoidal": analizar_senoidal(),
    }
    return resultados


if __name__ == "__main__":
    for nombre, datos in ejecutar().items():
        print(f"\n== {nombre} ==")
        for clave, valor in datos.items():
            print(f"  {clave}: {valor}")
