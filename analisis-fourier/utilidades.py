"""
utilidades.py
=============
Funciones comunes para el análisis de señales con la Transformada de Fourier.

Contiene:
    1. Definición de las señales elementales en el dominio del tiempo
       (pulso rectangular, escalón unitario y senoidal).
    2. El cálculo de la Transformada de Fourier usando np.fft.fft().
    3. Funciones de graficación (señal en el tiempo + magnitud + fase).

Convención usada en todo el proyecto (frecuencia en Hz):

    X(f) = ∫ x(t) · e^{-j 2π f t} dt
"""

import sys
import numpy as np
import matplotlib

# Permite imprimir símbolos como π o ↔ en la consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

matplotlib.use("Agg")  # backend sin ventana: las figuras se guardan como PNG
import matplotlib.pyplot as plt
from pathlib import Path

# Carpeta donde se guardan todas las figuras generadas
CARPETA_FIGURAS = Path(__file__).parent / "figuras"
CARPETA_FIGURAS.mkdir(exist_ok=True)

# Estilo general de las gráficas
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

AZUL = "#1f5fa8"
NARANJA = "#d9731a"
VERDE = "#2e8b57"
ROJO = "#c0392b"


# ---------------------------------------------------------------------------
# 1. DEFINICIÓN DE SEÑALES EN EL DOMINIO DEL TIEMPO
# ---------------------------------------------------------------------------

def eje_tiempo(fs=1000, t_inicio=-1.0, t_fin=1.0):
    """Crea el vector de tiempo discreto t[n] = t_inicio + n/fs.

    fs : frecuencia de muestreo en Hz (muestras por segundo).
    Con fs = 1000 Hz y una ventana de 2 s se obtienen N = 2000 muestras y
    una resolución en frecuencia de Δf = fs/N = 0.5 Hz.
    """
    n_muestras = int(round((t_fin - t_inicio) * fs))
    return t_inicio + np.arange(n_muestras) / fs


def pulso_rectangular(t, ancho=0.2, centro=0.0, amplitud=1.0):
    """Pulso rectangular rect((t - centro)/ancho).

    Vale `amplitud` cuando |t - centro| < ancho/2 y 0 fuera.
    En los bordes exactos se usa el valor medio (amplitud/2), que es la
    convención que hace coincidir mejor la FFT con la transformada teórica.
    """
    d = np.abs(t - centro)
    borde = np.isclose(d, ancho / 2)
    x = np.where(d < ancho / 2, amplitud, 0.0)
    x[borde] = amplitud / 2
    return x


def escalon_unitario(t, t0=0.0):
    """Función escalón u(t - t0): 0 antes de t0, 1 después (0.5 en t = t0)."""
    x = np.where(t > t0, 1.0, 0.0)
    x[np.isclose(t, t0)] = 0.5
    return x


def senoidal(t, frecuencia=50.0, amplitud=1.0, fase=0.0):
    """Señal senoidal x(t) = A · sin(2π f0 t + φ)."""
    return amplitud * np.sin(2 * np.pi * frecuencia * t + fase)


# ---------------------------------------------------------------------------
# 2. CÁLCULO DE LA TRANSFORMADA DE FOURIER
# ---------------------------------------------------------------------------

def transformada_fourier(x, fs, t_inicio=0.0, normalizacion="continua"):
    """Calcula el espectro de x[n] usando np.fft.fft().

    Parámetros
    ----------
    x : arreglo con las muestras de la señal.
    fs : frecuencia de muestreo (Hz).
    t_inicio : instante de tiempo de la primera muestra x[0].
    normalizacion :
        "continua" -> aproxima la transformada de Fourier continua:
                       X(f) ≈ Δt · FFT{x} · e^{-j2π f t_inicio}
                       (unidades: amplitud·segundo). Se usa para pulsos y
                       escalones, y permite comparar con la fórmula teórica.
        "amplitud" -> |FFT{x}| / N. Para una senoidal de amplitud A aparecen
                       dos picos de altura A/2 en ±f0 (espectro de amplitud).

    Regresa
    -------
    f : eje de frecuencias en Hz, ordenado de -fs/2 a fs/2 (fftshift).
    X : espectro complejo en el mismo orden.
    """
    N = len(x)
    X = np.fft.fft(x)                     # DFT: X[k] = Σ x[n] e^{-j2πkn/N}
    f = np.fft.fftfreq(N, d=1 / fs)       # frecuencias de cada bin k (Hz)

    if normalizacion == "continua":
        # La FFT supone que la señal empieza en t = 0. Como nuestra ventana
        # empieza en t_inicio, se corrige la fase con e^{-j2πf t_inicio}
        # (propiedad de desplazamiento en el tiempo) y se multiplica por Δt
        # para aproximar la integral de la transformada continua.
        X = X / fs * np.exp(-1j * 2 * np.pi * f * t_inicio)
    elif normalizacion == "amplitud":
        X = X / N
    else:
        raise ValueError("normalizacion debe ser 'continua' o 'amplitud'")

    # fftshift reordena para que la frecuencia 0 quede al centro
    return np.fft.fftshift(f), np.fft.fftshift(X)


def fase_visible(X, umbral_relativo=1e-3):
    """Devuelve la fase de X en radianes, ocultando (NaN) los bins con
    magnitud despreciable. Donde |X| ≈ 0 la fase es solo ruido numérico
    y no tiene significado físico, así que no se grafica."""
    fase = np.angle(X)
    fase[np.abs(X) < umbral_relativo * np.max(np.abs(X))] = np.nan
    return fase


# ---------------------------------------------------------------------------
# 3. VISUALIZACIÓN
# ---------------------------------------------------------------------------

def graficar_tiempo_y_espectro(t, x, f, X, titulo, archivo,
                               lim_t=None, lim_f=None, teorico=None,
                               etiqueta_mag="|X(f)|"):
    """Figura de 3 paneles: x(t), |X(f)| y ∠X(f).

    teorico : tupla opcional (f_teo, X_teo) con la transformada analítica,
              que se dibuja punteada sobre la magnitud calculada con FFT.
    """
    fig, ejes = plt.subplots(3, 1, figsize=(9, 8.2))
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    # (a) Dominio del tiempo
    ejes[0].plot(t, x, color=AZUL, lw=1.4)
    ejes[0].set_title("(a) Dominio del tiempo", loc="left", fontsize=10)
    ejes[0].set_xlabel("Tiempo t [s]")
    ejes[0].set_ylabel("x(t)")
    if lim_t:
        ejes[0].set_xlim(lim_t)

    # (b) Magnitud del espectro
    ejes[1].plot(f, np.abs(X), color=NARANJA, lw=1.4, label="FFT (np.fft.fft)")
    if teorico is not None:
        ejes[1].plot(teorico[0], np.abs(teorico[1]), "k--", lw=1,
                     label="Teórica (analítica)")
        ejes[1].legend(loc="upper right", fontsize=8)
    ejes[1].set_title("(b) Magnitud del espectro", loc="left", fontsize=10)
    ejes[1].set_xlabel("Frecuencia f [Hz]")
    ejes[1].set_ylabel(etiqueta_mag)

    # (c) Fase del espectro
    ejes[2].plot(f, fase_visible(X), ".", color=VERDE, ms=3)
    ejes[2].set_title("(c) Fase del espectro", loc="left", fontsize=10)
    ejes[2].set_xlabel("Frecuencia f [Hz]")
    ejes[2].set_ylabel("∠X(f) [rad]")
    ejes[2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ejes[2].set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

    if lim_f:
        ejes[1].set_xlim(lim_f)
        ejes[2].set_xlim(lim_f)

    fig.tight_layout()
    ruta = CARPETA_FIGURAS / archivo
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def error_relativo(a, b):
    """Error relativo máximo ||a - b||∞ / ||b||∞ entre dos espectros."""
    return np.max(np.abs(a - b)) / np.max(np.abs(b))
