"""
aplicaciones.py
===============
Parte 3 de la actividad: aplicaciones reales de la Transformada de Fourier.

Aplicación 1 - Eliminación de interferencia de la red eléctrica en un ECG
    Los electrocardiógramas captan el "zumbido" de la red eléctrica
    (60 Hz en México y América; 50 Hz en Europa). En el tiempo el ruido se
    mezcla con los latidos y es difícil de separar, pero en frecuencia
    aparece como un pico aislado en 60 Hz (y su armónico en 120 Hz).
    Se anulan esos bins de la FFT y se reconstruye la señal con la IFFT.

Aplicación 2 - Decodificación de tonos telefónicos DTMF
    Al presionar una tecla del teléfono se suman dos senoidales: una del
    grupo bajo (697, 770, 852, 941 Hz) y otra del grupo alto (1209, 1336,
    1477, 1633 Hz). En el tiempo la señal parece "desordenada", pero su
    espectro muestra exactamente dos picos que identifican la tecla.

Ejecutar:  python aplicaciones.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from utilidades import CARPETA_FIGURAS, AZUL, NARANJA, VERDE, ROJO

rng = np.random.default_rng(seed=7)   # semilla fija -> resultados repetibles


# ---------------------------------------------------------------------------
# APLICACIÓN 1: FILTRADO DE 60 Hz EN UN ECG
# ---------------------------------------------------------------------------

def ecg_sintetico(t, ritmo_bpm=72):
    """Genera un ECG sintético sumando gaussianas para las ondas P, Q, R, S y T
    de cada latido (modelo simplificado, solo para fines didácticos)."""
    periodo = 60 / ritmo_bpm
    #          amplitud, retraso respecto a R [s], ancho [s]
    ondas = [(0.15, -0.20, 0.025),   # onda P
             (-0.12, -0.035, 0.010),  # onda Q
             (1.00, 0.00, 0.012),     # onda R
             (-0.25, 0.035, 0.010),   # onda S
             (0.30, 0.25, 0.040)]     # onda T
    ecg = np.zeros_like(t)
    for centro_r in np.arange(0.3, t[-1] + periodo, periodo):
        for amp, desfase, ancho in ondas:
            ecg += amp * np.exp(-((t - centro_r - desfase) ** 2) / (2 * ancho ** 2))
    return ecg


def filtrar_ecg(fs=500, duracion=4.0, f_red=60.0, ancho_banda=1.0):
    t = np.arange(0, duracion, 1 / fs)
    limpio = ecg_sintetico(t)

    # Ruido: 60 Hz de la red + armónico de 120 Hz + ruido blanco del sensor
    interferencia = (0.40 * np.sin(2 * np.pi * f_red * t)
                     + 0.10 * np.sin(2 * np.pi * 2 * f_red * t + 0.7))
    ruidoso = limpio + interferencia + 0.03 * rng.standard_normal(len(t))

    # 1) Al dominio de la frecuencia
    X = np.fft.fft(ruidoso)
    f = np.fft.fftfreq(len(t), 1 / fs)

    # 2) Filtro "notch" ideal: se anulan los bins cercanos a ±60 y ±120 Hz
    mascara = np.ones(len(f), dtype=bool)
    for fr in (f_red, 2 * f_red):
        mascara &= np.abs(np.abs(f) - fr) > ancho_banda
    X_filtrado = X * mascara

    # 3) De regreso al tiempo con la transformada inversa
    filtrado = np.real(np.fft.ifft(X_filtrado))

    def snr_db(senal, referencia):
        ruido = senal - referencia
        return 10 * np.log10(np.sum(referencia ** 2) / np.sum(ruido ** 2))

    snr_antes, snr_despues = snr_db(ruidoso, limpio), snr_db(filtrado, limpio)

    # --- Gráficas
    fig, ejes = plt.subplots(3, 1, figsize=(9, 8.5))
    fig.suptitle("Aplicación 1: eliminación del ruido de 60 Hz en un ECG",
                 fontsize=13, fontweight="bold")
    zoom = t < 2
    ejes[0].plot(t[zoom], ruidoso[zoom], color=ROJO, lw=0.8, label="ECG con ruido")
    ejes[0].plot(t[zoom], limpio[zoom], color="k", lw=1, alpha=.6, label="ECG original")
    ejes[0].set_title(f"(a) Tiempo: señal contaminada (SNR = {snr_antes:.1f} dB)",
                      loc="left", fontsize=10)
    ejes[0].set_xlabel("Tiempo t [s]")
    ejes[0].set_ylabel("mV")
    ejes[0].legend(fontsize=8, loc="upper right")

    pos = f >= 0
    mag_db = 20 * np.log10(np.abs(X[pos]) / len(t) + 1e-12)
    ejes[1].plot(f[pos], mag_db, color=NARANJA, lw=0.9)
    for fr in (f_red, 2 * f_red):
        ejes[1].axvspan(fr - ancho_banda, fr + ancho_banda, color=ROJO, alpha=.2)
        ejes[1].annotate(f"{fr:.0f} Hz", xy=(fr, mag_db[np.argmin(abs(f[pos] - fr))]),
                         xytext=(fr + 8, -10), fontsize=9, color=ROJO,
                         arrowprops=dict(arrowstyle="->", color=ROJO))
    ejes[1].set_xlim(0, 200)
    ejes[1].set_ylim(-90, 0)
    ejes[1].set_title("(b) Frecuencia: el ECG ocupa < 40 Hz; la red eléctrica es "
                      "un pico aislado (zona sombreada = bins eliminados)",
                      loc="left", fontsize=10)
    ejes[1].set_xlabel("Frecuencia f [Hz]")
    ejes[1].set_ylabel("|X(f)| [dB]")

    ejes[2].plot(t[zoom], filtrado[zoom], color=VERDE, lw=1.1, label="ECG filtrado (IFFT)")
    ejes[2].plot(t[zoom], limpio[zoom], "k--", lw=0.8, label="ECG original")
    ejes[2].set_title(f"(c) Tiempo: señal recuperada (SNR = {snr_despues:.1f} dB)",
                      loc="left", fontsize=10)
    ejes[2].set_xlabel("Tiempo t [s]")
    ejes[2].set_ylabel("mV")
    ejes[2].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    ruta = CARPETA_FIGURAS / "08_aplicacion_ecg_60hz.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta, snr_antes, snr_despues


# ---------------------------------------------------------------------------
# APLICACIÓN 2: DECODIFICACIÓN DTMF
# ---------------------------------------------------------------------------

FILAS = [697, 770, 852, 941]          # grupo bajo [Hz]
COLUMNAS = [1209, 1336, 1477, 1633]   # grupo alto [Hz]
TECLADO = ["123A", "456B", "789C", "*0#D"]


def tono_dtmf(tecla, fs, duracion):
    """Suma de dos senoidales correspondiente a una tecla del teclado."""
    for i, fila in enumerate(TECLADO):
        if tecla in fila:
            f_baja, f_alta = FILAS[i], COLUMNAS[fila.index(tecla)]
    t = np.arange(0, duracion, 1 / fs)
    return np.sin(2 * np.pi * f_baja * t) + np.sin(2 * np.pi * f_alta * t)


def detectar_tecla(segmento, fs):
    """Busca el pico más fuerte de cada grupo en el espectro del segmento."""
    X = np.abs(np.fft.rfft(segmento * np.hanning(len(segmento))))
    f = np.fft.rfftfreq(len(segmento), 1 / fs)
    energia = lambda f0: X[np.argmin(np.abs(f - f0))]
    fila = int(np.argmax([energia(fr) for fr in FILAS]))
    col = int(np.argmax([energia(fc) for fc in COLUMNAS]))
    return TECLADO[fila][col], f, X


def decodificar_dtmf(numero="7391", fs=8000, t_tono=0.08, t_pausa=0.04):
    # Construcción de la señal: tono - silencio - tono - ... + ruido
    partes = []
    for tecla in numero:
        partes += [tono_dtmf(tecla, fs, t_tono), np.zeros(int(t_pausa * fs))]
    x = np.concatenate(partes)
    x = x + 0.3 * rng.standard_normal(len(x))
    t = np.arange(len(x)) / fs

    # Decodificación: se toma la FFT de cada ventana donde hay tono
    n_tono, n_ciclo = int(t_tono * fs), int((t_tono + t_pausa) * fs)
    detectado, espectros = "", []
    for i in range(len(numero)):
        segmento = x[i * n_ciclo: i * n_ciclo + n_tono]
        tecla, f, X = detectar_tecla(segmento, fs)
        detectado += tecla
        espectros.append((f, X))

    # --- Gráficas
    fig = plt.figure(figsize=(9, 8.5))
    fig.suptitle(f"Aplicación 2: decodificación DTMF del número «{numero}»",
                 fontsize=13, fontweight="bold")
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, x, color=AZUL, lw=0.4)
    for i, tecla in enumerate(numero):
        ax1.text((i * n_ciclo + n_tono / 2) / fs, 2.9, tecla, ha="center",
                 fontsize=12, fontweight="bold")
    ax1.set_ylim(-3.2, 3.6)
    ax1.set_title("(a) Tiempo: cuatro tonos con ruido (no se distingue qué tecla es)",
                  loc="left", fontsize=10)
    ax1.set_xlabel("Tiempo t [s]")

    ax2 = fig.add_subplot(3, 1, 2)
    f0, X0 = espectros[0]
    ax2.plot(f0, X0 / X0.max(), color=NARANJA, lw=1)
    for fr in FILAS + COLUMNAS:
        ax2.axvline(fr, color="gray", ls=":", lw=0.8)
    ax2.set_xlim(500, 1800)
    ax2.set_title(f"(b) Espectro del primer tono: dos picos → tecla «{numero[0]}» "
                  "(líneas punteadas = frecuencias DTMF)", loc="left", fontsize=10)
    ax2.set_xlabel("Frecuencia f [Hz]")
    ax2.set_ylabel("|X(f)| normalizada")

    ax3 = fig.add_subplot(3, 1, 3)
    fs_e, ts_e, S = signal.spectrogram(x, fs, nperseg=256, noverlap=192)
    ax3.pcolormesh(ts_e, fs_e, 10 * np.log10(S + 1e-12), shading="gouraud",
                   cmap="magma", vmin=-60)
    ax3.set_ylim(500, 1800)
    ax3.set_title("(c) Espectrograma (FFT por ventanas): frecuencias vs tiempo",
                  loc="left", fontsize=10)
    ax3.set_xlabel("Tiempo t [s]")
    ax3.set_ylabel("f [Hz]")
    ax3.grid(False)

    fig.tight_layout()
    ruta = CARPETA_FIGURAS / "09_aplicacion_dtmf.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta, numero, detectado


def ejecutar():
    return {"ecg": filtrar_ecg(), "dtmf": decodificar_dtmf()}


if __name__ == "__main__":
    r = ejecutar()
    _, antes, despues = r["ecg"]
    print(f"ECG: SNR antes = {antes:.1f} dB, después = {despues:.1f} dB")
    _, original, detectado = r["dtmf"]
    print(f"DTMF: marcado = {original}, decodificado = {detectado}")
