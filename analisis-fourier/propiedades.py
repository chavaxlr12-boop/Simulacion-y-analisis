"""
propiedades.py
==============
Parte 2 de la actividad: verificación numérica de las propiedades de la
Transformada de Fourier.

    1. Linealidad:                a·x1(t) + b·x2(t)  <->  a·X1(f) + b·X2(f)
    2. Desplazamiento en tiempo:  x(t - t0)          <->  X(f) · e^{-j2πf t0}
    3. Escalamiento:              x(a·t)             <->  (1/|a|) · X(f/a)
    4. (Extra) Modulación:        x(t)·cos(2πfc t)   <->  ½[X(f-fc) + X(f+fc)]
    5. (Extra) Parseval:          Σ|x[n]|²           =   (1/N) Σ|X[k]|²

Cada propiedad se verifica de dos formas: con una gráfica y con un número
(el error relativo entre ambos lados de la igualdad, que debe ser ~0).

Ejecutar:  python propiedades.py
"""

import numpy as np
import matplotlib.pyplot as plt

from utilidades import (eje_tiempo, pulso_rectangular, senoidal,
                        transformada_fourier, error_relativo, fase_visible,
                        CARPETA_FIGURAS, AZUL, NARANJA, VERDE, ROJO)

FS = 1000
T = 0.2                                   # ancho base del pulso [s]
t = eje_tiempo(FS, -1.0, 1.0)


def _guardar(fig, nombre):
    fig.tight_layout()
    ruta = CARPETA_FIGURAS / nombre
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# 1. LINEALIDAD
# ---------------------------------------------------------------------------
def verificar_linealidad(a=2.0, b=0.5):
    """z(t) = a·x1(t) + b·x2(t) con x1 = pulso y x2 = senoidal de 30 Hz.

    Se calcula la FFT de z directamente y se compara con a·X1 + b·X2.
    """
    x1 = pulso_rectangular(t, ancho=T)
    x2 = senoidal(t, frecuencia=30)
    z = a * x1 + b * x2

    f, X1 = transformada_fourier(x1, FS, t[0])
    _, X2 = transformada_fourier(x2, FS, t[0])
    _, Z = transformada_fourier(z, FS, t[0])
    Z_lineal = a * X1 + b * X2               # lado derecho de la propiedad

    fig, ejes = plt.subplots(2, 1, figsize=(9, 6))
    fig.suptitle(f"Linealidad: {a}·x1(t) + {b}·x2(t)  ↔  {a}·X1(f) + {b}·X2(f)",
                 fontsize=13, fontweight="bold")
    ejes[0].plot(t, a * x1, color=AZUL, lw=1, alpha=.6, label=f"{a}·x1 (pulso)")
    ejes[0].plot(t, b * x2, color=VERDE, lw=1, alpha=.6, label=f"{b}·x2 (seno 30 Hz)")
    ejes[0].plot(t, z, color="k", lw=1.2, label="z(t) = suma")
    ejes[0].set_xlim(-0.3, 0.3)
    ejes[0].set_xlabel("Tiempo t [s]")
    ejes[0].legend(fontsize=8, loc="upper right")
    ejes[0].set_title("(a) Dominio del tiempo", loc="left", fontsize=10)

    ejes[1].plot(f, np.abs(Z), color=NARANJA, lw=2.5, label="|FFT{z}|")
    ejes[1].plot(f, np.abs(Z_lineal), "k--", lw=1, label=f"|{a}·X1 + {b}·X2|")
    ejes[1].set_xlim(-50, 50)
    ejes[1].set_xlabel("Frecuencia f [Hz]")
    ejes[1].set_ylabel("|Z(f)|  [s]")
    ejes[1].legend(fontsize=8, loc="upper right")
    ejes[1].set_title("(b) Magnitud: ambas curvas se superponen exactamente",
                      loc="left", fontsize=10)
    ruta = _guardar(fig, "04_linealidad.png")

    return ruta, error_relativo(Z, Z_lineal)


# ---------------------------------------------------------------------------
# 2. DESPLAZAMIENTO EN EL TIEMPO
# ---------------------------------------------------------------------------
def verificar_desplazamiento(t0=0.3):
    """y(t) = x(t - t0): el pulso se retrasa t0 segundos.

    La magnitud |Y(f)| = |X(f)| NO cambia; solo se agrega una fase lineal
    -2π f t0 (recta con pendiente proporcional al retraso).
    """
    x = pulso_rectangular(t, ancho=T)
    y = pulso_rectangular(t, ancho=T, centro=t0)

    f, X = transformada_fourier(x, FS, t[0])
    _, Y = transformada_fourier(y, FS, t[0])
    Y_teo = X * np.exp(-1j * 2 * np.pi * f * t0)

    # Diferencia de fase medida ∠(Y·X*) = ∠Y − ∠X, solo donde |X| no es ~0.
    # Ambas fases se muestran "envueltas" en (−π, π]: la recta −2πf·t0 se ve
    # como un diente de sierra que da una vuelta completa cada 1/t0 Hz.
    util = np.abs(X) > 1e-3 * np.abs(X).max()
    dif_fase = np.where(util, np.angle(Y * np.conj(X)), np.nan)
    fase_teo = np.angle(np.exp(-1j * 2 * np.pi * f * t0))

    fig, ejes = plt.subplots(3, 1, figsize=(9, 8.2))
    fig.suptitle(f"Desplazamiento en el tiempo: x(t − {t0})  ↔  X(f)·e^(−j2πf·{t0})",
                 fontsize=13, fontweight="bold")
    ejes[0].plot(t, x, color=AZUL, label="x(t)")
    ejes[0].plot(t, y, color=ROJO, label=f"x(t − {t0})")
    ejes[0].set_xlim(-0.5, 0.7)
    ejes[0].set_xlabel("Tiempo t [s]")
    ejes[0].legend(fontsize=8)
    ejes[0].set_title("(a) Dominio del tiempo", loc="left", fontsize=10)

    ejes[1].plot(f, np.abs(X), color=AZUL, lw=2.5, label="|X(f)|")
    ejes[1].plot(f, np.abs(Y), "--", color=ROJO, lw=1.2, label="|Y(f)| (desplazada)")
    ejes[1].set_xlim(-30, 30)
    ejes[1].set_xlabel("Frecuencia f [Hz]")
    ejes[1].legend(fontsize=8)
    ejes[1].set_title("(b) Magnitud: idéntica, el retraso no cambia el contenido "
                      "de frecuencias", loc="left", fontsize=10)

    ejes[2].plot(f, fase_teo, "k-", lw=0.8, alpha=.6, label="−2π f t0 (teórica, envuelta)")
    ejes[2].plot(f, dif_fase, ".", color=ROJO, ms=3.5, label="∠Y(f) − ∠X(f) medida")
    ejes[2].set_xlim(-10, 10)
    ejes[2].set_xlabel("Frecuencia f [Hz]")
    ejes[2].set_ylabel("[rad]")
    ejes[2].set_yticks([-np.pi, 0, np.pi])
    ejes[2].set_yticklabels(["-π", "0", "π"])
    ejes[2].legend(fontsize=8, loc="upper right")
    ejes[2].set_title(f"(c) Fase: se suma una fase lineal −2π·f·t0 "
                      f"(una vuelta cada 1/t0 = {1 / t0:.2f} Hz)",
                      loc="left", fontsize=10)
    ruta = _guardar(fig, "05_desplazamiento_tiempo.png")

    return ruta, error_relativo(Y, Y_teo), error_relativo(np.abs(Y), np.abs(X))


# ---------------------------------------------------------------------------
# 3. ESCALAMIENTO (tiempo <-> frecuencia)
# ---------------------------------------------------------------------------
def verificar_escalamiento(factores=(0.5, 1.0, 2.0)):
    """x(a·t) para varios factores a.

    a > 1 comprime la señal en el tiempo  -> el espectro se EXPANDE en
           frecuencia y su altura baja a 1/|a|.
    a < 1 expande la señal en el tiempo   -> el espectro se COMPRIME.
    Se compara la FFT de x(a·t) con (1/|a|)·X(f/a), donde X(f/a) se obtiene
    interpolando el espectro numérico del pulso original.
    """
    x = pulso_rectangular(t, ancho=T)
    f, X = transformada_fourier(x, FS, t[0])
    colores = [VERDE, AZUL, ROJO]
    errores = {}

    fig, ejes = plt.subplots(2, 1, figsize=(9, 6.5))
    fig.suptitle("Escalamiento: x(a·t)  ↔  (1/|a|)·X(f/a)",
                 fontsize=13, fontweight="bold")
    for a, color in zip(factores, colores):
        # rect(a·t / T) = rect(t / (T/a)) -> pulso de ancho T/a
        xa = pulso_rectangular(t, ancho=T / a)
        _, Xa = transformada_fourier(xa, FS, t[0])

        # Lado derecho: (1/|a|)·X(f/a). X es real (pulso par) -> se
        # interpola la parte real del espectro original.
        Xa_teo = np.interp(f / a, f, X.real) / abs(a)
        errores[a] = error_relativo(Xa.real, Xa_teo)

        ejes[0].plot(t, xa, color=color, lw=1.4,
                     label=f"a = {a}: ancho T/a = {T / a:.2f} s")
        ejes[1].plot(f, np.abs(Xa), color=color, lw=1.6,
                     label=f"a = {a}: 1er cero en a/T = {a / T:.1f} Hz")
        ejes[1].plot(f, np.abs(Xa_teo), "k:", lw=0.9)

    ejes[0].set_xlim(-0.6, 0.6)
    ejes[0].set_xlabel("Tiempo t [s]")
    ejes[0].legend(fontsize=8, loc="upper right")
    ejes[0].set_title("(a) Dominio del tiempo: comprimir / expandir el pulso",
                      loc="left", fontsize=10)
    ejes[1].set_xlim(-40, 40)
    ejes[1].set_xlabel("Frecuencia f [Hz]")
    ejes[1].set_ylabel("|X(f)|  [s]")
    ejes[1].legend(fontsize=8, loc="upper right")
    ejes[1].set_title("(b) Frecuencia: pulso más angosto → espectro más ancho "
                      "(punteado = (1/|a|)·X(f/a))", loc="left", fontsize=10)
    ruta = _guardar(fig, "06_escalamiento.png")
    return ruta, errores


# ---------------------------------------------------------------------------
# 4. (EXTRA) MODULACIÓN = DESPLAZAMIENTO EN FRECUENCIA
# ---------------------------------------------------------------------------
def verificar_modulacion(fc=20.0):
    """Multiplicar por un coseno traslada el espectro a ±fc.

    Es el principio de la radio AM: la información (pulso) se "sube" a la
    frecuencia de la portadora fc.
    """
    x = pulso_rectangular(t, ancho=T)
    y = x * np.cos(2 * np.pi * fc * t)

    f, X = transformada_fourier(x, FS, t[0])
    _, Y = transformada_fourier(y, FS, t[0])
    Y_teo = 0.5 * (np.interp(f - fc, f, X.real) + np.interp(f + fc, f, X.real))

    fig, ejes = plt.subplots(2, 1, figsize=(9, 6))
    fig.suptitle(f"Modulación: x(t)·cos(2π·{fc:.0f}t)  ↔  ½[X(f−{fc:.0f}) + X(f+{fc:.0f})]",
                 fontsize=13, fontweight="bold")
    ejes[0].plot(t, x, color=AZUL, lw=1, alpha=.5, label="x(t) (mensaje)")
    ejes[0].plot(t, y, color=ROJO, lw=1.2, label="x(t)·cos(2πfc t)")
    ejes[0].set_xlim(-0.2, 0.2)
    ejes[0].set_xlabel("Tiempo t [s]")
    ejes[0].legend(fontsize=8)
    ejes[0].set_title("(a) Dominio del tiempo", loc="left", fontsize=10)
    ejes[1].plot(f, np.abs(X), color=AZUL, alpha=.5, label="|X(f)| original")
    ejes[1].plot(f, np.abs(Y), color=ROJO, lw=1.6, label="|Y(f)| modulada")
    ejes[1].plot(f, np.abs(Y_teo), "k:", lw=1, label="½[X(f−fc)+X(f+fc)]")
    ejes[1].set_xlim(-50, 50)
    ejes[1].set_xlabel("Frecuencia f [Hz]")
    ejes[1].legend(fontsize=8)
    ejes[1].set_title("(b) El espectro se copia en ±fc con la mitad de altura",
                      loc="left", fontsize=10)
    ruta = _guardar(fig, "07_modulacion.png")
    return ruta, error_relativo(Y.real, Y_teo)


# ---------------------------------------------------------------------------
# 5. (EXTRA) TEOREMA DE PARSEVAL
# ---------------------------------------------------------------------------
def verificar_parseval():
    """La energía es la misma calculada en el tiempo o en la frecuencia."""
    x = pulso_rectangular(t, ancho=T) + senoidal(t, 30, 0.5)
    X = np.fft.fft(x)
    energia_tiempo = np.sum(np.abs(x) ** 2)
    energia_frec = np.sum(np.abs(X) ** 2) / len(x)
    return energia_tiempo, energia_frec


def ejecutar():
    r = {}
    r["linealidad"] = verificar_linealidad()
    r["desplazamiento"] = verificar_desplazamiento()
    r["escalamiento"] = verificar_escalamiento()
    r["modulacion"] = verificar_modulacion()
    r["parseval"] = verificar_parseval()
    return r


def resumen(r):
    """Tabla de texto con el resultado numérico de cada verificación."""
    lineas = ["Propiedad                         | Error relativo / resultado",
              "----------------------------------|---------------------------"]
    lineas.append(f"Linealidad                        | {r['linealidad'][1]:.2e}")
    lineas.append(f"Desplazamiento (espectro complejo)| {r['desplazamiento'][1]:.2e}")
    lineas.append(f"Desplazamiento (|Y| vs |X|)       | {r['desplazamiento'][2]:.2e}")
    for a, e in r["escalamiento"][1].items():
        lineas.append(f"Escalamiento a = {a:<4}             | {e:.2e}")
    lineas.append(f"Modulación (extra)                | {r['modulacion'][1]:.2e}")
    et, ef = r["parseval"]
    lineas.append(f"Parseval (extra)                  | E_t = {et:.4f}, E_f = {ef:.4f}")
    return "\n".join(lineas)


if __name__ == "__main__":
    print(resumen(ejecutar()))
