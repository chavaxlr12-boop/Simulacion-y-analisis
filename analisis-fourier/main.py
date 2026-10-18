"""
main.py
=======
Ejecuta toda la actividad de una sola vez:

    1. Señales elementales y su espectro      (senales_elementales.py)
    2. Verificación de propiedades            (propiedades.py)
    3. Aplicaciones reales: ECG y DTMF        (aplicaciones.py)

Las figuras se guardan en ./figuras y el resumen numérico en
./resultados.txt.

Ejecutar:  python main.py
"""

from pathlib import Path

import senales_elementales
import propiedades
import aplicaciones


def main():
    lineas = ["RESULTADOS NUMÉRICOS - ANÁLISIS DE SEÑALES CON FOURIER", "=" * 56]

    print("1) Señales elementales...")
    for nombre, datos in senales_elementales.ejecutar().items():
        lineas.append(f"\n[{nombre}]")
        for clave, valor in datos.items():
            if clave != "ruta":
                lineas.append(f"  {clave}: {valor:.6g}")

    print("2) Propiedades de la transformada...")
    lineas.append("\n[Propiedades]  (error ~1e-16 = exacto a precisión de máquina)")
    lineas.append(propiedades.resumen(propiedades.ejecutar()))

    print("3) Aplicaciones...")
    r = aplicaciones.ejecutar()
    _, antes, despues = r["ecg"]
    _, marcado, detectado = r["dtmf"]
    lineas.append("\n[Aplicaciones]")
    lineas.append(f"  ECG: SNR antes = {antes:.1f} dB -> después = {despues:.1f} dB "
                  f"(mejora de {despues - antes:.1f} dB)")
    lineas.append(f"  DTMF: número marcado = {marcado}, decodificado = {detectado}")

    texto = "\n".join(lineas)
    Path(__file__).with_name("resultados.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print("\nFiguras guardadas en ./figuras")


if __name__ == "__main__":
    main()
