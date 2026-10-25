"""
Actividad formativa 2: Simulación y análisis de señales con la Transformada de Fourier.
Requisitos: Python 3, NumPy y Matplotlib.
Instalación: pip install numpy matplotlib
Ejecución: python simulacion_fourier.py
"""
import numpy as np
import matplotlib.pyplot as plt

# Parámetros de muestreo
fs = 1000                 # Frecuencia de muestreo (Hz)
duracion = 1.0             # Duración de observación (s)
t = np.arange(0, duracion, 1 / fs)
N = len(t)
frecuencias = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs))

# Señales elementales
pulso = np.where((t >= 0.2) & (t < 0.4), 1.0, 0.0)
escalon = np.where(t >= 0.5, 1.0, 0.0)
f0 = 50
senoidal = np.sin(2 * np.pi * f0 * t)

def obtener_espectro(senal):
    """Devuelve la FFT centrada y normalizada por el número de muestras."""
    return np.fft.fftshift(np.fft.fft(senal)) / len(senal)

def graficar_senal(senal, nombre, numero):
    """Grafica señal temporal, magnitud y fase del espectro."""
    X = obtener_espectro(senal)
    fig, ejes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    ejes[0].plot(t, senal)
    ejes[0].set_title(f"{nombre}: dominio del tiempo")
    ejes[0].set_xlabel("Tiempo (s)")
    ejes[0].set_ylabel("Amplitud")
    ejes[0].grid(True, alpha=0.3)

    ejes[1].plot(frecuencias, np.abs(X))
    ejes[1].set_title("Espectro de magnitud")
    ejes[1].set_xlabel("Frecuencia (Hz)")
    ejes[1].set_ylabel("Magnitud normalizada")
    ejes[1].set_xlim(-200, 200)
    ejes[1].grid(True, alpha=0.3)

    ejes[2].plot(frecuencias, np.angle(X))
    ejes[2].set_title("Espectro de fase")
    ejes[2].set_xlabel("Frecuencia (Hz)")
    ejes[2].set_ylabel("Fase (rad)")
    ejes[2].set_xlim(-200, 200)
    ejes[2].set_ylim(-3.4, 3.4)
    ejes[2].grid(True, alpha=0.3)

    archivo = f"senal_{numero}.png"
    fig.savefig(archivo, dpi=160)
    plt.show()
    plt.close(fig)
    return X

# Analizar las tres señales
X_pulso = graficar_senal(pulso, "Pulso rectangular", 1)
X_escalon = graficar_senal(escalon, "Función escalón", 2)
X_senoidal = graficar_senal(senoidal, "Señal senoidal de 50 Hz", 3)

# Propiedad 1: linealidad de la DFT
senal_a = np.sin(2 * np.pi * 30 * t)
senal_b = 0.5 * np.sin(2 * np.pi * 80 * t)
error_linealidad = np.max(np.abs(
    np.fft.fft(senal_a + senal_b)
    - (np.fft.fft(senal_a) + np.fft.fft(senal_b))
))
print(f"Error máximo de linealidad: {error_linealidad:.3e}")

# Propiedad 2: desplazamiento temporal circular exacto en muestras
muestras_retraso = 13
senal_desplazada = np.roll(senoidal, muestras_retraso)
X_original = np.fft.fft(senoidal)
X_desplazada = np.fft.fft(senal_desplazada)
error_magnitud = np.max(np.abs(np.abs(X_original) - np.abs(X_desplazada)))
print(f"Diferencia máxima de magnitud por desplazamiento: {error_magnitud:.3e}")
print("El desplazamiento modifica la fase; la magnitud se conserva.")

# Propiedad 3: escalamiento temporal con pulsos de distinta duración
pulso_ancho = np.where((t >= 0.3) & (t < 0.7), 1.0, 0.0)
pulso_estrecho = np.where((t >= 0.4) & (t < 0.6), 1.0, 0.0)
X_ancho = obtener_espectro(pulso_ancho)
X_estrecho = obtener_espectro(pulso_estrecho)

fig, ejes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
ejes[0].plot(t, pulso_ancho, label="Pulso ancho: 0.4 s")
ejes[0].plot(t, pulso_estrecho, label="Pulso estrecho: 0.2 s")
ejes[0].set_xlim(0.2, 0.8)
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].set_ylabel("Amplitud")
ejes[0].set_title("Compresión temporal del pulso")
ejes[0].legend()
ejes[0].grid(True, alpha=0.3)

ejes[1].plot(frecuencias, np.abs(X_ancho), label="Pulso ancho: 0.4 s")
ejes[1].plot(frecuencias, np.abs(X_estrecho), label="Pulso estrecho: 0.2 s")
ejes[1].set_xlim(-50, 50)
ejes[1].set_xlabel("Frecuencia (Hz)")
ejes[1].set_ylabel("Magnitud normalizada")
ejes[1].set_title("Efecto del escalamiento en el espectro")
ejes[1].legend()
ejes[1].grid(True, alpha=0.3)
fig.savefig("escalamiento.png", dpi=160)
plt.show()
plt.close(fig)

print("Simulación terminada. Se generaron las gráficas de las señales y del escalamiento.")
