from shared_utils import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fs = 44100
    duration = 0.1
    freq = 500
    lowcut = 450
    highcut = 550

    t = np.linspace(0, duration, int(duration * fs))
    sine = np.sin(2 * np.pi * freq * t)
    filtered_sine = apply_bandpass_filter(sine, lowcut, highcut, fs,order=4)
    # filtered_sine /= np.max(np.abs(filtered_sine))
    plt.subplot(2, 1, 1)
    plt.plot(t, sine)
    plt.xlabel('Time (s)')
    plt.ylabel('Sine(t)')
    plt.subplot(2, 1, 2)
    plt.plot(t, filtered_sine)
    plt.xlabel('Time (s)')
    plt.ylabel('Filtered Sine(t)')
    plt.show()
