from shared_utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import freqz

def find_optimal_orders(bands,sampling_rate,ripple=0.5,filter_type='cheby1'):
    """
    This function plots the transfer function for multiple orders of the filter
    :param bands: bands to filter
    :param sampling_rate: sampling_rate
    :param ripple:desired ripple (default 0.5)
    :param filter_type: Type of the desired chebyshev filter (default 'cheby1')
    :return:
    """
    original_signal = generate_signal("01010101")
    plt.figure()
    original_signal_spectrum = fft(original_signal)
    f_original = np.linspace(0 , sampling_rate , len(original_signal))
    plt.subplot(2, 1, 1)
    plt.plot(f_original, np.abs(original_signal_spectrum))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    for low_order in range(1,3):
        for high_order in range(1,3):
            for stop_order in range(1,3):
                filtered_signal_spectrum = fft(
                    multi_bandpass_filter(signal=original_signal, bands=bands, fs=sampling_rate, ripple=ripple,
                                          filter_type=filter_type, order_for_bandstop=stop_order,order_for_low=low_order,order_for_high=high_order))
                f_filtered = np.linspace(0 , sampling_rate , len(filtered_signal_spectrum))
                plt.plot(f_filtered, filtered_signal_spectrum,
                         label=f'low={low_order}, high={high_order}, stop={stop_order}')

    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # find_optimal_orders([[450,550],[950,1050]],44100)
    fs = 44100
    duration = 0.1
    # freq = 500
    # freq = 1000
    freq_0 = 500
    freq_1 = 1000
    lowcut = 450
    highcut = 550
    num_bits = 2
    margin = 50
    # t = np.linspace(0, duration, int(duration * fs))
    t_bit = np.linspace(0, duration, int(duration * fs))
    t = np.linspace(0, num_bits * duration, int(num_bits * duration * fs))
    # sine = np.sin(2 * np.pi * freq * t)
    # filtered_sine = apply_bandpass_filter(sine, lowcut, highcut, fs,order=4)
    # filtered_sine = apply_chebyshev_filter(sine, lowcut, highcut,fs, order=2)
    # filtered_sine /= np.max(np.abs(filtered_sine))
    # multi_sine = generate_signal("01")
    multi_sine = generate_signal("01", freq_0=50, freq_1 = 1450)
    filtered_multi_sine = multi_bandpass_filter(multi_sine, [[freq_0 - margin , freq_0 + margin],
                                                             [freq_1 - margin , freq_1 + margin]], fs,
                                                order_for_high=2,order_for_low=2,order_for_bandstop=2)
    plt.subplot(3, 1, 1)
    # plt.plot(t, sine)
    plt.plot(t, multi_sine)
    plt.xlabel('Time (s)')
    plt.ylabel('Multi_sine(t)')
    plt.subplot(3, 1, 2)
    plt.plot(t, filtered_multi_sine)
    plt.xlabel('Time (s)')
    plt.ylabel('Filtered_multi_sine(t)')
    plt.subplot(3,1,3)
    w, h = freqz(multi_bandpass_filter(multi_sine, [[freq_0 - margin , freq_0 + margin],
                                                             [freq_1 - margin , freq_1 + margin]], fs,
                                                order_for_high=2,order_for_low=2,order_for_bandstop=2, filter=False))
    plt.plot(w, 20 * np.log(np.abs(h)))
    plt.xlabel('freq(Hz)')
    plt.ylabel('H(z)')
    plt.tight_layout()
    plt.show()
