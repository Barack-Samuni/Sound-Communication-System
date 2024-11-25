import  numpy as np
import wave
from scipy.signal import butter, lfilter
from shared_utils import *
import pyaudio

# noinspection PyUnresolvedReferences
def modulate_message(text, output_file, freq_0=500, freq_1=1000, lowcut=480, highcut=1020,
                     ripple=0.5, duration=0.1, sample_rate=44100, order=2):
    """
    modulates a text message into a sound using FSK modulation filtered by BPF and encoded with Hamming code
    :param text: text to be modulated
    :param output_file: path of to save the modulated sound as a .wav file. Will be useful in order to play it
    :param freq_0: frequency of '0' (in Hz)
    :param freq_1: frequency of '1' (in Hz)
    from a non-computer device
    :param lowcut: lower cutoff frequency
    :param highcut: upper cutoff frequency
    :param ripple: ripple in dB
    :param duration: bit duration
    :param sample_rate: sampling rate
    :param order: order of the chebyshev filter
    :return: the normalized modulated signal
    """
    binary_data = ''.join(format(ord(char),'08b') for char in text)
    encoded_binary_data_with_barkers = hamming_encode(START_BARKER + binary_data + END_BARKER)
    print(f'Original binary data: {binary_data}')
    print(f'Full encoded binary data (with barkers): {encoded_binary_data_with_barkers}')
    modulated_signal = generate_signal(bits=binary_data, freq_0=freq_0, freq_1=freq_1, duration=duration,
                                        sample_rate=sample_rate)
    # filtered_signal = apply_bandpass_filter(signal, lowcut, highcut, sample_rate, order=5)  # filter the data
    filtered_signal = apply_chebyshev_filter(modulated_signal, lowcut,highcut,fs=sample_rate,ripple=ripple,order=order)

    # Save the filtered sound into a file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes((filtered_signal * 32767.0).astype(np.int16).tobytes())

    return  filtered_signal

def play_audio(signal, sample_rate=44100):
    """
    This function plays an audio file (saved after modulating the message)
    :param signal: the signal to be played
    :param sample_rate: the sample rate to play the signal
    :return:
    """
    p = pyaudio.PyAudio()
    mono_channels = 1
    stream = p.open(format=pyaudio.paFloat32,channels=mono_channels,rate=sample_rate,output=True)
    print("Playing audio...")
    stream.write(signal.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio playback complete.")


# Example Usage
if __name__ == "__main__":
    message = 'test'
    output_file = 'test.wav'
    signal = modulate_message(text=message,output_file=output_file,freq_0=1000, freq_1=1500)
    play_audio(signal=signal)