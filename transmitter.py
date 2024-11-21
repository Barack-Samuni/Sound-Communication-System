import  numpy as np
import wave
from scipy.signal import butter, lfilter
from shared_utils import *
import pyaudio

def hamming_encode(data):
    """
    This function encodes a data using Hamming (7,6) code.
    :param data: data to encode
    :return: encoded data
    """
    G = np.array([
        [1, 0, 0 ,0, 1, 1 ,0],
        [0, 1, 0, 0 ,1, 0 ,1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ])

    encoded = []

    for i in range(0, len(data), 4):
        block = np.array(list(map(int, data[i:i+4].ljust(4, '0')))) # Pad to 4 bits
        encoded_block = np.dot(block, G) % 2
        encoded.extend(encoded_block)
    return ''.join(map(str, encoded))


# noinspection PyUnresolvedReferences
def modulate_message(text, output_file, lowcut=480, highcut=1020, duration=0.1, sample_rate=44100):
    """
    modulates a text message into a sound using FSK modulation filtered by BPF and encoded with Hamming code
    :param text: text to be modulated
    :param output_file: path of to save the modulated sound as a .wav file. Will be useful in order to play it
    from a non-computer device
    :param lowcut: lower cutoff frequency
    :param highcut: upper cutoff frequency
    :param duration: bit duration
    :param sample_rate: sampling rate
    :return: the normalized modulated signal
    """
    binary_data = ''.join(format(ord(char),'08b') for char in text)
    encoded_binary_data = hamming_encode(binary_data)
    encoded_binary_with_barkers = START_BARKER + encoded_binary_data + END_BARKER
    print(f'Original binary data: {binary_data}')
    print(f'Hamming encoded binary data: {encoded_binary_data}')
    print(f'Full encoded binary data (with barkers): {encoded_binary_with_barkers}')

    freq_0 = 500    # frequency for '0'
    freq_1 = 1000   # frequency for '1'
    amplitude = 0.8

    signal = []
    for bit in encoded_binary_with_barkers:
        freq = freq_1 if bit == '1' else freq_0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = amplitude * np.sin(2 * np.pi * freq * t) # sound harmony
        signal.extend(tone)

    signal = np.array(signal, dtype=np.float32)
    filtered_signal = apply_bandpass_filter(signal, lowcut, highcut, sample_rate, order=5)  # filter the data

    # Normalize the signal
    normalized_signal = filtered_signal / np.max(np.abs(filtered_signal))

    # Save the filtered sound into a file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes((filtered_signal * 32767).astype(np.int16).tobytes())

    return  normalized_signal

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
    message = 'My name is Barack'
    output_file = 'My_name_is_Barack.wav'
    signal = modulate_message(text=message,output_file=output_file)
    play_audio(signal=signal)