import  numpy as np
import wave
from scipy.signal import butter, lfilter
import pyaudio


# noinspection PyTupleAssignmentBalance
def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    this function designs a Butterworth filter
    :param lowcut: lower cutoff frequency
    :param highcut: upper cutoff frequency
    :param fs: sampling frequency
    :param order: order of filter
    :return:  coefficients of the designed filter
    """
    nyquist = fs / 2            # Nyquist Theorem
    low = lowcut / nyquist      # Normalization
    high = highcut / nyquist    # Normalization
    b, a = butter(order, [low, high], btype='band')
    return b, a         # b - numerator a - denominator

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    this function applies Butterworth filter on a given data
    :param data: given binary data
    :param lowcut: lower cutoff frequency
    :param highcut: upper cutoff frequency
    :param fs: sampling frequency
    :param order: order of filter
    :return: filtered data
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


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
def modulate_message(text, output_file, lowcut=400, highcut=1200, duration=0.1, sample_rate=44100):
    """
    modulates a text message into a sound using FSK modulation filtered by BPF and encoded with Hamming code
    :param text: text to be modulated
    :param output_file: path of to save the modulated sound as a .wav file. Will be useful in order to play it
    from a non-computer device
    :param lowcut: lower cutoff frequency
    :param highcut: upper cutoff frequency
    :param duration: bit duration
    :param sample_rate: sampling rate
    :return:
    """
    binary_data = ''.join(format(ord(char),'08b') for char in text)
    encoded_binary_data = hamming_encode(binary_data)
    print(f'Original binary data: {binary_data}')
    print(f'Hamming encoded binary data: {encoded_binary_data}')

    freq_0 = 500    # frequency for '0'
    freq_1 = 1000   # frequency for '1'
    amplitude = 0.5 # 1 VPP

    signal = []
    for bit in encoded_binary_data:
        freq = freq_1 if bit == '1' else freq_0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = amplitude * np.sin(2 * np.pi * freq * t) # sound harmony
        signal.extend(tone)

    signal = np.array(signal, dtype=np.float32)
    filtered_signal = apply_bandpass_filter(signal, lowcut, highcut, sample_rate, order=5)  # filter the data

    # Save the filtered sound into a file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes((filtered_signal * 32767).astype(np.int16).tobytes())

    print(f"The modulated sound saved to {output_file}")

def play_audio(audio_file):
    """
    This function plays an audio file (saved after modulating the message)
    :param audio_file: audio file to play
    :return:
    """
    chunk = 1024
    wf = wave.open(audio_file, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=wf.getnchannels(),rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)

    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()


# Example Usage
if __name__ == "__main__":
    message = 'My name is Barack'
    output_file = 'My_name_is_Barack.wav'
    modulate_message(text=message,output_file=output_file)
    play_audio(audio_file=output_file)