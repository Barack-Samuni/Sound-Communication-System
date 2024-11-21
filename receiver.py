from scipy.signal import find_peaks
import numpy as np
from transmitter import butter_bandpass, apply_bandpass_filter
import wave
import pyaudio


def hamming_decode(data):
    """
    As our transmitted message was hamming encoded, we need to decode it.
    :param data: encoded data that needs to be decoded
    :return: The decoded data
    """
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0 ,1, 0],
        [1, 0, 1, 1, 0, 0, 1],
    ])

    decoded = []
    for i in range(0, len(data), 7):   # 7 bit code block
        block = np.array(list(map(int, data[i:i+7].ljust(7,'0'))))  # 7 bit padding
        syndrome = np.dot(H, block) % 2

        if np.any(syndrome):    # syndrome != 0 means that there's an error in thr given index
            error_position = int(''.join(map(str,syndrome[::-1])),2) - 1
            block[error_position] = ~block[error_position]  # flip the bit

        decoded_block = block[:4]   # remove parity bits
        decoded.extend(decoded_block)
    return ''.join(map(str, decoded))

def record_audio(output_file, record_seconds=5, sample_rate=44100):
    """
    records an audio via microphone and saves it as .wav file
    :param output_file: path to save the .wav file
    :param record_seconds: the length of the record
    :param sample_rate: sampling rate
    :return:
    """
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=sample_rate,frames_per_buffer=chunk, input=True)
    print("Recording...")

    frames = []
    for _ in range(0, int(sample_rate / chunk * record_seconds)):    # number of samples
        data = stream.read(chunk)
        frames.append(data)
    print("Recording complete.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))


def demodulate_message(audio_file, lowcut=400, highcut=1200, sample_rate=44100, duration=0.1):
    """
    This function demodulates the audio back to text
    :param audio_file: audio file to read the recording from
    :param lowcut: lower cutoff frequency
    :param highcut: upper cutoff frequency
    :param sample_rate: sampling frequency
    :param duration: duration of tone
    :return: demodulated text
    """
    with wave.open(audio_file, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

    # Apply filtering on the signal
    filtered_signal = apply_bandpass_filter(signal, lowcut, highcut, sample_rate)
    step = int(duration * sample_rate)
    binary_data_with_hamming = ""

    for i in range(0, len(filtered_signal), step):
        segment = filtered_signal[i:i+step]
        fft = np.abs(np.fft.fft(segment))[:step // 2]
        freqs = np.fft.fftfreq(len(segment), 1 / sample_rate)[:step // 2]

        # Detect dominant frequency (which should represent the bit)
        peak_indices, _ = find_peaks(fft, height=0.1)
        peak_freqs = freqs[peak_indices]
        freq_0 = 500        # frequency for '0'
        freq_1 = 1000       # frequency for '1'
        margin = 50         # margin for detecting the bit

        if any(abs(f - freq_0) < margin for f in peak_freqs):  # this bit is close to '0'
            binary_data_with_hamming += "0"

        elif any(abs(f - freq_1) < margin for f in peak_freqs):
            binary_data_with_hamming += "1"

    # decode the hamming data
    binary_data = hamming_decode(binary_data_with_hamming)
    print(f"Decoded binary data: {binary_data}")

    # convert the binary data back to text
    text = ''.join(chr(int(binary_data[i:i+8], 2)) for i in range(0, len(binary_data), 8)) # convert binary to int first
    return text

# Example usage
if __name__ == "__main__":
    recorded_file = "decoded_My_name_is_barack.wav"
    record_audio(recorded_file, record_seconds=30)
    decoded_text = demodulate_message(recorded_file)
    print(f"Decoded text: {decoded_text}")


