import numpy as np
import wave
import pyaudio
from shared_utils import *
from scipy.signal import correlate

def normalize_signal(signal):
    """
    Normalize the signal to range [-1, 1].
    :param signal: signal to be normalized
    :return: signal after normalization
    """
    max_val = np.max(np.abs(signal))

    if max_val > 0:
        return signal / max_val

    return signal

def normalized_cross_correlation(signal, barker_signal):
    """
    This function computes the normal cross correlation between two signals.
    :param signal: recorded signal
    :param barker_signal: barker referenced signal
    :return: normalized cross correlation
    """
    correlation = correlate(signal, barker_signal, mode='valid')
    norm_factor = np.sqrt(np.sum(signal ** 2) * np.sum(barker_signal ** 2)) # multiplication of the two energy norms

    if norm_factor > 0:
        return correlation / norm_factor
    return correlation



def generate_barker_signal(barker, freq_0=500, freq_1=1000, duration=0.1, sample_rate=44100):
    """
    generate a reference signal for the barker. We will use it to cross-correlate with the recorded signal and align it
    :param barker: the barker code to modulate
    :param freq_0: frequency for '0'
    :param freq_1: frequency for '1'
    :param duration: duration of tone
    :param sample_rate: sampling rate
    :return: barker as a signal
    """
    signal = []
    for bit in barker:
        freq = freq_1 if bit == '1' else freq_0
        t = np.linspace(0, duration, int(sample_rate * duration) , endpoint=False)
        tone = np.sin(2 * np.pi * freq * t)
        signal.extend(tone)

    # Normalize the barker reference signal
    signal = np.array(signal, dtype=np.float32)
    return signal / np.max(np.abs(signal))  # Scale to [-1, 1]


def align_to_start_barker(signal, barker_signal,chunk_size, sample_rate, threshold=0.8):
    """
    Use chunkwise cross correlation to align the received signal to the start barker
    :param signal: received signal
    :param barker_signal: reference barker signal
    :param chunk_size: chunk size for cross-correlation
    :param sample_rate: sampling rate
    :param threshold: threshold for cross-correlation
    :return: aligned signal
    """
    barker_length = len(barker_signal)
    num_samples = chunk_size + barker_length - 1    # sliding window size
    start_index = None

    # process signal in sliding windows
    for i in range(0, len(signal) - num_samples, chunk_size):
        chunk = signal[i:i + num_samples]
        normalized_chunk = normalize_signal(chunk)
        cross_corr = normalized_cross_correlation(normalized_chunk, barker_signal)

        # check if correlation exceeds the threshold
        if np.max(cross_corr) > threshold:
            start_index = i + np.argmax(cross_corr)
            print(f"Start barker detected at index {start_index}")
            break

    return start_index


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


def decode_message_with_hamming(binary_data):
    """
    This function decodes a message with hamming and also checks for start and end barkers
    :param binary_data: data to decode
    :return: decoded message as text or original binary data
    """
    messsage_start_index = binary_data.find(START_BARKER)
    messsage_end_index = binary_data.find(END_BARKER)

    if messsage_start_index != -1:  # start barker was found:
        print(f"Start barker detected at index {messsage_start_index}")

    if messsage_end_index != -1:
        print(f"End barker detected at index {messsage_end_index}")

    # check if start and end barkers are present in the data:
    if messsage_start_index == -1 or messsage_end_index == -1 or messsage_end_index <= messsage_start_index:
        return None, binary_data

    # Decode the message between the barkers
    hamming_data = binary_data[messsage_start_index + len(START_BARKER):messsage_end_index]
    decoded_data = hamming_decode(hamming_data)
    message = "".join(chr(int(decoded_data[i:i + 8], 2)) for i in range(0, len(decoded_data), 8))
    return message, ""  # still return a tuple for consistency

def record_audio_and_process_message(output_file,sample_rate=44100, chunk_size=1024, duration=0.1):
    """
    records an audio via microphone and saves it as .wav file
    :param chunk_size: size of chunk in bits
    :param duration: duration of tone in seconds
    :param sample_rate: sampling rate
    :param output_file: path to output .wav file
    :return:
    """

    format_of_audio = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=format_of_audio, channels=channels, rate=sample_rate, frames_per_buffer=chunk_size, input=True)
    print("Listening for start barker...")

    frames = []
    binary_data = ""

    # Generate a reference start barker signal
    barker_signal = generate_barker_signal(barker=START_BARKER, duration=duration, sample_rate=sample_rate)

    try:
        while True:
            # Read audio data from the stream and append it to the frames list
            data = stream.read(chunk_size)
            frames.append(data)

            # keep buffer manageable
            signal = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32767.0
            normalized_signal = normalize_signal(signal)

            # check for alignment
            start_index = align_to_start_barker(normalized_signal, barker_signal, chunk_size, sample_rate)
            if start_index is not None:
                print(f"ALigned signal starting at index {start_index}")
                aligned_signal = normalized_signal[start_index:]

                # Demodulate and decode
                binary_data += demodulate_message(signal=aligned_signal, duration=duration, sample_rate=sample_rate)

                # check for start and end barker and decode message
                message , binary_data = decode_message_with_hamming(binary_data=binary_data)
                if message:
                    print(f"Decoded message: {message}")
                    break
    finally:
        print("Recording stopped...")
        stream.stop_stream()
        stream.close()
        p.terminate()


    # save recorded audio to file
    print(f"saving recorded audio to {output_file}")
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format_of_audio))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    print("Audio was saved successfully.")


def demodulate_message(signal, duration=0.1, sample_rate=44100, freq_0=500, freq_1=1000):
    """
    This function demodulates the audio back to text
    :param signal: audio signal
    :param duration: duration of tone in seconds
    :param sample_rate: sampling rate
    :param freq_0: frequency of 0
    :param freq_1: frequency of 1
    :return: demodulated binary data
    """
    step = int(duration * sample_rate)  # number of samples per segment
    binary_data = ""
    margin = 20

    for i in range(0, len(signal), step):
        segment = signal[i:i+step]

        if len(segment) < step: # skip incomplete segment
            continue

        # Apply filters
        segment_0 = apply_bandpass_filter(data=segment, lowcut=freq_0 - margin, highcut=freq_0 + margin, fs=sample_rate)
        segment_1 = apply_bandpass_filter(data=segment, lowcut=freq_1 - margin, highcut=freq_1 + margin, fs=sample_rate)

        # Calculate energies
        energy_0 = np.sum(segment_0 ** 2)
        energy_1 = np.sum(segment_1 ** 2)

        # Choose the bit based on higher energy
        binary_data += "0" if energy_0 > energy_1 else "1"

    return binary_data


# Example usage
if __name__ == "__main__":
    record_audio_and_process_message("my_name_is_Barack_decoded.wav")


