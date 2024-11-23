import numpy as np
import pyaudio
from shared_utils import *
import  keyboard
import wave
from shared_utils import hamming_decode

CHUNK_SIZE = 4410  # Number of samples for one bit (0.1s at 44100 Hz)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 0.7


def parse_bit(bit_segment, sample_rate=44100, bit_duration=0.1, threshold=0.5):
    """
    Parse a single bit from a signal segment.

    :param bit_segment: Array representing the signal for one bit.
    :param sample_rate: Sampling rate in Hz.
    :param bit_duration: Duration of the bit in seconds.
    :param threshold: Minimum correlation threshold for confident detection.
    :return: The parsed bit ('0' or '1') if successful, or None if not.
    """
    # Constants
    freq_0 = 500
    freq_1 = 1000

    # Bandpass filtering for '0' and '1'
    filtered_0 = apply_bandpass_filter(bit_segment, freq_0 - 50, freq_0 + 50, sample_rate,order=2)
    filtered_1 = apply_bandpass_filter(bit_segment, freq_1 - 50, freq_1 + 50, sample_rate,order=2)

    # Energy comparison
    energy_0 = np.sum(filtered_0** 2)
    energy_1 = np.sum(filtered_1 ** 2)

    if energy_0 > energy_1:
        # Likely bit is '0'
        reference_signal = generate_signal("0", freq_0, freq_1, bit_duration, sample_rate)
        correlation = normalized_cross_correlation(filtered_0, reference_signal)
        if np.abs(np.mean(correlation)) > threshold:
            print(f"Bit parsed as '0' with correlation: {np.mean(correlation):.2f}")
            return '0'
        else:
            pass
            # print(f"Failed to parse '0'. Correlation: {np.mean(correlation):.2f}")
    else:
        # Likely bit is '1'
        reference_signal = generate_signal("1", freq_0, freq_1, bit_duration, sample_rate)
        correlation = normalized_cross_correlation(filtered_1, reference_signal)
        if np.abs(np.mean(correlation)) > threshold:
            print(f"Bit parsed as '1' with correlation: {np.mean(correlation):.2f}")
            return '1'
        else:
            pass
            # print(f"Failed to parse '1'. Correlation: {np.mean(correlation):.2f}")


    return None


def continuously_parse_bits():
    """
    Continuously record audio and attempt to parse a bit from each chunk of data.
    Stops when the user presses 'q'.
    """
    parsed_bits = ""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    print("Listening for bits... Press 'q' to stop.")

    try:
        while True:
            # Record one bit's worth of data (CHUNK_SIZE samples)
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            signal = normalize_signal(np.frombuffer(data, dtype=np.int16).astype(np.float32))

            # Attempt to parse a bit
            parsed_bit = parse_bit(signal, sample_rate=RATE, bit_duration=CHUNK_SIZE / RATE, threshold=THRESHOLD)

            if parsed_bit is not None:
                parsed_bits += parsed_bit
                print(f"Parsed bit: {parsed_bit}")

            # Check if 'q' was pressed to quit
            if keyboard.is_pressed('q'):
                print("Stopping...")
                break

    except KeyboardInterrupt:
        print("Recording stopped by user.")
    finally:
        print(f"Parsed bits: {parsed_bits}")
        decoded_bits = hamming_decode(parsed_bits)
        print(f"decoded parsed bits: {decoded_bits}")
        decoded_message = ""

        for i in range(0, len(decoded_bits),8):
            decoded_message += chr(int(decoded_bits[i:i+8],2))

        print(f"Decoded message: {decoded_message}")
        stream.stop_stream()
        stream.close()
        p.terminate()


def parse_bits_from_wav(wav_file, bit_duration=0.1, sample_rate=44100, threshold=0.5):
    """
    Parse bits from a .wav file using bit-parsing logic.

    :param wav_file: Path to the .wav file containing the signal.
    :param bit_duration: Duration of each bit in seconds.
    :param sample_rate: Sampling rate in Hz.
    :param threshold: Minimum correlation threshold for confident detection.
    :return: List of parsed bits.
    """
    # Open and read the .wav file
    with wave.open(wav_file, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        if n_channels != 1 or sampwidth != 2 or framerate != sample_rate:
            raise ValueError("Unsupported WAV file format. Ensure mono, 16-bit PCM, and correct sample rate.")

        signal = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32)
        signal = normalize_signal(signal)

    # Process the signal in chunks corresponding to each bit
    bit_samples = int(bit_duration * sample_rate)
    parsed_bits = ""

    for i in range(0, len(signal) - bit_samples + 1, bit_samples):
        bit_segment = signal[i:i + bit_samples]
        parsed_bit = parse_bit(bit_segment, sample_rate, bit_duration, threshold)
        if parsed_bit is not None:
            parsed_bits += parsed_bit

    return parsed_bits


if __name__ == "__main__":
    # bits = parse_bits_from_wav('My_name_is_Barack.wav')
    # print(f"bits are: {''.join(bits)}")
    # decoded_bits = hamming_decode(bits)
    # print(f"decoded_bits are: {''.join(decoded_bits)}")
    # decoded_message = ""
    # for i in range(0, len(decoded_bits),8):
    #     decoded_message += chr(int(decoded_bits[i:i + 8], 2))
    #
    # print(f"decoded_message: {decoded_message}")
    q

