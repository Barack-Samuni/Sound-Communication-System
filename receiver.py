import numpy as np
import wave
import pyaudio
from shared_utils import *


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
    try:
        while True:
            # Record and buffer
            data = stream.read(chunk_size)
            frames.append(data)

            # keep buffer manageable
            if len(frames) > 2 * sample_rate // chunk_size:
                frames = frames[-2 * sample_rate // chunk_size:]

            # process buffer
            signal = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32767.0
            binary_data += demodulate_message(signal=signal, duration=duration, sample_rate=sample_rate)

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


