import numpy as np
import pyaudio
import wave
import keyboard
from shared_utils import *


CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
BIT_DURATION = 0.1                                          # Duration od each bit in seconds
BIT_SAMPLES = int(RATE * BIT_DURATION)                      # Number of samples per bit
PACKET_SIZE = 14                                            # Number of bits in a packet
PACKET_SAMPLES = PACKET_SIZE * BIT_SAMPLES
THRESHOLD = 0.5                                             # Cross-correlation threshold
BUFFER_SIZE = PACKET_SAMPLES                                # Rolling buffer size
OUTPUT_FILE_NAME = "my_name_is_barack_recorded.wav"
FREQ_0 =  500                                               # frequency for '0'
FREQ_1 = 1000                                               # frequency for '1'
MARGIN = 20                                                 # margin for bandwidth of the filter


def record_audio_and_process_message():
    """
    records an audio and tries to parse a message, and saves it to a .wav file
    :return:
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    print("Listening for transmissions... Press 'q' to stop.")

    rolling_buffer = []     # initialize an empty buffer
    binary_data = ""        # Parsed bits
    decoded_message = ""    # Final decoded message

    # prepare to save recorded audio
    with wave.open(OUTPUT_FILE_NAME, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        try:
            while not keyboard.is_pressed("q"):
                # Read audio data
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                wf.writeframes(data)
                signal = normalize_signal(np.frombuffer(data, dtype=np.int16).astype(np.float32))

                # Update rolling buffer
                rolling_buffer.extend(signal)           # Add new signal data
                if len(rolling_buffer) > BUFFER_SIZE:   # Truncate if buffer exceeds max size
                    rolling_buffer = rolling_buffer[-BUFFER_SIZE:]

                # Search for start barker
                hamming_start_barker = hamming_encode(START_BARKER)
                barker_signal = generate_signal(bits=hamming_start_barker, freq_0=FREQ_0, freq_1=FREQ_1,
                                                duration=BIT_DURATION, sample_rate=RATE)
                correlation = normalized_cross_correlation(signal, barker_signal)

                if np.mean(correlation) > THRESHOLD:
                    start_index = np.argmax(np.correlate(rolling_buffer, barker_signal, mode="valid"))
                    print(f"Start barker detected at index {start_index} with correlation {correlation:.2f}")
                    rolling_buffer = rolling_buffer[start_index:]   # Align to time=0
                    binary_data = ""                                # Reset binary data

                    # Parse bits
                    while len(rolling_buffer) >= BIT_SAMPLES:
                        bit_segment = rolling_buffer[:BIT_SAMPLES]
                        rolling_buffer = rolling_buffer[BIT_SAMPLES:]   # shift buffer

                        # Bandpass filtering
                        filtered_0 = apply_bandpass_filter(bit_segment, FREQ_0 - MARGIN, FREQ_0 + MARGIN, RATE)
                        filtered_1 = apply_bandpass_filter(bit_segment, FREQ_1 - MARGIN, FREQ_1 + MARGIN, RATE)
                        energy_0 = np.sum(filtered_0 ** 2)
                        energy_1 = np.sum(filtered_1 ** 2)

                        # Detect bit
                        if energy_0 > energy_1:
                            reference_signal = generate_signal(bits="0",freq_0=FREQ_0,
                                                               freq_1=FREQ_1,duration=BIT_DURATION,sample_rate=RATE)
                            correlation = normalized_cross_correlation(filtered_0, reference_signal)
                            if np.mean(correlation) > THRESHOLD:
                                binary_data += "0"

                            else:
                                print("Failed to parse bit. Requesting retransmission.")
                                break
                        else:
                            reference_signal = generate_signal(bits="1",freq_0=FREQ_0,freq_1=FREQ_1,
                                                               duration=BIT_DURATION,sample_rate=RATE)
                            correlation = normalized_cross_correlation(filtered_1, reference_signal)

                            if np.mean(correlation) > THRESHOLD:
                                binary_data += "1"

                            else:
                                print("Failed to parse bit. Requesting retransmission.")
                                break
                        # Parse packets
                        if len(binary_data) >= PACKET_SIZE:
                            packet = binary_data[:PACKET_SIZE]
                            binary_data = binary_data[PACKET_SIZE:] # Remove parsed packets
                            decoded_packet = hamming_decode(packet)

                            if decoded_packet == START_BARKER:
                                print("Synchronized with start barker.")

                            elif decoded_packet == END_BARKER:
                                print("End barker detected. Message conplete.")
                                print(f"Decoded message: {decoded_message}")
                                decoded_message = ""
                                break

                            else:
                                decoded_message += chr(int(decoded_packet, 2))
        except KeyboardInterrupt:
            print("Recording stopped by user.")

        finally:
            print(f"Recording saved to {OUTPUT_FILE_NAME}")
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    record_audio_and_process_message()


