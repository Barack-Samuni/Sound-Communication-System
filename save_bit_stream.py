from shared_utils import *
import wave

def save_bit_stream(output_file_for_0,output_file_for_1):
    bit_stream_0 = "0" * 300  # half a minute of "0"
    bit_stream_1 = "1" * 300  # half a minute of "1"

    signal_for_0 = generate_signal(bits=bit_stream_0)
    signal_for_1 = generate_signal(bits=bit_stream_1)

    with wave.open(output_file_for_0, 'wb') as wf_0:
        wf_0.setnchannels(1)  # Mono audio
        wf_0.setsampwidth(2)  # 16-bit PCM
        wf_0.setframerate(44100)
        wf_0.writeframes((signal_for_0 * 32767).astype(np.int16).tobytes())

    with wave.open(output_file_for_1, 'wb') as wf_1:
        wf_1.setnchannels(1)
        wf_1.setsampwidth(2)
        wf_1.setframerate(44100)
        wf_1.writeframes((signal_for_1 * 32767).astype(np.int16).tobytes())



if __name__ == "__main__":
    save_bit_stream('0.wav', '1.wav')

