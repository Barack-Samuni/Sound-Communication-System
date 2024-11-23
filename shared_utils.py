import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import correlate


# barker_codes
START_BARKER = "10101011"   # start barker in binary
END_BARKER = "11110000"     # end barker in binary

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
    nyquist = fs * 0.5           # Nyquist Theorem, make a float out of it
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


def generate_signal(bits, freq_0=500, freq_1=1000, duration=0.1, sample_rate = 44100):
    """
    Generate a normalized audio signal for given bits.
    :param bits: bits to generate a signal from
    :param freq_0: frequency for '0'
    :param freq_1: frequency for '1'
    :param duration: duration of bit
    :param sample_rate: sampling rate
    :return:
    """
    signal= []
    for bit in bits:
        freq = freq_1 if bit == "1" else freq_0
        t = np.linspace(0, duration, int(duration * sample_rate),endpoint=False)
        tone = np.sin(2 * np.pi * freq * t)
        signal.extend(tone)
    signal = np.array(signal, dtype=np.float32)
    return signal / np.max(np.abs(signal))  # Normalize to [-1, 1]

def hamming_encode(data_bits):
    """
    Encode binary data using (7,4) Hamming code.
    :param data_bits: a stream of bits
    :return: encoded data as binary string in thr following format:
    p1, p2, d1, p3, d2, d3, d3, d4
    p1 - parity bit for bits 3,5,7 in the code (d1, d2, d4)
    p2- parity bit for bits 3,6,7 in the code (d1, d3, d4)
    p3 - parity bit for bits 5,6,7 in the code (d2, d3, d4)
    """
    G = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
    ], dtype=int)

    encoded_bits = ""

    for i in range(0, len(data_bits), 4):   # 4 bits at the time

        # convert the input data bits into a row vector
        block = np.array([int(bit) for bit in data_bits[i:i+4].rjust(4,'0')])  # pad to 4 bits

        # perform matrix multiplication and apply modulo-2 (parity checking)
        encoded = (np.dot(G.T, block) % 2).flatten()

        # Convert the result into a binary string
        encoded_bits += "".join(map(str, encoded))
    return encoded_bits

def hamming_decode(encoded_bits):
    """
    Decode binary data using (7,4) hamming code.
    :param encoded_bits: The encoded binary string
    :return:Decoded data as a binary string
    """
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],       # parity check matrix
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=int)

    R = np.array([
        [0 ,0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0 ,0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    decoded_bits = ""
    for i in range(0, len(encoded_bits), 7):    # convert 7 bits to 4 bits
        block = np.array([int(bit) for bit in encoded_bits[i:i+7].rjust(7,'0')])    # pad to 7 bits
        syndrome = np.dot(H, block) % 2   # make the multiplication valid since block a column vector
        if np.any(syndrome):    # Error detected (syndrome != 0)
            error_index = int("".join(map(str, syndrome[::-1])), 2) - 1
            if 0 <= error_index < len(encoded_bits):    # valid range
                print(f"There was an error at index {error_index}")
                block[error_index] ^= 1 # flip the bit and correct the error
        bits_string = map(str, map(str, np.dot(R, block) % 2))
        decoded_bits+= "".join(bits_string)
    return decoded_bits

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
        return np.max(correlation) / norm_factor
    return np.max(correlation)

#   check hamming decode and decode
if __name__ == "__main__":
    data = "1011"
    print(f"Original data: {data}")
    code = hamming_encode(data)
    print(f"The encoded word is: {code}")
    decoded_data = hamming_decode(code)
    print(f"The decoded word is: {decoded_data}")

    for i in range(len(code)):
        flipped_bit = int(code[i]) ^ 1
        error_code = ""
        for j in range(len(code)):
            if j == i:
                error_code += str(flipped_bit)
            else:
                error_code += code[j]
        print(f"error code is: {error_code}")
        decoded_data = hamming_decode(error_code)
        print(f"The decoded word is: {decoded_data}")

