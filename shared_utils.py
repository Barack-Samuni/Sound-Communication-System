import numpy as np
from scipy.signal import butter, lfilter, cheby1 , cheby2, filtfilt
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


# noinspection PyTupleAssignmentBalance
def apply_chebyshev_filter(signal, lowcut, highcut, fs, order=5, ripple=0.5, filter_type='cheby1'):
    """
    Apply a Chebyshev bandpass filter (Type I or II).

    :param signal: Input signal to filter.
    :param lowcut: Lower cutoff frequency in Hz.
    :param highcut: Upper cutoff frequency in Hz.
    :param fs: Sampling rate in Hz.
    :param order: Filter order (default=5).
    :param ripple: Passband ripple in dB (for Type I) or stopband attenuation in dB (for Type II).
    :param filter_type: 'cheby1' for Type I or 'cheby2' for Type II (default is the cheby1 filter for noise suppression).
    :return: Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Ensure cutoff frequencies are valid
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError("Cutoff frequencies must be between 0 and Nyquist frequency.")

    # Design the Chebyshev filter
    if filter_type == 'cheby1':
        b, a = cheby1(order, ripple, [low, high], btype='band')
    elif filter_type == 'cheby2':
        b, a = cheby2(order, ripple, [low, high], btype='band')
    else:
        raise ValueError("Invalid filter_type. Use 'cheby1' or 'cheby2'.")

    # Apply zero-phase filtering
    return filtfilt(b, a, signal)


# noinspection PyTupleAssignmentBalance
def multi_bandpass_filter(signal, bands, fs, order_for_low=5,order_for_high=5,order_for_bandstop=5,
                          ripple=0.5, filter_type='cheby1', filter=True):
    """
    Applies a chebyshev bandpass filter (Type I or II) using the cascading of chebyshev highpass filter,
    bandstopfilters, and a low-pass-filter.
    and a low pass-filter.
    :param signal: signal to be filtered
    :param bands: bands to be filtered. This a matrix, where each row is a band.
    For example: [[low_0,high_0], [low_1,high_1], ...]
    :param fs: sampling frequency
    :param order_for_low: order of LPF (default=5).
    :param order_for_high: order of HPF (default=5).
    :param order_for_bandstop: order of bandstop (default=5).
    :param ripple:
    :param filter_type:
    :return:
    """
    nyquist = 0.5 * fs
    high_pass_cutoff = bands[0][0] / nyquist
    if filter_type == 'cheby1':
        b , a = cheby1(order_for_high, ripple, high_pass_cutoff, btype='high')

    elif filter_type == 'cheby2':
        b,a = cheby2(order_for_high, ripple, high_pass_cutoff, btype='high')

    else:
        raise ValueError("Invalid filter_type. Use 'cheby1' or 'cheby2'.")

    a_total = [a]
    b_total = [b]

    # number of bands - 1 band-stop filters
    for i in  range(len(bands) -1):
        bandstop_low_cutoff = bands[i][1] / nyquist     # start stopping at the end of the current pass band
        bandstop_high_cutoff = bands[i+1][0] / nyquist  # stop stopping at the start of the next pass band

        if filter_type == 'cheby1':
            b, a = cheby1(order_for_bandstop, ripple, [bandstop_low_cutoff, bandstop_high_cutoff], btype='bandstop')

        elif filter_type == 'cheby2':
            b , a = cheby2(order_for_bandstop, ripple, [bandstop_low_cutoff, bandstop_high_cutoff], btype='bandstop')

        else:
            raise ValueError("Invalid filter_type. Use 'cheby1' or 'cheby2'.")

        a_total.append(a)
        b_total.append(b)

    # low - pass - filter
    low_pass_cutoff = bands[-1][1] / nyquist    # last band high frequency

    if filter_type == 'cheby1':
        b , a = cheby1(order_for_low, ripple, low_pass_cutoff, btype='low')

    elif filter_type == 'cheby2':
        b, a = cheby2(order_for_low, ripple, low_pass_cutoff, btype='low')

    else:
        raise ValueError("Invalid filter_type. Use 'cheby1' or 'cheby2'.")

    a_total.append(a)
    b_total.append(b)

    # combine the filters altogether
    combined_a = a_total[0]                         # constant of the first filter denominator

    for a in a_total[1:]:
        combined_a = np.convolve(combined_a, a)     # convolve all the filters together

    combined_b = b_total[0]                         # constant value of the numerator

    for b in b_total[1:]:
        combined_b = np.convolve(combined_b, b)    # convolve all the filters together

    if not filter is True:
        return combined_b , combined_a
    return filtfilt(combined_b , combined_a, signal)


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
        return correlation / norm_factor
    return correlation

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

