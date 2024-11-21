import numpy as np
from scipy.signal import butter, lfilter

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
