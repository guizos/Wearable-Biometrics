import logging

import numpy
import numpy as np
import peakutils
from matplotlib.pylab import plot
from biosppy.signals import ecg
from scipy.signal import argrelextrema

from data.segmentation.general import peak_radius_based_segmenter
from utils import peakdetect

def t40_peaks(ecg_time_signal,frequency=100):
    """
        Creates a set of labeled data samples composed of ecg beat samples
        We use the biosppy to get the main peaks (R point). Then
        we consider a beat 40 points behind and after that point
        We skip the first 3 and last 0.5 seconds to reduce noise
        :param ecg_time_signal:
        :return: an array of LabeledSamples without outliers
        """
    d = ecg_time_signal
    if len(d) < 90 or np.abs(d).max() == 0:
        return []
    out = ecg.ecg(signal=d, sampling_rate=frequency, show=None)
    peaks = out[2]
    # If the first peak is R, we remove it, we want to start reading
    # from the first T.
    if len(peaks) == 0:
        return []
    while peaks[0] < 40:
        peaks = peaks[1:]
    while peaks[-1] > len(d) - 40:
        peaks = peaks[:-2]
    return peaks

def t40_beat_samples(ecg_time_signal,frequency=100):
    peaks = t40_peaks(ecg_time_signal=ecg_time_signal,frequency=frequency)
    return peak_radius_based_segmenter(ecg_time_signal, peaks, radius=40)

def beat_samples(ecg_time_signal,frequency=100):
    out = ecg.ecg(signal=ecg_time_signal, sampling_rate=frequency, show=None)
    return [ecg_template for ecg_template in out[4]]

#TO adapt to general approach based on peaks
def beat_timewindow_samples(ecg_time_signal, time=5, frequency=100):
    limit = time * frequency
    final_peaks = []
    out = ecg.ecg(signal=ecg_time_signal, sampling_rate=frequency, show=None)
    peaks = out['rpeaks']
    for i in range(len(peaks)):
        if i == 0 or current_peak - peaks[i] >= limit:
            current_peak = peaks[i]
            final_peaks.append(current_peak)
    return [ecg_time_signal[peak - 40:peak + (limit - 40)] for peak in final_peaks]

#TO adapt to general approach based on peaks
def QRSComplex_beat_samples(ecg_time_signal, frequency=100):
    labeled_samples = []
    out = ecg.ecg(signal=ecg_time_signal, sampling_rate=frequency, show=None)
    peaks = out['rpeaks']
    valleys = peakdetect.peakdetect(ecg_time_signal.data, lookahead=5, delta=0)
    valleys_x = [valley[0] for valley in valleys[1]]
    for peak in peaks:
        peak_x = peak
        if valleys_x[0] < peak_x < valleys_x[-1] and len(valleys_x) > 2:
            begin_x = [valley for valley in valleys_x if valley < peak_x][-1]
            end_x = [valley for valley in valleys_x if valley > peak_x][0]
            if end_x - begin_x < 20:
                sample = ecg_time_signal[begin_x:end_x]
                labeled_samples.append(sample)
    return labeled_samples

#TO adapt to general approach based on peaks
def QRSComplesAllZero_beat_samples(ecg_time_signal, frequency=100):
    presamples = QRSComplex_beat_samples(ecg_time_signal, frequency)
    samples = []
    for presample in presamples:
        origin = presample[0]
        new_data = [point - origin for point in presample]
        samples.append(new_data)
    return samples

segmentation_functions = {"t40_beat_samples": t40_beat_samples, "beat_samples": beat_samples, "beat_timewindow_samples": beat_timewindow_samples, "QRSComplex_beat_samples": QRSComplex_beat_samples , "QRSComplesAllZero_beat_samples": QRSComplesAllZero_beat_samples}




class BeatUtils:

    @staticmethod
    def filter(ecg_labeled_time_signals,weight=1.5):
        lengths = [len(ecg_labeled_time_signal.data) for ecg_labeled_time_signal in ecg_labeled_time_signals]
        average = numpy.mean(lengths)
        stdeviation = numpy.std(lengths)
        return [ecg_labeled_time_signal for ecg_labeled_time_signal in ecg_labeled_time_signals if average - (1.5 * stdeviation) < len(ecg_labeled_time_signal.data) < average + (1.5 * stdeviation)]


class ECGUtils:

    @staticmethod
    def plot_splits(data,lookahead,delta):
        splits = ECGUtils.split_points(data,lookahead,delta)
        splits_y = [data[i] for i in splits]
        plot(data)
        plot(splits,splits_y,'ro')

    @staticmethod
    def split_points(data,lookahead=19,delta=0):
        points = []
        #peaks = peakdetect.peakdetect(data, lookahead=lookahead, delta=delta)
        peaks = peakdetect.peakdetect_fft(data,range(len(data)),pad_len=5)
        if len(peaks[0])==0:
            return []
        peaks_x = [peak[0] for peak in peaks[0]]
        peaks_y = [peak[1] for peak in peaks[0]]
        # If the first peak is R, we remove it, we want to start reading
        # from the first T.
        if(peaks_y[0]>peaks_y[1]):
            peaks_x = peaks_x[1:]
        points.append(peaks_x[0])
        major = True
        #Filter split points. One small, one big. If not, skip that point.
        for i in range(len(peaks_x)-1):
            if major and peaks_y[i]>peaks_y[i+1]:
                i+=1
            elif not major and  peaks_y[i]<peaks_y[i+1]:
                i+=1
            else:
                points.append(peaks_x[i])
                i+=1
                major = not major


        return points

