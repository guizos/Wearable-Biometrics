import logging

import numpy
import numpy as np
import peakutils
from matplotlib.pylab import plot
from biosppy.signals import ecg
from scipy.signal import argrelextrema

from data.dataset import LabeledSamplesDatabase, LabeledSample
from utils import peakdetect

class OriginalECGBeatLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, data_folder, interpolation_size=80, smoothed=True, normalize=True):
        ecg_labeled_samples = []
        for data_file in data_folder.data_files:
            if data_file.activity != 3:
                normalized_and_smoothed_ecg_sample = LabeledSamplesDatabase.process_sample(data_file.get_ecg_time_signal(),0,smoothed,normalize)
                new_samples = self.labeled_beat_samples(normalized_and_smoothed_ecg_sample)
                if len(new_samples)>0:
                    ecg_labeled_samples.extend(new_samples)
        #interpolate, smoothing and normalization have been carried out in a previous step to ease peak detection
        self.labeled_samples = LabeledSamplesDatabase.process_samples(ecg_labeled_samples,interpolation_size,None,None)
        logging.info("ECG Database created with interpolation size="+str(interpolation_size)+" smoothed="+str(smoothed)+
                     " and normalize="+str(normalize))

    def labeled_beat_samples(self, ecg_labeled_time_signal):
        """
        Creates a set of labeled data samples composed of ecg beat samples
        We use the gradient four times to get the main peka (R point). Then
        we consider a beat 40 points behind and after that point
        We skip the first 3 and last 0.5 seconds to reduce noise
        :param ecg_labeled_time_signal:
        :return: an array of LabeledSamples without outliers
        """
        d = ecg_labeled_time_signal.data[300:-50]
        i_der = numpy.gradient(numpy.gradient([value*-2.0 for value in d]))
        out = ecg.ecg(signal=d,sampling_rate=100,show=None)
        peaks = out[2]
        # If the first peak is R, we remove it, we want to start reading
        # from the first T.
        if len(peaks)==0:
            return []
        while peaks[0]<40:
            peaks = peaks[1:]
        while peaks[-1]> len(d)-40:
            peaks = peaks[:-2]
        return [LabeledSample(d[peak-40:peak+40],ecg_labeled_time_signal.user_id,ecg_labeled_time_signal.activity,peak) for
                peak in peaks]


class ECGBeatLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, data_folder):
        self.labeled_samples = []
        for data_file in data_folder.data_files:
            if data_file.activity != 3:
                try:
                    out = ecg.ecg(signal=data_file.get_ecg_time_signal().data,sampling_rate=100,show=None)
                    self.labeled_samples.extend([LabeledSample(ecg_template,data_file.user_id,data_file.activity) for ecg_template in out[4]])
                except Exception:
                    logging.info("Error with user {0} and activity {1}".format(data_file.user_id,data_file.activity))
        logging.info("ECG Database created with biosppy library")


class ECGBeatTimeLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, data_folder, time=5, frequency=100):
        self.labeled_samples = []
        self.data_folder = data_folder
        self.time = time
        limit = time*frequency
        final_peaks = []
        for data_file in data_folder.data_files:
            if data_file.activity != 3:
                out = ecg.ecg(signal=data_file.get_ecg_time_signal().data,sampling_rate=frequency,show=None)
                normalized_and_smoothed_ecg_sample = LabeledSamplesDatabase.process_sample(data_file.get_ecg_time_signal(),0,True,True)
                peaks = out['rpeaks']
                for i in range(len(peaks)):
                    if i == 0 or current_peak-peaks[i] >= limit:
                        current_peak = peaks[i]
                        final_peaks.append(current_peak)
                d = normalized_and_smoothed_ecg_sample.data
                self.labeled_samples.extend([LabeledSample(d[peak-40:peak+(limit-40)],data_file.user_id,data_file.activity) for
                peak in final_peaks])
        logging.info("ECG Database created with time per sample="+str(time)+" and frequency="+str(frequency))



class ECGQRSComplexBeatLabeledSamplesDatabase(ECGBeatLabeledSamplesDatabase):

    def __init__(self, data_folder):
        self.labeled_samples = []
        for data_file in data_folder.data_files:
            if data_file.activity != 3:
                normalized_and_smoothed_ecg_sample = LabeledSamplesDatabase.process_sample(data_file.get_ecg_time_signal(),0,True,True)
                out = ecg.ecg(signal=data_file.get_ecg_time_signal().data,sampling_rate=100,show=None)
                peaks = out['rpeaks']
                valleys = peakdetect.peakdetect(normalized_and_smoothed_ecg_sample.data,lookahead=5,delta=0)
                valleys_x = [valley[0] for valley in valleys[1]]
                for peak in peaks:
                    peak_x = peak
                    if valleys_x[0] < peak_x < valleys_x[-1] and len(valleys_x) > 2:
                        begin_x = [valley for valley in valleys_x if valley < peak_x][-1]
                        end_x = [valley for valley in valleys_x if valley > peak_x][0]
                        if end_x-begin_x < 20:
                            sample = LabeledSample(normalized_and_smoothed_ecg_sample.data[begin_x:end_x],data_file.user_id,data_file.activity)
                            interpolated = LabeledSamplesDatabase.process_sample(sample,15,None,None)
                            self.labeled_samples.append(interpolated)


class ECGQRSComplexAllBeginZeroBeatLabeledSamplesDatabase(ECGQRSComplexBeatLabeledSamplesDatabase):

    def __init__(self, data_folder):
        database = ECGQRSComplexBeatLabeledSamplesDatabase(data_folder)
        self.labeled_samples = []
        for labeled_sample in database.labeled_samples:
            origin = labeled_sample.data[0]
            new_data = [point-origin for point in labeled_sample.data]
            self.labeled_samples.append(LabeledSample(new_data,labeled_sample.user_id,labeled_sample.activity))



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

class PPGUtils:

    @staticmethod
    def plot_splits(data,lookahead,delta):
        splits = PPGUtils.split_points(data,lookahead,delta)
        splits_y = [data[i] for i in splits]
        plot(data)
        plot(splits,splits_y,'ro')

    @staticmethod
    def split_points(data,lookahead=19,delta=0):
        points = []
        peaks = peakdetect.peakdetect(data, lookahead=lookahead, delta=delta)
        peaks_x = [peak[0] for peak in peaks[1]]
        peaks_y = [peak[1] for peak in peaks[1]]
        # If the first peak is R, we remove it, we want to start reading
        # from the first T.
        return peaks_x