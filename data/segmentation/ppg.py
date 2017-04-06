import logging

import numpy
from peakutils import indexes

from data.dataset import LabeledSamplesDatabase, LabeledSample


class PPGBeatLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, data_folder, interpolation_size=80, smoothed=True, normalize=True):
        ppg_labeled_samples = []
        self.data_folder = data_folder
        for data_file in data_folder.data_files:
            if data_file.activity != 3:
                normalized_and_smoothed_ppg_sample = LabeledSamplesDatabase.process_sample(data_file.get_ppg_time_signal(),0,smoothed,normalize)
                normalized_and_smoothed_ecg_sample = LabeledSamplesDatabase.process_sample(data_file.get_ecg_time_signal(),0,smoothed,normalize)
                ppg_labeled_samples.extend(self.labeled_ppg_beat_samples(normalized_and_smoothed_ppg_sample,normalized_and_smoothed_ecg_sample))
        #interpolate, etc.
        self.labeled_samples = LabeledSamplesDatabase.process_samples(ppg_labeled_samples,interpolation_size,smoothed,normalize)


    def labeled_ppg_beat_samples(self,ppg_labeled_time_signal,ecg_labeled_time_signal):
        """
        Creates a set of PPGBeats from raw data. As data is being read at the same time,
        we use the same strategey using data from ECG to extract each beat. Generally, the peak of
        the PPG signal is obtained between 300 and 400 miliseconds after the ECG beat, so we consider that
        a PPG signal starts at the time we detect the ECG peak

        keyword arguments:
        data -- An arbitrary length data set of PPG signals.
        """
        d = ppg_labeled_time_signal.data
        i_der = numpy.gradient(numpy.gradient([value*-2.0 for value in ecg_labeled_time_signal.data]))
        peaks_ecg = indexes(i_der,thres=0.05,min_dist=50)
        if len(peaks_ecg)==0:
            return []
        while peaks_ecg[0]<40:
            peaks_ecg = peaks_ecg[1:]
        while peaks_ecg[-1]> len(d)-80:
            peaks_ecg = peaks_ecg[:-2]
        return [LabeledSample(d[peak:peak+80],ppg_labeled_time_signal.user_id,ppg_labeled_time_signal.activity) for
                peak in peaks_ecg]


class PPGBeatTimeLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, data_folder, time=5,  smoothed=True, normalize=True):
        self.labeled_samples = []
        self.time = time
        for data_file in data_folder.data_files:
            if data_file.activity != 3:
                normalized_and_smoothed_ppg_sample = LabeledSamplesDatabase.process_sample(data_file.get_ppg_time_signal(),0,smoothed,normalize)
                normalized_and_smoothed_ecg_sample = LabeledSamplesDatabase.process_sample(data_file.get_ecg_time_signal(),0,smoothed,normalize)
                self.labeled_samples.extend(self.labeled_ppg_beat_samples(normalized_and_smoothed_ppg_sample,normalized_and_smoothed_ecg_sample))
        logging.info("PPG Database created with time per sample="+str(time)+" smoothed="+str(smoothed)+" and normalize="+str(normalize))

    def labeled_ppg_beat_samples(self,ppg_labeled_time_signal,ecg_labeled_time_signal):
        """
        Creates a set of PPGBeats from raw data. As data is being read at the same time,
        we use the same strategey using data from ECG to extract each beat. Generally, the peak of
        the PPG signal is obtained between 300 and 400 miliseconds after the ECG beat, so we consider that
        a PPG signal starts at the time we detect the ECG peak

        keyword arguments:
        data -- An arbitrary length data set of PPG signals.
        """
        d = ppg_labeled_time_signal.data
        limit = self.time * 100.0
        i_der = numpy.gradient(numpy.gradient([value*-2.0 for value in ecg_labeled_time_signal.data]))
        peaks_ecg = indexes(i_der,thres=0.05,min_dist=50)
        if len(peaks_ecg)==0:
            return []
        while peaks_ecg[0]<40:
            peaks_ecg = peaks_ecg[1:]
        while peaks_ecg[-1]> len(d)-limit:
            peaks_ecg = peaks_ecg[:-2]
        final_peaks = []
        current_peak = 0
        for i in range(len(peaks_ecg)):
            if i == 0 or current_peak-peaks_ecg[i] >= limit:
                current_peak = peaks_ecg[i]
                final_peaks.append(current_peak)
        return [LabeledSample(d[peak-40:peak+(limit-40)],ppg_labeled_time_signal.user_id,ppg_labeled_time_signal.activity) for
                peak in final_peaks]


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