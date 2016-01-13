import numpy
import peakutils
from matplotlib.pylab import plot

from datasource.generaldata import LabeledSample
from preprocessing import interpolation
from preprocessing import normalization
from preprocessing import smoothing
from samplers import activity
from utils import peakdetect


class LabeledSamplesDatabase:

    def get_all_user_ids(self):
        return self.data_folder.get_all_user_ids()


    def get_labeled_training_and_test_samples(self,activity_training, user_id,num_training_samples):
        if activity_training!=0:
            samples = activity.ActivitySampler(activity_training,user_id,num_training_samples,self.labeled_samples)
        else:
            samples = activity.RandomSampler(user_id,num_training_samples,self.labeled_samples)
        (train,test) = samples.labeled_training_and_test_samples()
        return (train,test)

    @staticmethod
    def process_samples(labeled_samples, interpolation_size, smoothed, normalize=True):
        if interpolation_size > 0:
            interpolated_labeled_samples = [LabeledSample(interpolation.linear(sample.data,interpolation_size),sample.user_id,sample.activity) for sample in labeled_samples]
        else:
            interpolated_labeled_samples = labeled_samples
        if normalize:
            normalized_labeled_samples = [LabeledSample(normalization.normalize(time_signal.data),time_signal.user_id,time_signal.activity) for time_signal in interpolated_labeled_samples]
        else:
            normalized_labeled_samples = interpolated_labeled_samples
        if smoothed:
            smoothed_labeled_samples = [LabeledSample(smoothing.savitzky_golay(sample.data,9,3),sample.user_id,sample.activity) for sample in normalized_labeled_samples]
        else:
            smoothed_labeled_samples = normalized_labeled_samples
        return smoothed_labeled_samples

    @staticmethod
    def process_sample(sample, interpolation_size, smoothed, normalize=True):
        if interpolation_size > 0:
            interpolated_labeled_sample = LabeledSample(interpolation.linear(sample.data,interpolation_size),sample.user_id,sample.activity)
        else:
            interpolated_labeled_sample = sample
        if normalize:
            normalized_labeled_sample = LabeledSample(normalization.normalize(interpolated_labeled_sample.data),interpolated_labeled_sample.user_id,interpolated_labeled_sample.activity)
        else:
            normalized_labeled_sample = interpolated_labeled_sample
        if smoothed:
            smoothed_labeled_sample = LabeledSample(smoothing.savitzky_golay(normalized_labeled_sample.data,9,3),normalized_labeled_sample.user_id,normalized_labeled_sample.activity)
        else:
            smoothed_labeled_sample = normalized_labeled_sample
        return smoothed_labeled_sample


class ECGBeatLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, data_folder, interpolation_size, smoothed=True, normalize=True):
        ecg_labeled_samples = []
        self.data_folder = data_folder
        for data_file in data_folder.data_files:
            if data_file.activity != 3:
                normalized_and_smoothed_ecg_sample = LabeledSamplesDatabase.process_sample(data_file.get_ecg_time_signal(),0,smoothed,normalize)
                ecg_labeled_samples.extend(self.labeled_ecg_beat_samples(normalized_and_smoothed_ecg_sample))
        #interpolate, smoothing and normalization have been carried out in a previous step to ease peak detection
        self.labeled_samples = LabeledSamplesDatabase.process_samples(ecg_labeled_samples,interpolation_size,None,None)

    def labeled_ecg_beat_samples(self,ecg_labeled_time_signal):
        """
        Creates a set of labeled data samples composed of ecg beat samples
        We use the gradient four times to get the main peka (R point). Then
        we consider a beat 40 points behind and after that point
        We skip the first 3 and last 0.5 seconds to reduce noise
        :param ecg_labeled_time_signal:
        :return: an array of LabeledSamples without outliers
        """
        d = ecg_labeled_time_signal.data[300:-50]
        time_signals = []
        der = numpy.gradient(numpy.gradient(numpy.gradient(numpy.gradient(d))))
        peaks = peakutils.indexes(der)
        # If the first peak is R, we remove it, we want to start reading
        # from the first T.
        if len(peaks)!=0:
            if peaks[0]<40:
                peaks = peaks[1:]
            peaks_iterable = iter(peaks)
            initial = next(peaks_iterable, None)
            for peak in peaks_iterable:
                end = next(peaks_iterable, None)
                if end:
                    ecgbeat = LabeledSample(d[initial:end - 1],ecg_labeled_time_signal.user_id,ecg_labeled_time_signal.activity)
                    initial = end
                    time_signals.append(ecgbeat)
        return BeatUtils.filter(time_signals)

class ECGQRSComplexBeatLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, data_folder, interpolation_size=20, smoothed=True, normalize=True):
        self.data_folder = data_folder
        ecg_database = ECGBeatLabeledSamplesDatabase(data_folder, 0, smoothed,normalize)
        samples = []
        for ecg_beat in ecg_database.labeled_samples:
            to_append = self.labeled_qrs_beat_sample(ecg_beat)
            if to_append:
                samples.append(to_append)
        self.labeled_samples = LabeledSamplesDatabase.process_samples(samples,interpolation_size,None,None)

    def labeled_qrs_beat_sample(self,ecgbeat_labeled_time_signal):
        """
        Creates a set of labeled data samples composed of the QRS section
        of the ECG from timesignal raw data. An ECG signal is composed
        of the PQRST fiduicial points. We first extract the peaks as in
        ECGBeatLabeledSamplesDatabase. Then, from each peak we take 15 readings
        back and forward. We skip the first 300 seconds to reduce noise
        :param ecg_labeled_time_signal:
        :return: an array of LabeledSamples without outliers
        """
        d = ecgbeat_labeled_time_signal.data
        valleys = peakdetect.peakdetect(d,lookahead=5,delta=0)
        valleys_x = [valley[0] for valley in valleys[1]]
        valleys_y = [valley[1] for valley in valleys[1]]
        #Not valid QRS complex found
        if len(valleys_x)<2:
            return None
        peak_x = numpy.argmax(d[valleys_x[0]:valleys_x[-1]])+valleys_x[0]
        begin_x = [valley for valley in valleys_x if valley < peak_x][-1]
        end_x = [valley for valley in valleys_x if valley > peak_x][0]
        return LabeledSample(d[begin_x:end_x],ecgbeat_labeled_time_signal.user_id,ecgbeat_labeled_time_signal.activity)

class ECGQRSComplexAllBeginZeroBeatLabeledSamplesDatabase(ECGQRSComplexBeatLabeledSamplesDatabase):

    def __init__(self, data_folder, interpolation_size=20, smoothed=True, normalize=True):
        self.data_folder = data_folder
        ecg_database = ECGBeatLabeledSamplesDatabase(data_folder, 0, smoothed,normalize)
        samples = []
        for ecg_beat in ecg_database.labeled_samples:
            to_append = self.labeled_qrs_beat_sample(ecg_beat)
            if to_append:
                samples.append(to_append)
        self.labeled_samples = ECGQRSComplexAllBeginZeroBeatLabeledSamplesDatabase.make_samples_start_at_zero(LabeledSamplesDatabase.process_samples(samples,interpolation_size,None,None))

    @staticmethod
    def make_samples_start_at_zero(labeled_samples):
        resulting_labeled_samples = []
        for labeled_sample in labeled_samples:
            origin = labeled_sample.data[0]
            new_data = [point-origin for point in labeled_sample.data]
            resulting_labeled_samples.append(LabeledSample(new_data,labeled_sample.user_id,labeled_sample.activity))
        return resulting_labeled_samples


class PPGBeatLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, data_folder, interpolation_size, smoothed, normalize=True):
        ppg_labeled_samples = []
        self.data_folder = data_folder
        for data_file in data_folder.data_files:
            ppg_labeled_samples.extend(self.labeled_ppg_beat_samples(data_file.get_ppg_time_signal()))
        #interpolate, etc.
        self.labeled_samples = LabeledSamplesDatabase.process_samples(ppg_labeled_samples,interpolation_size,smoothed,normalize)



    def labeled_ppg_beat_samples(self,ppg_labeled_time_signal):
        """
        Creates a set of PPGBeats from raw data. The PPG read is not very
        detailed. We just look for valleys and extract a beat between each
        two valleys


        keyword arguments:
        data -- An arbitrary length data set of PPG signals.
        """
        d = ppg_labeled_time_signal.data
        time_signals = []
        peaks = PPGUtils.split_points(d)
        # If the first peak is R, we remove it, we want to start reading
        # from the first T.
        peaks_iterable = iter(peaks)
        initial = next(peaks_iterable, None)
        for peak in peaks_iterable:
            end = peak
            ppgbeat = LabeledSample(d[initial:end - 1],ppg_labeled_time_signal.user_id,ppg_labeled_time_signal.activity)
            initial = end
            time_signals.append(ppgbeat)
        return BeatUtils.filter(time_signals)


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