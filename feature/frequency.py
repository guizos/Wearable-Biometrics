from scipy.fftpack import fft, dct

from datasource.generaldata import LabeledSample
from feature.beats import LabeledSamplesDatabase


class FFTLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, ecg_database, num_highest_frequencies, frequency=0.1):
        self.labeled_samples = []
        for ecg_beat in ecg_database.labeled_samples:
            to_append = fft(ecg_beat.data)
            self.labeled_samples.append(LabeledSample(to_append[:num_highest_frequencies],ecg_beat.user_id,ecg_beat.activity))

class DCTLabeledSamplesDatabase(LabeledSamplesDatabase):

    def __init__(self, ecg_database, num_highest_frequencies, frequency=0.1):
        self.labeled_samples = []
        for ecg_beat in ecg_database.labeled_samples:
            to_append = dct(ecg_beat.data)
            self.labeled_samples.append(LabeledSample(to_append[:num_highest_frequencies],ecg_beat.user_id,ecg_beat.activity))