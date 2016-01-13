#from datasource import generaldata


class PPGBeatSet():

    def __init__(self,d,split_mode=1):
        """
        Creates a set of PPGBeats from raw data. The PPG read is not very
        detailed. We just look for valleys and extract a beat between each
        two valleys


        keyword arguments:
        data -- An arbitrary length data set of PPG signals.
        """
        self.time_signals = []
        if split_mode== 1:
            peaks = PPGUtils.split_points(d)
        # If the first peak is R, we remove it, we want to start reading
        # from the first T.
        peaks_iterable = iter(peaks)
        initial = next(peaks_iterable, None)
        for peak in peaks_iterable:
            end = peak
            ecgbeat = generaldata.LabeledTimeSignal(d[initial:end - 1])
            initial = end
            self.time_signals.append(ecgbeat)




