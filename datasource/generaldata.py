import numpy
from matplotlib.pylab import plot, title, xlabel, ylabel, xlim
__author__ = 'guizos'


class OriginalTimeSignalCollection:

    def __init__(self,user_id,activity,time_signal,seconds=5):
        #The frequency of all our readings is 100H
        self.time_signals = time_signal.split(seconds*100)[0:-1]
        self.user_id = user_id
        self.activity = activity


    def get_time_signal(self,number):
        if number < len(self.time_signals):
            return self.time_signals[number]
        else:
            return None

class ModifiedTimeSignalCollection(OriginalTimeSignalCollection):

    def __init__(self,user_id,activity,time_signals):
        #The frequency of all our readings is 100H
        self.time_signals = time_signals
        self.user_id = user_id
        self.activity = activity

class LabeledSample:

    def __init__(self,data,user_id,activity):
        self.data = data
        self.user_id = user_id
        self.activity = activity

class LabeledTimeSignal(LabeledSample):

    def smoothed_movingaverage(self,window_size):
        """
        Creates a smoothed version of the TimeSignal.

        keyword arguments:
        window_size -- The window to be used by the moving average

        return -- a smoothed TimeSignal
        """
        window = numpy.ones(int(window_size))/float(window_size)
        return LabeledTimeSignal(numpy.convolve(self.data, window, 'same'))

    def plot(self,plot_title):
        plot(range(len(self.data)),self.data,alpha=0.8,color='red')
        title(plot_title)
        xlabel("Samples")
        ylabel("Signal")
        xlim((0,len(self.data)-1))

    def split(self,size):
        """Yield successive n-sized chunks from data."""
        return [LabeledTimeSignal(self.data[i:i + size], self.user_id, self.activity) for i in xrange(0, len(self.data), size)]




