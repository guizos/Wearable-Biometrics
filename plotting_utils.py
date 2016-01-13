__author__ = 'NickFoubert'
from matplotlib.pylab import plot, title, xlabel, ylabel, xlim


def draw_plot(data,plot_title):
    plot(range(len(data)),data,alpha=0.8,color='red')
    title(plot_title)
    xlabel("Samples")
    ylabel("Signal")
    xlim((0,len(data)-1))


def draw_plot_with_peaks(data,plot_title,peak_function,lookahead=300):
    peaks = peak_function(data,lookahead=lookahead)
    peaks_x = [peak[0] for peak in peaks[0]]
    peaks_y = [peak[1] for peak in peaks[0]]
    plot(range(len(data)),data,'b-',peaks_x,peaks_y,'ro')
    title(plot_title)
    xlabel("Samples")
    ylabel("Signal")
    xlim((0,len(data)-1))

def draw_plot_with_peaks_and_valleys(data,plot_title,lookahead=300,delta=0):
    peaks = peak_function(data,lookahead=lookahead,delta=delta)
    peaks_x = [peak[0] for peak in peaks[0]]
    peaks_y = [peak[1] for peak in peaks[0]]
    valleys_x =  [peak[0] for peak in peaks[1]]
    valleys_y =  [peak[1] for peak in peaks[1]]
    plot(range(len(data)),data,'b-',peaks_x,peaks_y,'ro',valleys_x,valleys_y,'go')
    title(plot_title)
    xlabel("Samples")
    ylabel("Signal")
    xlim((0,len(data)-1))