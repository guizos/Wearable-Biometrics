import numpy


def movingaverage(array, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(array, window, 'same')