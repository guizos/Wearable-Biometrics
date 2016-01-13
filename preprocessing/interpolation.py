import numpy
from scipy import interpolate

def linear(v,elements):
    x = range(len(v))
    f = interpolate.interp1d(x,v,kind="linear")
    return f(numpy.linspace(0,len(v)-1,num=elements,endpoint=None))

def nearest(v,elements):
    x = range(len(v))
    f = interpolate.interp1d(x,v,kind="nearest")
    return f(numpy.linspace(0,len(v)-1,num=elements,endpoint=None))

def zero(v,elements):
    x = range(len(v))
    f = interpolate.interp1d(x,v,kind="zero")
    return f(numpy.linspace(0,len(v)-1,num=elements,endpoint=None))

def slinear(v,elements):
    x = range(len(v))
    f = interpolate.interp1d(x,v,kind="slinear")
    return f(numpy.linspace(0,len(v)-1,num=elements,endpoint=None))

def quadratic(v,elements):
    x = range(len(v))
    f = interpolate.interp1d(x,v,kind="quadratic")
    return f(numpy.linspace(0,len(v)-1,num=elements,endpoint=None))

def cubic(v,elements):
    x = range(len(v))
    f = interpolate.interp1d(x,v,kind="cubic")
    return f(numpy.linspace(0,len(v)-1,num=elements,endpoint=None))

