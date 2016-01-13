import matplotlib.pyplot as plt

def normalize(x):
    normalizer = plt.Normalize()
    result = normalizer(x)
    return result