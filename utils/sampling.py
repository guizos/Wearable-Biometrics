from numpy import random


def reservoir_sampling(iterator, limit):
    sample = []
    i = 0
    for line in iter(iterator):
        if i < limit:
            sample.append(line)
        elif i >= limit and random.random() < limit/float(i+1):
            replace = random.randint(0,len(sample)-1)
            sample[replace] = line
        i+=1
    return sample