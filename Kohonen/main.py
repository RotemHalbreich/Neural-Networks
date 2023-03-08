import random
import numpy as np
from Kohonen import SOM
import math
from numpy.random.mtrand import dirichlet


def NonUniformRandom2(radius=1):
    r = radius * math.sqrt(np.random.random()) * 0.5
    theta = 2 * np.pi * np.random.rand()

    return 0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)


def UniformRandom(radius=1):
    r = radius * math.sqrt(np.random.random()) * 0.5
    theta = 2 * np.pi * np.random.rand()

    return 0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)


def NonUniformRandom1(radius=1):
    r = radius * np.random.random() * 0.5
    theta = 2 * np.pi * np.random.rand()

    return 0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)


def ReplicateNTimes(func, rad, Ntrials=1000):
    xpoints, ypoints = [], []
    for _ in range(Ntrials):
        xp, yp = func(rad)
        xpoints.append(xp)
        ypoints.append(yp)
    dist = (xpoints, ypoints)
    return dist


def ReplicateNTimes2(func, rad, Ntrials=1000):
    i = 0
    xpoints, ypoints = [], []
    while i < Ntrials:
        xp, yp = func(rad)
        if xp <= 0.5 and i <= Ntrials * 0.8:
            xpoints.append(xp)
            ypoints.append(yp)
            i = i + 1
        if xp > 0.5 and i > Ntrials * 0.8:
            xpoints.append(xp)
            ypoints.append(yp)
            i = i + 1
    dist = (xpoints, ypoints)
    return dist


def InitializesSamples(size=2000, task=1.0):
    samples = np.zeros((size, 2))
    np.random.seed(11)
    Ntrials = size
    radius = 1

    # uniform disk
    if task == 1.0:
        dist = ReplicateNTimes(UniformRandom, radius, Ntrials=Ntrials)
        samples[:, 0] = dist[0]
        samples[:, 1] = dist[1]

    # ring
    if task == 1.1:
        c = 0
        while c < size:
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            if 2 <= x ** 2 + y ** 2 <= 4:
                samples[c, 0] = x
                samples[c, 1] = y
                c = c + 1

    # non-uniform half disk : 80% in left
    if task == 1.2:
        dist = ReplicateNTimes2(NonUniformRandom2, radius, Ntrials=Ntrials)
        samples[:, 0] = dist[0]
        samples[:, 1] = dist[1]

    # non-uniform disk : most of the points in the center
    if task == 1.3:
        dist = ReplicateNTimes(NonUniformRandom1, radius, Ntrials=Ntrials)
        samples[:, 0] = dist[0]
        samples[:, 1] = dist[1]

    return samples


def main():
    samples0 = InitializesSamples()
    samples1 = InitializesSamples(task=1.1)
    samples2 = InitializesSamples(task=1.2)
    samples3 = InitializesSamples(task=1.3)

    op0 = "One-dimensional SOM over uniform disk"
    op1 = "One-dimensional SOM over ring"
    op2 = "One-dimensional SOM over non-uniform disk"
    op3 = "One-dimensional SOM over non-uniform-2 disk"
    op2D_0 = "Two-dimensional SOM over uniform disk"
    op2D_1 = "Two-dimensional SOM over ring"
    op2D_2 = "Two-dimensional SOM over non-uniform disk"
    op2D_3 = "Two-dimensional SOM over non-uniform-2 disk"


    """ QUESTION A.1: """
    # som = SOM(100, 2, samples0)
    # som.learn(opt=op0)

    # som = SOM(10, 10, 2, samples0)
    # som.learn(opt=op2D_0)

    """ QUESTION A.2: """
    # non-uniform disk
    # som = SOM(100, 2, samples2)
    # som.learn(opt=op2)

    # som = SOM(10, 10, 2, samples2)
    # som.learn(opt=op2D_2)

    # non-uniform-2 disk
    # som = SOM(100, 2, samples3)
    # som.learn(opt=op3)

    # som = SOM(10, 10, 2, samples3)
    # som.learn(opt=op2D_3)

    """ QUESTION A.3: """
    som = SOM(30, 2, samples1)
    som.learn(opt=op1)


if __name__ == '__main__':
    main()