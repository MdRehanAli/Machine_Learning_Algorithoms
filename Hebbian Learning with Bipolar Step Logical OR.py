import numpy as np
from matplotlib import pyplot as plt

# Activation Function
def Bipolar_Step(x):
    return 1 if x >= 0 else -1

# Training
x = np.array([[-1, -1, 1, 1],[-1, 1, -1, 1], [1, 1, 1, 1]])
y = np.array([-1, 1, 1, 1])
w = np.array([0, 0, 0])

for i in range(x.shape[1]):
    sample = x[:, i]
    w[0] = w[0] + y[0][i] * sample


