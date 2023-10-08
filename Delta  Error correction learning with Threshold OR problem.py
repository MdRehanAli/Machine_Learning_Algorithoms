import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt

# Activation Function
def Threshold_Function(z):
    return 1 if z >= 0.2 else 0

# Training Sample Data
eta = 0.1
epoch = 5

x = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, 1, 1]])
y = np.array([[-1, -1, -1, 1]])
w = np.array([[0.3, -0.1, -0.1]])




