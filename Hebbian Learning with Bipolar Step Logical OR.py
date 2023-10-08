import numpy as np
from matplotlib import pyplot as plt

# Activation Function
def Bipolar_Step(x):
    return 1 if x >= 0 else -1

# Training
x = np.array([[-1, -1, 1, 1],[-1, 1, -1, 1], [1, 1, 1, 1]])
y = np.array([[-1, 1, 1, 1]])
w = np.array([[0, 0, 0]])

for i in range(x.shape[1]):
    sample = x[:, i]
    w[0] = w[0] + y[0][i] * sample
    print(w)


# Testing
predicted = []
for i in range(x.shape[1]):
    sample = x[:,i]
    y_ = np.dot(sample, w[0])
    predicted.append(Bipolar_Step(y_))
print(predicted)


# Decision Boundary Calculation
dx = []
dy = []
for i in range(-2, 4):
    dx.append(i)
    dy.append((-w[0][0]/w[0][1])*i + (-w[0][2]/w[0][1]))

# Graph
plt.scatter([-1, 1, 1], [1, -1, 1])
plt.scatter(-1, -1,)
plt.show()