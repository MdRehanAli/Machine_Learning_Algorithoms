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
    sample = x[:, i]
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
plt.plot([-1, 1, 1], [1, -1, 1], 'o')
plt.plot(-1, -1, '8')
plt.plot(dx, dy, label="Decision Boundary")
for i_x, i_y in zip([-1, -1, 1, 1], [-1, 1, -1, 1]):
    plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
plt.title("Hebbian OR using Bipolar Step Function")
plt.legend()
plt.grid()
plt.show()