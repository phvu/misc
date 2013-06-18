import math
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-6, 6, 0.05)
y1 = np.array([max(a, 0) for a in x])
y2 = np.array([1./(1+math.exp(-a)) for a in x])
y3 = np.array([math.tanh(a) for a in x])
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.plot(x, y1, label='ReLU', linewidth=2)
plt.plot(x, y2, label='Logistic', linewidth=2)
plt.plot(x, y3, label='tanh', linewidth=2)
plt.legend()
plt.show()

