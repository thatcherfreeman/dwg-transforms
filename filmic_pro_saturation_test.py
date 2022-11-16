import matplotlib.pyplot as plt
import numpy as np

def mix(x, y, t):
    # (1 - t)x + ty
    return (x + (y - x) * t)

def forward(rgb_linear):
    p = np.sum(rgb_linear * np.array([0.2126, 0.7152, 0.0722]), axis=-1, keepdims=True)
    o = 0.16667 * np.log(p) + 1.0
    output = mix(rgb_linear, p, o)
    return output

x = np.linspace(0, 1, 1024)
zeros = np.zeros_like(x)
plt.plot(x, forward(np.stack([x, zeros, zeros], axis=1))[:, 0], color='red')
plt.plot(x, forward(np.stack([zeros, x, zeros], axis=1))[:, 1], color='green')
plt.plot(x, forward(np.stack([zeros, zeros, x], axis=1))[:, 2], color='blue')
plt.plot(x, forward(np.stack([x, x, x], axis=1))[:, 0], color='black')
plt.show()