import matplotlib.pyplot as plt
import numpy as np

def mix(x, y, t):
    return (x + (y - x) * t)

def lin2log(inVec):
    l = (np.log2(inVec) + 8.0) * 0.125
    x = mix(np.maximum(0.0, l), inVec, 0.1267)
    y = 0.28985507246
    outVec = np.maximum(0.0, mix(x, np.power(inVec, y), inVec))
    return outVec

def log2lin(inVec):
    u = np.exp2((inVec / 0.125) - 8.0)
    x = 1.1612159730893894
    y = 0.6090138106343165
    outVec = np.power(u, mix(x, y, inVec))
    return outVec

x = np.linspace(0.0, 1.0, 1024)
y = lin2log(log2lin(x))

# plt.plot(x, y, label="lin2log(log2lin(x))")
# plt.plot([0.0, 1.0], [0.0, 1.0], label="y=x")
# plt.legend()
# plt.show()

plt.plot(x, x**(1/3.6), label="3.6 gamma function")
plt.plot(x, (8/9)*(0.125 * np.log2(x) + 1), label="y=(8/9)*(0.125 * log2(x) + 1)")
# plt.plot([-8, 1], [0, 1], label="y=.125 x")
plt.plot(x, lin2log(x), label="lin2log(x)")
plt.legend()
plt.xscale('log')
plt.xlabel("x")
plt.ylabel("code value")
plt.show()