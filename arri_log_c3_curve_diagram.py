import matplotlib.pyplot as plt
import numpy as np

def lin2log(x):
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809

    if x > cut:
        return c * np.log10(a * x + b) + d
    else:
        return e * x + f

log_values = np.linspace(-10.0, 10.0, 1000)
lin_values = 0.18 * (2.0**log_values)
code_values = [lin2log(x) for x in lin_values]

l = min(log_values)
r = max(log_values)

plt.plot(log_values, code_values, label="arri logc3 curve")
plt.hlines(y=0.149658, xmin=l, xmax=r, label="Cutoff", color='green')
plt.vlines(x=np.log2(0.010591 / 0.18), ymin=-0.1, ymax=1.1, label="Cutoff", color='green')
plt.hlines(y=0.0, xmin=l, xmax=r, label="y=0", color="black")
plt.hlines(y=1.0, xmin=l, xmax=r, label="y=1.0", color="orange")
plt.axhspan(ymin=1.0, ymax=1.1, xmin=l, xmax=r, label="clipping risk", alpha=0.3, color="orange")
plt.axhspan(ymin=-0.1, ymax=0.149658, xmin=l, xmax=r, label="offset not appropriate", alpha=0.3, color="red")
plt.legend()
plt.xlabel("Exposure value (0 is middle gray)")
plt.ylabel("Code Value")
plt.ylim(-0.1, 1.1)
plt.xlim(l, r)
plt.show()

