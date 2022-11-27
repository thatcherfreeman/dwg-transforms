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


def interpolate(x, low, high):
    output = (x - low) / (high - low)
    output[x < low] = 0.0
    output[x > high] = 1.0
    return output


def lin2log_thatcher(x, gamma, lin_cutoff=0.125, smoothing=0.04):
    log_portion = (8/9)*(0.125 * np.log2(x) + 1)
    pow_portion = x**gamma
    interp = interpolate(x, lin_cutoff - smoothing, lin_cutoff + smoothing)
    output = pow_portion * interp + (1-interp) * log_portion
    output = np.maximum(0.0, output)
    return output, interp

def log2lin_thatcher(y, gamma, log_cutoff=0.55, smoothing=0.05):
    exp_portion = 2**(((y * 9/8) - 1) * 8)
    pow_portion = y**(1/gamma)
    interp = interpolate(y, log_cutoff - smoothing, log_cutoff + smoothing)
    output = pow_portion * interp + (1-interp) * exp_portion
    return output, interp


x_lin = 10**np.linspace(np.log10(0.17), 0.0, 4000)[:-1]
y_cv = lin2log(x_lin)
gamma_est = np.linalg.lstsq(np.log(np.expand_dims(x_lin, axis=1)), np.log(np.expand_dims(y_cv, axis=1)))[0][0]
print("Estimated gamma: ", gamma_est)


x = np.linspace(0.0, 1.0, 4096)
code_value = lin2log(x)
approx_code_value, interp = lin2log_thatcher(x, gamma_est)
recovered_x = log2lin_thatcher(code_value, gamma_est)[0]

print("linear Error: ", np.quantile(np.abs(recovered_x - x) / (np.abs(recovered_x + x)/2), 0.99))
print("max Code Value Error: ", 1024 * np.max(np.abs(code_value - lin2log(recovered_x))))
plt.plot(code_value, lin2log(recovered_x), color='orange', label='lin2log(log2lin_approx(code_value))')
plt.plot(x, x, color='blue', label='y=x')
plt.legend()
plt.show()


plt.plot(x, code_value, label="lin2log(x)", color='blue')
plt.plot(x[interp==0], approx_code_value[interp==0], label=f"lin2log approximation gamma={1.0/gamma_est}", color='orange')
plt.plot(x[interp==1], approx_code_value[interp==1], color='red')
plt.plot(x[(interp > 0.0) & (interp < 1.0)], approx_code_value[(interp > 0.0) & (interp < 1.0)], color='green')
plt.xscale('log')
plt.ylabel('code value')
plt.xlabel("exposure value")
plt.legend()
plt.show()

linear_val, interp = log2lin_thatcher(lin2log(x), gamma_est)
z = x

plt.plot(x, z, label="y=x", color='blue')
plt.plot(x[interp==0], linear_val[interp==0], label=f"log2lin_thatcher(lin2log(x)) approximation gamma={1.0/gamma_est}", color='orange')
plt.plot(x[interp==1], linear_val[interp==1], color='red')
plt.plot(x[(interp > 0.0) & (interp < 1.0)], linear_val[(interp > 0.0) & (interp < 1.0)], color='green')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel("exposure value")
plt.ylabel("recovered exposure value")
plt.show()



x = np.linspace(0.0, 1.0, 4096)
approx_lin_value, interp = log2lin_thatcher(x, gamma_est)
plt.plot(x[interp==0], approx_lin_value[interp==0], label=f"lin2log approximation gamma={1.0/gamma_est}", color='orange')
plt.plot(x[interp==1], approx_lin_value[interp==1], color='red')
plt.plot(x[(interp > 0.0) & (interp < 1.0)], approx_lin_value[(interp > 0.0) & (interp < 1.0)], color='green')
plt.yscale('log')
plt.ylabel('recovered exposure value')
plt.xlabel('code value')
plt.show()