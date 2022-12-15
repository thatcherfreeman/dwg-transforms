import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def least_squares_line(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
    assert len(x.shape) == 2 and len(y.shape) == 2
    assert x.shape == y.shape

    n, c = x.shape
    m, b = np.zeros(c), np.zeros(c)
    for c in range(x.shape[1]):
        x_points = x[:, c]
        y_points = y[:, c]
        n = x_points.shape[0]
        xy = x_points * y_points
        x2 = x_points**2
        sx = np.sum(x_points)
        sy = np.sum(y_points)
        sxy = np.sum(xy)
        sx2 = np.sum(x2)
        m[c] = (n * sxy - sx * sy) / (n * sx2 - (sx**2))
        b[c] = (sy - m[c] * sx) / n
    return m, b

def flatten(img):
    return np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))


# Read images
color_img = flatten(open_image("gopro-native-color-linear0000.exr"))
flat_img = flatten(open_image("gopro-native-flat-linear0000.exr"))

# Convert to log and remove Nans
log10_color = np.log10(color_img)
log10_flat = np.log10(flat_img)
na_mask = np.isfinite(log10_color).all(axis=1) & np.isfinite(log10_flat).all(axis=1)
log10_color = log10_color[na_mask, :]
log10_flat = log10_flat[na_mask, :]

# Find the gamma and intercept to match the logs.
# Using a two-step method, we first fit, then remove outliers, then fit again.
# WARNING: this outlier detection method doesn't appear to be effective.
m, b = least_squares_line(log10_color, log10_flat)
avg_gamma = np.mean(m)
avg_intercept = np.mean(b)
predicted_flats = avg_gamma * log10_color + avg_intercept
error = np.abs(predicted_flats - log10_flat)
low_error_mask = (error < np.percentile(error, 99)).all(axis=1)
m, b = least_squares_line(log10_color[low_error_mask, :], log10_flat[low_error_mask, :])

# Aggregate the per-channel gamma/intercept numbers.
avg_gamma = np.mean(m)
avg_intercept = np.mean(b)
# avg_gamma = .75
# avg_intercept = np.log10(1.0)
print(f"log10 y = {avg_gamma} * log10(x) + {avg_intercept}")
print(f"y = {10**avg_intercept} * (x ** {avg_gamma})")
gamma_func = lambda x: (10**avg_intercept) * (x ** avg_gamma)

# Plot the chart.
min_val = min(np.min(flat_img), np.min(color_img))
max_val = max(np.max(flat_img), np.max(color_img))
colors = {
    "red": 0,
    "green": 1,
    "blue": 2,
}
sample_idxs = np.random.choice(flat_img.shape[0], 3000)
for color_str, c in colors.items():
    plt.scatter(color_img[sample_idxs, c], flat_img[sample_idxs, c], color=color_str, alpha=0.05)

lin_values = 10**np.linspace(-5, 0, 1000)
gamma_corrected_values = [gamma_func(x) for x in lin_values]

plt.xlabel("log10 color image exposure values")
plt.ylabel("log10 flat image exposure values")
plt.plot(lin_values, gamma_corrected_values, color='orange', label=f"y = {10**avg_intercept} * (x ** {avg_gamma})")
plt.plot([lin_values[0], max_val], [lin_values[0], max_val], color='black', label='y=x')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('gopro_color_to_native.png')

