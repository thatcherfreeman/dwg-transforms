ACES Recommendation:
1. Linearize the image
2. Black Point subtraction, pixel correction, flat fielding, etc.
3. White balance (per-channel scaling so that a neutrally colored object under the scene illuminanthas equal RGB code values)
4. Clipping
5. 3x3 Matrix Application: conversion from Camera to AP0 then chromatic adaptation
6. Exposure Scaling (18% reflector corresponds to 0.18)

In practice, we can assume the "White balance" step occured already in-camera. Under this assumption, the camera takes these steps:
1. obtain linear image (RAW image)
2. Black point subtraction
3. White Balance (channel scaling) in the camera native space
4. Clipping
5. 3x3 matrix, log encode the image to convert to some log profile that is written to disk

Thus, our IDT then only has to do the following steps
6. Re-linearize the image by inverting the lin2log function.
7. 3x3 Matrix Application to go from the log profile to the target working color space
8. Exposure Scaling.
9. (If necessary) Apply log encoding curve to convert from linear to the working space encoding.
fdsafd