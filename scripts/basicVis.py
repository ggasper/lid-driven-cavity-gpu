# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import h5py

filename = "../results/results.h5"

with h5py.File(filename, "r") as dataFile:
    attrs = dict(dataFile.attrs)
    pos = dataFile["domain/pos"][()]
    v = dataFile["velocity"][()]
    p = dataFile["pressure"][()]


arrowScale = 10
arrowWidth = 0.002
scatterMarkerSize = 5

fig, ax = plt.subplots(1, 2)
img = ax[0].scatter(*pos, s=scatterMarkerSize, c=np.linalg.norm(v, axis=0))
plt.colorbar(img, ax=ax, location="bottom", pad=0.15, fraction=0.1, aspect=50,
                label=r"$| v |$")
ax[1].quiver(*pos, *v, scale=np.max(v) * arrowScale, width=arrowWidth)

for a in ax:
    a.set_xlabel("$x$")
ax[0].set_ylabel("$y$")

fig.savefig("test.png")