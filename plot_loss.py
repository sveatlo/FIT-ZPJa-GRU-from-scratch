#!/usr/bin/env python3

import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    raise Exception("invalid number of arguments: use loss file as argument")

loss_file_path = Path(sys.argv[1])

f = open(loss_file_path, "r")
raw = f.read()
ys = raw.split(" ")
y = [float(yi) for yi in ys[:-1]]
f.close()

plt.plot(y)
plt.title('Loss progress')
plt.savefig(f'{loss_file_path.stem}.png')
plt.show()
