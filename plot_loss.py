import matplotlib.pyplot as plt
import time

f = open("losses.log", "r")
raw = f.read()
ys = raw.split(" ")
y = [float(yi) for yi in ys[:-1]]
f.close()

plt.plot(y)
plt.title('Loss progress')
plt.savefig(f'loss_{int(time.time())}.png')
plt.show()
