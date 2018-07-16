import numpy as np
import matplotlib.pylab as plt

x, y, = [], []

with open(r'E:\PycharmProjects\machine_learning\Regression\ex0.txt', 'r') as f:
    contents = f.readlines()
    for line in contents:
        line = line.strip('\n')
        line = line.split('\t')
        x.append(float(line[1]))
        y.append(float(line[2]))

x, y = np.array(x), np.array(y)
plt.figure()
plt.scatter(x, y, c='g', s=6)
plt.show()

theta = np.zeros((2, 1))
