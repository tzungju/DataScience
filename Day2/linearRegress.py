import matplotlib.pyplot as plt
import numpy as np

n = 50
x = range(0,n)
y = np.add(np.add(range(0,n), np.random.random_integers(0,8,n)), -4)

y[-1] += 100

 
# Get current size
fig_size = plt.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 20
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

fig, ax = plt.subplots()
fit = np.polyfit(x, y, deg=1)
# y = ax+b, a = fit[0] , b = fit[1]
ax.plot(np.array(x).astype(int), fit[0] * np.array(x).astype(int) + fit[1], color='red')


ax.scatter(x, y)

fig.show()
plt.show()