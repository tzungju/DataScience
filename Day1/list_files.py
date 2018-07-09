import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 20
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

# 列出目錄中所有的檔案
# 4 * 4 個視窗
fig, ax = plt.subplots(nrows=4, ncols=4,) # sharex=True, sharey=True,
ax = ax.flatten()
i=0
for img_path in os.listdir("images"):
    if img_path.endswith(".jpg"):
        img=mpimg.imread("images/"+img_path)
        ax[i].imshow(img, cmap='gray')
        #ax[i].imshow(img, interpolation='nearest', aspect='auto') #, cmap='Greys')
        i+=1


plt.show()