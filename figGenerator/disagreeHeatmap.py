# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:18:04 2017

@author: PengTian
"""


import pandas
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

numFile = pandas.read_excel('../../data/ropData/disagreementFigure.xlsx',sheetname = 1)

myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
myColors1 = (sns.color_palette("BuGn_r")[4],sns.color_palette("BuGn_r")[2], sns.color_palette("BuGn_r")[0] )
cmap = LinearSegmentedColormap.from_list('Custom', myColors1, len(myColors1))

sns.set(font_scale=1.5)
fig = plt.figure()
ax = sns.heatmap(numFile, cmap=cmap, linecolor='lightgray',yticklabels=5)

# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1.7, 1.0, 0.3333])
colorbar.set_ticklabels(['Plus', 'Pre-Plus', 'Normal'])
colorbar.ax.tick_params(labelsize=20) 
#for tick in ax.xaxis.get_major_ticks():
#                tick.label.set_fontsize(10) 

# X - Y axis labels
#ax.set_ylabel('Image Index',fontsize=15)
#ax.set_xlabel('Expert Index',fontsize=15)
plt.xlabel('Expert Index',fontsize=25)
plt.ylabel('Image Index', fontsize=25)

# Only y-axis labels need their rotation set, x-axis labels already have a rotation of 0
_, labels = plt.yticks()
plt.setp(labels, rotation=0)

plt.show()
fig.savefig('../../pic/disHeatMap.pdf',dpi=600)