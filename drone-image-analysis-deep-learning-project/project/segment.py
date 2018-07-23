'''
	File name: segment.py
	Program Description: A program that uses Simple Linear Iterative Clustering (SLIC) algorithm to segment superpixels in a drone image.
	Date: 22 July 2018
	Author: Loria Roie Grace N. Malingan 
'''

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import measure
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

# a function that returns the x-coordinates and y-coordinates of the labeled regions
def sp_idx(s, index = True):
	u = np.unique(s)
	if index:
		return [np.where(s == i) for i in u]
	else:
		return [s[s == i] for i in u]


# load the image
image = cv2.imread("field.tif")
img = Image.open("field.tif")

numSegments = 1750
# apply SLIC and extract (approximately) the supplied number of segments
segments_slic = slic(image, n_segments = numSegments, sigma = 5, multichannel=True, convert2lab=True)

# Access the properties of labeled regions
regions = measure.regionprops(segments_slic, intensity_image=None, cache=True)
i=0
for prop in regions:
	i+=1
	if i <3:
		print(prop.centroid[0])
#print([r.area for r in regions])


# call the function that returns a tuple of xy coordinates
superpixel_list = sp_idx(segments_slic) 

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments_slic))
plt.axis("on")
 
# show the plots
plt.show()

