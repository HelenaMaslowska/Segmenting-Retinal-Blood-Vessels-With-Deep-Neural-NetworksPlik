from typing import Literal
import skimage as ski
from skimage import io, exposure, data
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, data, filters, graph, measure, morphology
# data
filename = 'Data/01_dr.jpg'
mask_filename = 'Data/01_dr_mask.tif'
manual_filename = 'Data/01_dr.tif'


# read images
def loadImage(filename):
    image = ski.io.imread(filename)
    image = image.astype(np.float64)
    # if len(image.shape) == 3:
    #     image = ski.color.rgb2gray(image)
    image /= np.max(image)
    return image

def loadImageTif(mask_filename):
	im = Image.open(mask_filename)
	imarray = np.array(im)
	return imarray

def showImage(image, title=None, cmap: Literal["gray", "magma"] = "gray" , axis='on', shape=None ):
	if shape: 				image = image.reshape(shape)       
	plt.imshow(image, cmap=cmap)
	if title is not None: 	plt.title(title)
	if axis == 'off': 		plt.axis('off')

	plt.show()


# save image
def saveImage(image, filename):
    # data = (image * 255).astype(np.uint8)
    ski.io.imsave(filename, data)
