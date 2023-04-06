import skimage as ski
from skimage import io, exposure, data
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, data, filters, graph, measure, morphology

##############################################################################################
# Usefull functions for image processing
##############################################################################################

# extract green channel from image
def extract_green_channel(image):
	green_channel = image[:, :, 1]
	return green_channel

def sharpen(image):
	image = ski.filters.unsharp_mask(image, radius=20, amount=2) #20-40 radius, 1-2 amount
	return image

def brightness(image):
	image = ski.exposure.adjust_gamma(image, 0.8)
	return image

def contrast(image):
	image = ski.exposure.rescale_intensity(image, in_range=(0.009, 0.013))
	# image = ski.exposure.equalize_hist(image)
	return image

def extract_vessels(retina_source: np.ndarray):
	retina = color.rgb2gray(retina_source)
	t0, t1 = filters.threshold_multiotsu(retina, classes=3)
	mask = (retina > t0)
	# vessels = ski.filters.frangi(retina, sigmas=range(1, 10)) * mask
	vessels = filters.sato(retina, sigmas=range(1, 10)) * mask
	return vessels 	# return labeled - for better visualization

def threshold(vessels: np.ndarray):
	thresholded = filters.apply_hysteresis_threshold(vessels, 0.01, 0.03)
	labeled = ndi.label(thresholded)[0]
	return thresholded

##############################################################################################333
# Rest of the code is from main file that could be used in the future
##############################################################################################333
# create green image
def green_image(image):
	green_channel = image[:,:,1]
	green_img = np.zeros(image.shape)
	green_img[:,:,1] = green_channel
	return green_img

# find edges
def findEdges(image):
	thresh = 0.2
	image = scipy.ndimage.gaussian_filter(image, sigma=3)
	image = ski.filters.sobel(image) ** 0.5
	binary = (image > thresh) * 255
	binary = np.uint8(binary)
	return binary

def extract_vessels_copy(retina_source):
	# retina_source = data.retina()

	_, ax = plt.subplots()
	ax.imshow(retina_source)
	ax.set_axis_off()
	_ = ax.set_title('Human retina')

	retina = color.rgb2gray(retina_source)
	t0, t1 = filters.threshold_multiotsu(retina, classes=3)
	mask = (retina > t0)
	# vessels = ski.filters.frangi(retina, sigmas=range(1, 10)) * mask
	vessels = filters.sato(retina, sigmas=range(1, 10)) * mask

	_, axes = plt.subplots(nrows=1, ncols=2)
	axes[0].imshow(retina, cmap='gray')
	axes[0].set_axis_off()
	axes[0].set_title('grayscale')
	axes[1].imshow(vessels, cmap='magma')
	axes[1].set_axis_off()
	_ = axes[1].set_title('Sato vesselness')

	thresholded = filters.apply_hysteresis_threshold(vessels, 0.01, 0.03)
	labeled = ndi.label(thresholded)[0]

	_, ax = plt.subplots()
	ax.imshow(color.label2rgb(labeled, retina))
	ax.set_axis_off()
	_ = ax.set_title('thresholded vesselness')