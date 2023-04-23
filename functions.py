import skimage as ski
from skimage import io, exposure, data
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, data, filters, graph, measure, morphology
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced
##############################################################################################
# Usefull functions for image processing
##############################################################################################

# extract green channel from image
def extract_green_channel(image: np.ndarray) -> np.ndarray:
	green_channel = image[:, :, 1]
	return green_channel

def sharpen(image: np.ndarray) -> np.ndarray:
	image = ski.filters.unsharp_mask(image, radius=20, amount=2) #20-40 radius, 1-2 amount
	return image

def brightness(image: np.ndarray) -> np.ndarray:
	image = ski.exposure.adjust_gamma(image, 0.8)
	return image

def contrast(image: np.ndarray) -> np.ndarray:
	image = ski.exposure.rescale_intensity(image, in_range=(0.009, 0.013))
	# image = ski.exposure.equalize_hist(image)
	return image

def extract_vessels(retina_source: np.ndarray) -> np.ndarray:
	'''
	Extracts vessels from retina image
	:param retina_source: image of retina
	:return: image of vessels [0. 0. 0. ... 0. 0. 0.]
	'''
	retina = color.rgb2gray(retina_source)
	t0, t1 = filters.threshold_multiotsu(retina, classes=3)
	mask = (retina > t0)
	# vessels = ski.filters.frangi(retina, sigmas=range(1, 10)) * mask
	vessels = filters.sato(retina, sigmas=range(1, 10)) * mask
	return vessels 	# return labeled - for better visualization

def extract_and_use_vessels(retina_source: np.ndarray) -> np.ndarray:
	'''
	Extracts vessels from retina image and applies some filters
	:param retina_source: image of retina
	:return: image of vessels [False False False ... False False False]
	'''
	vessels = extract_vessels(retina_source)
	vessels = contrast(vessels)
	vessels = threshold(vessels)
	return vessels

def threshold(vessels: np.ndarray) -> np.ndarray:
	thresholded = filters.apply_hysteresis_threshold(vessels, 0.01, 0.03)
	labeled = ndi.label(thresholded)[0]
	return thresholded

def add_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
	return image * mask

def show_stats(vessels: np.ndarray, manual: np.ndarray, reportlib = "sklearn.metrics") -> None:
	y_pred = vessels.flatten()		# True - żyły, False - tło
	y_true = manual.flatten()		# 255 - żyły, 0 - tło
	model_mask = (y_true > 0)		# maska pikseli, które reprezentują żyły w modelu ekspertskim

	y_true_masked = np.zeros(y_true.shape, dtype=bool)		# utworzenie maski True/False dla modelu ekspertskiego
	y_true_masked[model_mask] = True

	accuracy = accuracy_score(y_true_masked.flatten(), y_pred.flatten())				# trafność - accuracy
	sensitivity = recall_score(y_true_masked.flatten(), y_pred.flatten()) 				# czułość - sensitivity, recall
	specificity = recall_score(y_true_masked.flatten(), y_pred.flatten(), pos_label=0)	# swoistość - specificity
	precision = precision_score(y_true_masked.flatten(), y_pred.flatten()) 				# precyzja - precision
	cm = confusion_matrix(y_true_masked.flatten(), y_pred.flatten())
	# print(confusion_matrix(y_true_masked.flatten(), y_pred.flatten()).flatten(), end="\n\n")
	print(f"Accuracy score:\t\t {accuracy:.6f}")
	print(f"Sensitivity score:\t {sensitivity:.6f}")
	print(f"Specificity score:\t {specificity:.6f}")
	print(f"Precision score:\t {precision:.6f}")
	print("G-mean: \t\t", round(np.sqrt(sensitivity * specificity), 6))
	print("Weighted average: \t", round((sensitivity + specificity) / 2, 6))

	if reportlib == "sklearn.metrics":
		print("\nClassification report with sklearn.metrics")
		print(classification_report(y_true_masked.flatten(), y_pred.flatten(), target_names=["background", "vessels"]))
	else:
		print("\nClassification report with imblearn.metrics")
		print(classification_report_imbalanced(y_true_masked, y_pred, target_names=['background', 'vessels']))
	return cm

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

def interpolate(image):
	min, max = image.min(), image.max()
	arr_subtracted = image - min  # Subtract the minimum
	# array([  0,  38,  98, 203, 248], dtype=uint8)
	arr_divided = arr_subtracted / (max - min)  # Divide by new max
	# array([0.        , 0.15322581, 0.39516129, 0.81854839, 1.        ])
	arr_multiplied = arr_divided * 255  # Multiply by dtype max
	# array([  0.        ,  39.07258065, 100.76612903, 208.72983871,
	#        255.        ])
	# Convert dtype to original uint8
	arr_rescaled = np.asarray(arr_multiplied, dtype=image.dtype)
	# array([  0,  39, 100, 208, 255], dtype=uint8)
	return arr_rescaled

def erase_above_average(image):
	average = image.mean(axis=0).mean(axis=0)
	print(average)
	for i in range(len(image)):
		for j in range(len(image[i])):
			if image[i][j] < average/2:
				image[i][j] = average
	return image

# filtr frangi
def frangi(image):
	image = ski.filters.frangi(image, black_ridges=True)
	return image