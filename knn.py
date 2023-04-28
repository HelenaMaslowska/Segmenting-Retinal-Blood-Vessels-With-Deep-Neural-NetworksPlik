import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple
from image_processing import loadImage, showImage, loadImageTif

def get_image_paths(directory_path: str) -> List:
    paths = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".ppm"):
            paths.append(os.path.join(directory_path, filename))

    return sorted(paths)


def load_image(path: str, as_arr: bool = False):
    img = open(path)
    if as_arr:
        return np.asarray(img)
    else:
        return img
    

def slice_image(image: np.ndarray, slice_size: int) -> list:
    width, height = image.shape[0], image.shape[1]
    offset = slice_size//2

    if len(image.shape) == 3:
        sliced_image = np.zeros((width+2*offset, height+2*offset, 3))
    else:
        sliced_image = np.zeros((width+2*offset, height+2*offset))

    sliced_image[offset:height+offset, offset:width+offset] = image
    slices = []
    for i in range(height):
        for j in range(width):
            img_temp = sliced_image[i:i+slice_size, j:j+slice_size]
            slices.append(img_temp)
    return slices


def get_list_slices(image_list: list, slice_size) -> list:
    slices = []
    for image in image_list:
        image_slices = slice_image(image, slice_size)
        for img in image_slices:
            slices.append(img)
    return slices


def load_sliced_images( data_dir: str, target_dir: str, image_shape: Tuple[int, int] = (256, 256), slice_size: int = 9) -> Tuple[list, list]:

    data_paths, target_paths = get_image_paths(data_dir), get_image_paths(target_dir)

    data_images = [cv2.resize(loadImage(path, as_arr=True), image_shape) for path in data_paths]
    data_slices = get_list_slices(data_images, slice_size)

    target_images = [cv2.resize(loadImage(path, as_arr=True), image_shape) for path in target_paths]
    target_slices = get_list_slices(target_images, slice_size)

    return data_slices, target_slices

def init_knn_classifier(train_data, train_target) -> KNeighborsClassifier:

    classifier = KNeighborsClassifier(n_neighbors=4)
    classifier.fit(train_data, train_target)

    return classifier