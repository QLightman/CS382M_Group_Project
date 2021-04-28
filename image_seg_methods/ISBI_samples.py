import scipy.io, scipy.misc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy import linalg, ndimage
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from PIL import Image
from libtiff import TIFF

# read images
test_img_tif = TIFF.open('test-volume.tif')

# to read an image in the current TIFF directory and return it as numpy array:
image = []
for img in test_img_tif.iter_images():  # do stuff with image
    image.append(img)

image = np.asarray(image)
print(image.shape)    # (30, 512, 512)
num, row, col = image.shape


def find_class_x(img: np.ndarray, label_pred: np.ndarray, k):
    bin = [0] * k
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i,j] < 80:
                bin[label_pred[i,j]] += 1
    return np.argmax(bin)


def segmentation(method: str, images, num_cluster=2):
    print(f"method: {method}")
    labels = []
    for idx in range(num):
        if idx % 3 == 0:
            print(f"idx: {idx}")
        original = images[idx]
        label = np.zeros_like(images[idx])
        img = original.reshape((-1, 1))
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=num_cluster, max_iter=1000, random_state=0).fit(img)
            kmeans_cluster = kmeans.predict(img)
            kmeans_cluster = kmeans_cluster.reshape((row, col))
            x = find_class_x(original, kmeans_cluster, num_cluster)
            label = (kmeans_cluster != x)
        if method == 'gaussian_full':
            gm1 = GaussianMixture(n_components=num_cluster, covariance_type='full', max_iter=1000, random_state=0).fit(img)
            gm1_cluster = gm1.predict(img)
            gm1_cluster = gm1_cluster.reshape((row, col))
            x = find_class_x(original, gm1_cluster, num_cluster)
            label = (gm1_cluster != x)
        if method == 'gaussian_tied':
            gm2 = GaussianMixture(n_components=num_cluster, covariance_type='tied', max_iter=1000, random_state=0).fit(img)
            gm2_cluster = gm2.predict(img)
            gm2_cluster = gm2_cluster.reshape((row, col))
            x = find_class_x(original, gm2_cluster, num_cluster)
            label = (gm2_cluster != x)
        if method == 'agglo':
            size = 128
            num_iter = int(row/size)
            for i in range(num_iter):
                for j in range(num_iter):
                    subimage = original[i*size:(i+1)*size, j*size:(j+1)*size]
                    img = subimage.reshape((-1,1))
                    agglo = AgglomerativeClustering(n_clusters=num_cluster).fit(img)
                    agglo_cluster = agglo.labels_
                    agglo_cluster = agglo_cluster.reshape((size, size))
                    x = find_class_x(subimage, agglo_cluster, num_cluster)
                    label[i*size:(i+1)*size, j*size:(j+1)*size] = (agglo_cluster != x)

        # label = ndimage.median_filter(label, size=3)
        # label = ndimage.median_filter(label, size=3)
        # label = ndimage.median_filter(label, size=3)
        labels.append(label)

    labels = np.asarray(labels)
    labels = np.logical_not(labels)
    return labels


kmeans_pred = segmentation('kmeans', image)
print(kmeans_pred.shape)    # (30, 512, 512)
tifffile.imsave('kmeans_k_2.tif', kmeans_pred)

gm_full_pred = segmentation('gaussian_full', image)
tifffile.imsave('gaussian_full_k_2.tif', gm_full_pred)

gm_tied_pred = segmentation('gaussian_tied', image)
tifffile.imsave('gaussian_tied_k_2.tif', gm_tied_pred)

agglo_pred = segmentation('agglo', image)
tifffile.imsave('agglo_k_2.tif', agglo_pred)



