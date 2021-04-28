import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from libtiff import TIFF

# read images
train_img_tif = TIFF.open('train-volume.tif')
train_label_tif = TIFF.open('train-labels.tif')
# to read an image in the current TIFF directory and return it as numpy array:
image = []
label = []
for img in train_img_tif.iter_images():  # do stuff with image
    image.append(img)
for lb in train_label_tif.iter_images():
    label.append(lb)

image, label = np.asarray(image), np.asarray(label)
row, col = image[0].shape  # 512, 512

## display image
# plt.subplot(1,2,1)
# plt.imshow(image[29][0:64, 0:64], cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(label[29][0:64, 0:64], cmap='gray')
# plt.show()

# spectral clustering first version
size = 64
n = size ** 2
W = np.zeros((n, n))  # (2^12, 2^12)
D = np.zeros_like(W)


def img_to_W(i, j):
    return i * size + j


def W_to_img(idx):
    return (idx // size, idx % size)


img = image[0]
for i in range(W.shape[0]):
    if i % 200 == 0:
        print(f"i: {i}")
    for j in range(W.shape[1]):

        img_i = W_to_img(i)
        img_j = W_to_img(j)
        dist_ij = np.abs(img_i[0] - img_j[0]) + np.abs(img_i[1] - img_j[1])

        if i == j:
            W[i, j] = 0
        elif dist_ij <= 20:
            alpha = 0.01
            pixel_i = img[img_i[0], img_i[1]]
            pixel_j = img[img_j[0], img_j[1]]
            diff = -alpha * (float(pixel_i) - float(pixel_j)) ** 2
            if diff >= -10:
                W[i, j] = np.exp(diff)

for i in range(n):
    D[i, i] = np.sum(W[i, :])

L = D - W

u, v = linalg.eig(a=L, b=D)
print(u)

sort_idx = np.argsort(u)[:2]

V = np.zeros((n, 2))
V[:, 0] = v[sort_idx[0]]
V[:, 1] = v[sort_idx[1]]
print(f"V: {V}")


# function to cluster rows of V
def k_means(V: np.ndarray, k=2, epochs=20):
    # V: 2^12 * 2
    row = V.shape[0]  # 2^12
    col = V.shape[1]  # 2
    centers = np.zeros((k, col))  # 2 * 2
    rand_indices = np.random.randint(low=0, high=row, size=2)
    centers[0] = V[rand_indices[0], :]
    centers[1] = V[rand_indices[1], :]
    # center_i = centers[:,i]
    cluster = np.zeros(row)

    for epoch in range(epochs):
        # assigment step
        for i in range(row):
            vector = V[i, :]
            dist_0 = np.linalg.norm(vector - centers[0, :])
            dist_1 = np.linalg.norm(vector - centers[1, :])
            distance = np.asarray([dist_0, dist_1])
            cluster[i] = np.argmin(distance)

        for i in range(k):
            n = np.sum(cluster == i)
            print(f"At epoch {epoch}, there are {n} row vectors in cluster {i}")

        # update step
        for i in range(k):
            sum = np.zeros_like(V[0, :])
            count = 0
            for row_idx in range(row):
                if cluster[row_idx] == i:
                    sum += V[row_idx, :]
                    count += 1
            if count != 0:
                centers[i] = sum / count
            # print(f"At epoch {epoch}, sum for cluster {i} is {sum}, \ncenter is {centers[i]}")

    # # a short summary
    # for i in range(k):
    #     n = np.sum(cluster == i)
    #     print(f"When k = {k}, there are {n} row vectors in cluster {i}")
    # print()

    return centers, cluster


centers, cluster = k_means(V)
pred = np.zeros((size, size))
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        pred[i, j] = cluster[i * size + j]

plt.subplot(1, 3, 1)
plt.imshow(image[0][0:size, 0:size], cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(pred, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(label[0][0:size, 0:size], cmap='gray')
plt.show()
