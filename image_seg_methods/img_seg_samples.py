import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral_scale
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture

mat1 = scipy.io.loadmat('238011.mat')
img1 = cv2.imread('238011.jpg')
# mat2 = scipy.io.loadmat('15088.mat')
# img2 = cv2.imread('15088.jpg')
#
# # mat1 = scipy.io.loadmat('12003.mat')
# # img1 = cv2.imread('12003.jpg')
# mat1 = scipy.io.loadmat('2092.mat')
# img1 = cv2.imread('2092.jpg')
label0 = np.asarray(mat1['groundTruth'][0][0][0][0][0])
label1 = np.asarray(mat1['groundTruth'][0][1][0][0][0])
label2 = np.asarray(mat1['groundTruth'][0][2][0][0][0])
label3 = np.asarray(mat1['groundTruth'][0][3][0][0][0])
label4 = np.asarray(mat1['groundTruth'][0][4][0][0][0])

plt.subplot(2,3,1)
plt.imshow(img1)
plt.subplot(2,3,2)
plt.imshow(label0)
plt.subplot(2,3,3)
plt.imshow(label1)
plt.subplot(2,3,4)
plt.imshow(label2)
plt.subplot(2,3,5)
plt.imshow(label3)
plt.subplot(2,3,6)
plt.imshow(label4)
plt.show()

img1 = np.asarray(img1)
# img1 = np.asarray(img1)[0:220, :]
original1 = img1
(row, col, _) = img1.shape
print(img1.shape)

# label1 = label2
# label1 = np.asarray(mat1['groundTruth'][0][1][0][0][0])[0:220, :]
print(label1.shape)

plt.subplot(1,2,1)
plt.imshow(img1)
plt.subplot(1,2,2)
plt.imshow(label1)
plt.show()

img1 = img1.reshape((-1, 3))
num_cluster = 4

kmeans = KMeans(n_clusters=num_cluster, max_iter=1000, random_state=0).fit(img1)
gm1 = GaussianMixture(n_components=num_cluster, covariance_type='full', max_iter=1000, random_state=0).fit(img1)
gm2 = GaussianMixture(n_components=num_cluster, covariance_type='tied', max_iter=1000, random_state=0).fit(img1)
# gm3 = GaussianMixture(n_components=num_cluster, covariance_type='spherical', max_iter=1000, random_state=0).fit(img1)
# spectral = SpectralClustering(n_clusters=num_cluster, affinity='nearest_neighbors', n_neighbors=int(row/2), random_state=0).fit(img1)
# agglo = AgglomerativeClustering(n_clusters=num_cluster).fit(img1)
# print(gm1.means_)
# dbscan = DBSCAN(eps=1.5).fit(img1)
# print(f"labels: {dbscan.labels_}")
# meanshift = MeanShift(bandwidth=2).fit(img1)

kmeans_cluster = kmeans.predict(img1)
gm1_cluster = gm1.predict(img1)
gm2_cluster = gm2.predict(img1)
# gm3_cluster = gm3.predict(img1)
# agglo_cluster = agglo.predict(img1)
# print(f"affinity matrix: {spectral.affinity_matrix_}")
# spectral_cluster = spectral.labels_
# agglo_cluster = agglo.labels_

# print(gm1_cluster.shape)
kmeans_cluster = kmeans_cluster.reshape((row, col))
gm1_cluster = gm1_cluster.reshape((row, col))
gm2_cluster = gm2_cluster.reshape((row, col))
# gm3_cluster = gm3_cluster.reshape((row, col))
# spectral_cluster = spectral_cluster.reshape((row, col))
# agglo_cluster = agglo_cluster.reshape((row, col))
# dbscan_cluster = dbscan.labels_.reshape((row, col))
# meanshift_cluster = meanshift.labels_.reshape((row, col))

spectral_cluster = spectral_scale.spectral_fit(original1, num_cluster)

plt.subplot(2,3,1)
plt.imshow(original1)
plt.title("original")
plt.subplot(2,3,2)
plt.imshow(label1)
plt.title("label")
plt.subplot(2,3,3)
plt.imshow(kmeans_cluster)
plt.title("kmeans")
plt.subplot(2,3,4)
plt.imshow(gm1_cluster)
plt.title("gaussian mix (full)")
plt.subplot(2,3,5)
plt.imshow(gm2_cluster)
plt.title("gaussian mix (tied)")
plt.subplot(2,3,6)
plt.imshow(spectral_cluster)
plt.title("spectral")
plt.show()


def get_accuracy(label, pred):
    (r, c) = label.shape
    count = 0
    for i in range(r):
        for j in range(c):
            if label[i,j] == pred[i,j]:
                count += 1
    return count/(r*c)


def get_accuracy_specific(label, pred):
    (r, c) = label.shape
    test = np.zeros_like(pred)
    for i in range(r):
        for j in range(c):
            if pred[i,j] == 0:
                test[i,j] = 1
            elif pred[i,j] == 2:
                test[i,j] = 3
            else:
                test[i,j] = 2

    accu = get_accuracy(label, test)
    return accu


# kmeans_cluster = (kmeans_cluster == 0)
# print(f"accuracy_k_means: {get_accuracy(label1, kmeans_cluster + 1)}")
# print(f"accuracy_gaussian mix (full): {get_accuracy(label1, gm1_cluster + 1)}")
# print(f"accuracy_gaussian mix (tied): {get_accuracy(label1, gm2_cluster + 1)}")
# print(f"accuracy_spectral: {get_accuracy(label1, (spectral_cluster) + 1)}")

print(f"accuracy_k_means: {get_accuracy_specific(label1, kmeans_cluster)}")
print(f"accuracy_gaussian mix (tied): {get_accuracy_specific(label1, gm2_cluster)}")


