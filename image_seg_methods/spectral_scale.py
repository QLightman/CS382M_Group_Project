import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.cluster import KMeans


def spectral_fit(img1, k):
    (row, col, _) = img1.shape
    img_new0 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    img_new1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img_new2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img1 = img1.reshape((-1, 3))
    img_new0 = img_new0.reshape((-1, 4))
    img_new1 = img_new1.reshape((-1, 3))
    img_new2 = img_new2.reshape((-1, 1))

    img1 = np.concatenate((img_new0, img_new1, img_new2), axis=-1)

    norm_of_rows = linalg.norm(img1, axis=1)
    normalized_img1 = img1 / norm_of_rows[:, np.newaxis]

    # print(normalized_img1.shape)  # (154401, 3)

    ones = np.ones((img1.shape[0], 1))
    prod = np.dot(np.transpose(normalized_img1), ones)
    diag_D = np.dot(normalized_img1, prod) - 1
    X_tilde = diag_D ** (-1 / 2) * normalized_img1

    # print(np.max(diag_D), np.min(diag_D))

    u, s, vt = linalg.svd(X_tilde, full_matrices=False)

    # print(u.shape)
    # u = u[:, 0:2]
    norm_of_u = linalg.norm(u, axis=1)
    normalized_u = u / norm_of_u[:, np.newaxis]

    num_cluster = k
    kmeans = KMeans(n_clusters=num_cluster, max_iter=1000, random_state=0).fit(normalized_u)
    # gm1 = GaussianMixture(n_components=num_cluster, covariance_type='full', max_iter=1000, random_state=0).fit(normalized_u)
    # gm2 = GaussianMixture(n_components=num_cluster, covariance_type='tied', max_iter=1000, random_state=0).fit(normalized_u)

    kmeans_cluster = kmeans.predict(normalized_u)
    # gm1_cluster = gm1.predict(normalized_u)
    # gm2_cluster = gm2.predict(normalized_u)

    kmeans_cluster = kmeans_cluster.reshape((row, col))

    return kmeans_cluster


