import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt

mat1 = scipy.io.loadmat('15088.mat')
img1 = cv2.imread('15088.jpg')
mat2 = scipy.io.loadmat('2092.mat')
img2 = cv2.imread('2092.jpg')


# Convert the BRG image to RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Convert the RGB image to HSV
# img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

img1 = np.asarray(img1)
print(img1.shape)

label1 = np.asarray(mat1['groundTruth'][0][0][0][0][0])
print(label1.shape)

plt.subplot(1,2,1)
plt.imshow(img1)
plt.subplot(1,2,2)
plt.imshow(label1)
plt.show()

# function to cluster A
def k_means(A: np.ndarray, k: int, epochs=50):
    row = A.shape[0]
    col = A.shape[1]
    np.random.seed(10)
    centers = np.random.rand(k, 3)
    # center_i = centers[:,i]
    cluster = np.zeros((row, col))
    sum_of_squares = 0

    for epoch in range(epochs):
        # assigment step
        for i in range(row):
            for j in range(col):
                clt = -1
                dist = np.inf
                vector = A[i, j, :]
                for center in range(k):
                    new_dist = np.linalg.norm(vector - centers[center, :])
                    if new_dist < dist:
                        dist = new_dist
                        clt = center
                cluster[i, j] = clt

        # update step
        for i in range(k):
            sum = np.zeros(3)
            count = 0
            for row_idx in range(row):
                for col_idx in range(col):
                    if cluster[row_idx, col_idx] == i:
                        sum += A[row_idx, col_idx, :]
                        count += 1
            if count != 0:
                centers[i] = sum / count
            # print(f"At epoch {epoch}, sum for cluster {i} is {sum}, \ncenter is {centers[i]}")

        print(f"epoch {epoch}")
        if epoch % 3 == 0 or epoch == epochs - 1:
            # print(f"At {epoch}th epoch, the assigned clusters are: \n{cluster}")
            # compute sum of squares
            sse = 0
            for row_idx in range(row):
                for col_idx in range(col):
                    sse += np.linalg.norm(A[row_idx, col_idx, :] - centers[int(cluster[row_idx, col_idx]), :]) ** 2
            print(f"sum of squares is: {sse}")
            sum_of_squares = sse

    # a short summary
    for i in range(k):
        n = np.sum(cluster == i)
        print(f"When k = {k}, there are {n} pixels in cluster {i}")
    print()

    return centers, cluster, sum_of_squares


# # part 1
# centers, cluster, sse = k_means(img1, k=3)

# part 2
min_cluster = 10
max_cluster = 12
# sum_of_squares = np.zeros(max_cluster)
# for k in range(min_cluster, max_cluster+1):
#     centers_k, cluster_k, sse = k_means(img1, k=k)
#     sum_of_squares[k - 1] = sse
#
# print(sum_of_squares)

centers_k, cluster_k, sse = k_means(img1, k=2, epochs=30)
# print(sse)

plt.imshow(cluster_k)
plt.show()

# fig, ax = plt.subplots()
#
# x_idx = np.arange(1, max_cluster+1)
# ax.plot(x_idx, sum_of_squares)
# ax.set_xlim(min_cluster, max_cluster, 1)
# plt.show()


