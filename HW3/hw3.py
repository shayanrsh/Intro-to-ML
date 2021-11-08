import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(521)

class_means = np.array([[+0.0, +2.5],
                        [-2.5, -2.0],
                        [+2.5, -2.0]])

class_covariances = np.array([[[+3.2, +0.0],
                               [+0.0, +1.2]],
                              [[+1.2, +0.8],
                               [+0.8, +1.2]],
                              [[+1.2, -0.8],
                               [-0.8, +1.2]]])

class_sizes = np.array([120, 80, 100])

points1 = np.random.multivariate_normal(class_means[0, :], class_covariances[0, :, :], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1, :], class_covariances[1, :, :], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2, :], class_covariances[2, :, :], class_sizes[2])

X = np.vstack((points1, points2, points3))

# generating the corresponding labels
y_truth = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))

plt.figure(figsize=(6, 6))
plt.plot(points1[:, 0], points1[:, 1], "r.", markersize=10)
plt.plot(points2[:, 0], points2[:, 1], "g.", markersize=10)
plt.plot(points3[:, 0], points3[:, 1], "b.", markersize=10)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

number_of_classes = np.max(y_truth)
number_of_samples = X.shape[0]

y_truth_matrix = np.zeros((number_of_samples, number_of_classes)).astype(int)
y_truth_matrix[range(number_of_samples), y_truth - 1] = 1


def k_sigmoid(x, w, w_0):
    return 1 / (1 + np.exp(-(np.matmul(x, w) + w_0)))


def gradient_w(x, y_truth_matrix_gradient_w, y_predicted_gradient_w):
    return -np.asarray(
        [np.matmul((y_truth_matrix_gradient_w[:, i] - y_predicted_gradient_w[:, i])
                   * y_predicted_gradient_w[:, i] * (1 - y_predicted_gradient_w[:, i]), x) for i in
         range(y_truth_matrix_gradient_w[0].size)]).T


def gradient_w0(y_truth_matrix_gradient_w0, y_predicted_gradient_w0):
    return -np.sum(((y_truth_matrix_gradient_w0 - y_predicted_gradient_w0) * y_predicted_gradient_w0)
                   * (1 - y_predicted_gradient_w0), axis=0)


eta = 0.01
epsilon = 0.001

W = np.random.uniform(low=-0.01, high=0.01, size=(X.shape[1], number_of_classes))
w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, number_of_classes))

iteration = 1
objective_values = []

while True:
    y_predicted = k_sigmoid(X, W, w0)
    objective_values = np.append(objective_values, 0.5 * np.sum(np.power(y_truth_matrix - y_predicted, 2)))

    W_old = W
    w0_old = w0
    W = W - eta * gradient_w(X, y_truth_matrix, y_predicted)
    w0 = w0 - eta * gradient_w0(y_truth_matrix, y_predicted)

    if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((W - W_old) ** 2)) < epsilon:
        break
    iteration = iteration + 1

print('Parameter estimations:')
print(W)
print(w0)

plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

y_predicted = np.argmax(y_predicted, axis=1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames=['y_pred'], colnames=['y_truth'])
print('\n' + str(confusion_matrix))

# evaluate discriminant function on a grid
x1_interval = np.linspace(-6, +6, 1201)
x2_interval = np.linspace(-6, +6, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), number_of_classes))

for c in range(number_of_classes):
    discriminant_values[:, :, c] = W[0, c] * x1_grid + W[1, c] * x2_grid + w0[0, c]

A = discriminant_values[:, :, 0]
B = discriminant_values[:, :, 1]
C = discriminant_values[:, :, 2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:, :, 0] = A
discriminant_values[:, :, 1] = B
discriminant_values[:, :, 2] = C

plt.figure(figsize=(6, 6))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize=10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize=10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize=10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize=12, fillstyle="none")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 1], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 2], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 2], levels=0, colors="k")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
