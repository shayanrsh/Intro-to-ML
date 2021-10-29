import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

np.random.seed(421)

# Mean vector
mu = np.array([[0.0, 2.5],
               [-2.5, -2.0],
               [2.5, -2.0]])

# Covariance matrix
cov = np.array([[[3.2, 0.0], [0.0, 1.2]],
                [[1.2, 0.8], [0.8, 1.2]],
                [[1.2, -0.8], [-0.8, 1.2]]])

# Data point Size
n = np.array([120, 80, 100])

# Generating random data points from three bivariant Gaussian densities
gen_data1 = np.random.multivariate_normal(mu[0, :], cov[0, :, :], size=n[0])
gen_data2 = np.random.multivariate_normal(mu[1, :], cov[1, :, :], size=n[1])
gen_data3 = np.random.multivariate_normal(mu[2, :], cov[2, :, :], size=n[2])

# Visualizing the data
plt.scatter(gen_data1[:, 0], gen_data1[:, 1])
plt.scatter(gen_data2[:, 0], gen_data2[:, 1])
plt.scatter(gen_data3[:, 0], gen_data3[:, 1])

plt.show()

# Predicting the mean values
mu1_hat = np.array([[np.mean(gen_data1[:, 0])], [np.mean(gen_data1[:, 1])]])
mu2_hat = np.array([[np.mean(gen_data2[:, 0])], [np.mean(gen_data2[:, 1])]])
mu3_hat = np.array([[np.mean(gen_data3[:, 0])], [np.mean(gen_data3[:, 1])]])

sample_mean = [mu1_hat, mu2_hat, mu3_hat]
print('sample_mean: ', sample_mean)

# Predicting the covariance matrix
cov1_hat = np.cov(gen_data1[:, 0], gen_data1[:, 1])
cov2_hat = np.cov(gen_data2[:, 0], gen_data2[:, 1])
cov3_hat = np.cov(gen_data3[:, 0], gen_data3[:, 1])
print('\n sample_covariances: \n', cov1_hat, '\n', cov2_hat, '\n', cov3_hat)

# Estimating class priors
p1_hat = n[0] / np.sum(n)
p2_hat = n[1] / np.sum(n)
p3_hat = n[2] / np.sum(n)
print('\n class_priors: ', p1_hat, p2_hat, p3_hat, '\n')

# Calculating W, w, and w0 for three data points
W1 = -1 / 2 * inv(cov1_hat)
w1 = np.dot(inv(cov1_hat), mu1_hat)
w1_0 = np.dot(-1 / 2 * (np.dot(mu1_hat.T, inv(cov1_hat))), mu1_hat) - np.log(2 * np.pi) \
       - 1 / 2 * np.log(np.linalg.det(cov1_hat)) \
       + np.log(p1_hat)

W2 = -1 / 2 * inv(cov2_hat)
w2 = np.dot(inv(cov2_hat), mu2_hat)
w2_0 = np.dot(-1 / 2 * (np.dot(mu2_hat.T, inv(cov2_hat))), mu2_hat) - np.log(2 * np.pi) \
       - 1 / 2 * np.log(np.linalg.det(cov2_hat)) \
       + np.log(p2_hat)

W3 = -1 / 2 * inv(cov3_hat)
w3 = np.dot(inv(cov3_hat), mu3_hat)
w3_0 = np.dot(-1 / 2 * (np.dot(mu3_hat.T, inv(cov3_hat))), mu3_hat) - np.log(2 * np.pi) \
       - 1 / 2 * np.log(np.linalg.det(cov3_hat)) \
       + np.log(p3_hat)

# transpose the data points for ease of use
gen_data1 = gen_data1.T
gen_data2 = gen_data2.T
gen_data3 = gen_data3.T

# To avoid error empty list is needed before for loop
y_pred1, y_pred2, y_pred3 = [], [], []

"""
g(x) = xT.Wc.x + wc.T x + wc_0

Using the formula above score is calculated for each data point.
The maximum leads to classification.
"""

for i in range(n[0]):

    x = gen_data1[:, i]

    g1_x = np.dot(np.dot(x.T, W1), x) + np.dot(w1.T, x) + w1_0
    g2_x = np.dot(np.dot(x.T, W2), x) + np.dot(w2.T, x) + w2_0
    g3_x = np.dot(np.dot(x.T, W3), x) + np.dot(w3.T, x) + w3_0

    y_pred1.append([g1_x, g2_x, g3_x].index(max([g1_x, g2_x, g3_x])))

for i in range(n[1]):

    x = gen_data2[:, i]

    g1_x = np.dot(np.dot(x.T, W1), x) + np.dot(w1.T, x) + w1_0
    g2_x = np.dot(np.dot(x.T, W2), x) + np.dot(w2.T, x) + w2_0
    g3_x = np.dot(np.dot(x.T, W3), x) + np.dot(w3.T, x) + w3_0

    y_pred2.append([g1_x, g2_x, g3_x].index(max([g1_x, g2_x, g3_x])))

for i in range(n[2]):

    x = gen_data3[:, i]

    g1_x = np.dot(np.dot(x.T, W1), x) + np.dot(w1.T, x) + w1_0
    g2_x = np.dot(np.dot(x.T, W2), x) + np.dot(w2.T, x) + w2_0
    g3_x = np.dot(np.dot(x.T, W3), x) + np.dot(w3.T, x) + w3_0

    y_pred3.append([g1_x, g2_x, g3_x].index(max([g1_x, g2_x, g3_x])))

# Calculating the confusion matrix for the data points
conf_matrix1 = {i: y_pred1.count(i) for i in y_pred1}
conf_matrix2 = {i: y_pred2.count(i) for i in y_pred2}
conf_matrix3 = {i: y_pred3.count(i) for i in y_pred3}

print(f"{'y_truth':<10} {'1':<10} {'2':<10} {'3':<10}")
print('y_pred')
print(f"{'1':<10} {conf_matrix1.get(0) :<10} {conf_matrix1.get(1):<10} {conf_matrix1.get(2):<10}")
print(f"{'2':<10} {conf_matrix2.get(0):<10} {conf_matrix2.get(1):<10} {conf_matrix2.get(2):<10}")
print(f"{'3':<10} {conf_matrix3.get(0):<10} {conf_matrix3.get(1):<10} {conf_matrix3.get(2):<10}")

print('\n Please be patient for the plot with decision boundaries')

# Drawing decision boundaries
x1_interval = np.linspace(-7, +6, 501)
x2_interval = np.linspace(-6, +6, 501)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)

FULL_X = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T

g1_x = [np.dot(np.dot(np.transpose(XX), W1), XX) + np.dot(np.transpose(w1), XX) + w1_0 for XX in FULL_X]
g2_x = [np.dot(np.dot(np.transpose(XX), W2), XX) + np.dot(np.transpose(w2), XX) + w2_0 for XX in FULL_X]
g3_x = [np.dot(np.dot(np.transpose(XX), W3), XX) + np.dot(np.transpose(w3), XX) + w3_0 for XX in FULL_X]

g1_x = np.array(g1_x).reshape(-1)
g2_x = np.array(g2_x).reshape(-1)
g3_x = np.array(g3_x).reshape(-1)

# put g1_x, g2_x, and g3_x together in scores variable
scores = np.vstack((g1_x, g2_x, g3_x))

y_predicted = np.argmax(scores, axis=0)

discriminant_values = y_predicted.reshape(x1_grid.shape)

gen_data = np.vstack([gen_data1.T, gen_data2.T, gen_data3.T])
y_truth = np.array([0] * gen_data1.shape[1] + [1] * gen_data2.shape[1] + [2] * gen_data3.shape[1])
y_predicted = np.array(y_pred1 + y_pred2 + y_pred3)

# transposing the data again to be as same as the first generated one
gen_data1 = np.transpose(gen_data1)
gen_data2 = np.transpose(gen_data2)
gen_data3 = np.transpose(gen_data3)

# Visualizing the data
plt.scatter(gen_data1[:, 0], gen_data1[:, 1])
plt.scatter(gen_data2[:, 0], gen_data2[:, 1])
plt.scatter(gen_data3[:, 0], gen_data3[:, 1])

plt.plot(gen_data[y_predicted != y_truth, 0], gen_data[y_predicted != y_truth, 1], "ko", markersize=12,
         fillstyle="none")
plt.contour(x1_grid, x2_grid, discriminant_values, levels=2, colors="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
