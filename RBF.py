import KMeans
import numpy as np
import pandas as pd

class RBF(object):
    def __init__(self, tol):
        self.centers = []
        self.tol = tol
        self.weights = []
        self.gamma = 0


    """
    Training of RBF classifier
    """
    def fit(self, X, y, k, l, method):
        # K Means algorithm to reduce the size of hidden layer from
        # data.size to k
        kmeans = KMeans.KMeans(100, self.tol)
        # The mean of each cluster
        self.centers, clusters = kmeans.algorithm(X, k)

        dmax = self.d_max()
        self.gamma = 1 / (dmax/(2*k))**2

        if method == 'RLS':
            self.RLS(X, y, k, l)
        else:
            self.pseudo_inverse(X, y, k)

        """
        create a vector of kernel with size k
        Φ_j(x_i) = rbf(x_i, μ_j) i = 1,2,...,N, j = 1,...,k \
        where μ_j the centers from K means
        hidden layer
        """
        # print(self.centers)
        # F = np.zeros([len(X),k])
        # # Φ matrix
        # for i in range(len(X)):
        #     for j in range(k):
        #         F[i][j] = rbf(X[i], self.centers[j], self.gamma)
        #
        # # print(F)
        # F_pseudoinverse = np.linalg.pinv(F)
        # self.weights = F_pseudoinverse*y
        # self.weights = self.weights.transpose()

        # Get the weights according with RLS algorithm
        # self.RLS(X, y, k, l)

    """
    Implementation algorithm of RLS algorithm 
    for training in the output
    """
    def RLS(self, X, y, k, l):
        # initial weights to 0
        # self.weights = np.zeros(k)
        self.weights = np.random.randn(k)
        self.bias = np.random.randn(1)
        P = l*np.identity(k)
        avg_loss = 0

        for i, sample in enumerate(X):
            """
            create a vector of kernel with size k
            Φ_j(x_i) = rbf(x_i, μ_j) i = 1,2,...,N, j = 1,...,k 
            where μ_j the centers from K means
            hidden layer
            """
            phi = np.array([rbf(sample, c, self.gamma) for c in self.centers])
            F = np.dot(phi, self.weights) + self.bias

            # loss = (y[i] - F).flatten() ** 2
            # avg_loss += loss
            # print('Loss: {0:.2f}'.format(loss[0]))

            """
            RLS 
            P[n] = P[n-1] - (P[n-1]*Φ[n]*Φ^T[n]* P[n-1])/(1 + Φ^T[n]*P[n-1]*Φ[n])
            g[n] = P[n]*Φ[n]
            a[n] = d[n] - w^T[n-1]*Φ[n]
            w[n] = w[n-1] + g[n]*a[n]  
            """
            P = P - (P * np.dot(phi, phi) * P)/(1 + np.dot(phi, P*phi))
            g = np.matmul(P,phi)
            a = y[i] - np.dot(self.weights, phi)

            # update weights according with this form
            self.weights = self.weights + g*a


    """
    Training weights with pseuso inverse matrix G+ = (G^T G)^-1 G^T
    w = G+ * y where y is the training labels
    """
    def pseudo_inverse(self, X, y, k):
        G = np.zeros([len(X), k])
        # G matrix
        for i in range(len(X)):
            for j in range(k):
                G[i][j] = rbf(X[i], self.centers[j], self.gamma)

        # print(G)
        # Pseudo inverse
        G_plus = np.linalg.pinv(G)
        self.weights =G_plus * y
        self.weights = self.weights.transpose()

    """
    Prediction according with the form
    F = sum(w, Φ(x,t_i))
    """
    def predict(self, X, method):
        # print(self.weights)
        y_pred = []
        if method == 'RLS':
            for i, sample in enumerate(X):
                phi = np.array([rbf(sample, c, self.gamma) for c in self.centers])
                F = np.dot(self.weights, phi)
                y_pred.append(np.sign(F))
        else:
            for i, sample in enumerate(X):
                phi = np.array([rbf(sample, c, self.gamma) for c in self.centers])
                F = np.dot(self.weights[i], phi)
                y_pred.append(np.sign(F))

        return y_pred

    """
    F = dot(w,phi) + bias
    sign function -1, 1
    """
    # def predict(self, X, actual):
    #     # print(self.weights)
    #     y_pred = []
    #     avg_loss = 0
    #     for i, sample in enumerate(X):
    #         phi = np.array([rbf(sample, c, self.gamma) for c in self.centers])
    #         F = np.dot(self.weights, phi) + self.bias
    #         # MSE error
    #         avg_loss += (actual[i] - F).flatten() ** 2
    #         y_pred.append(np.sign(F))
    #     test_loss = avg_loss/len(X)
    #     y_pred = np.array(y_pred)
    #     y_pred = y_pred.reshape(y_pred.shape[0])
    #     correct = self.accuracy_metric(actual, y_pred)
    #     print(f"Test Error: \n Accuracy: {(correct):>0.1f}%, Avg loss: {float(test_loss):>8f} \n")
    #
    #     return y_pred, correct/100, test_loss

    """
    Return the max distance of all centers
    for gamma parameter of RBF kernel
    """
    def d_max(self):
        distances = []
        for center in self.centers:
            distances = [np.linalg.norm(center - c) ** 2 for c in self.centers]
        return max(distances)


    """
    Return accuracy
    """
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0



def rbf(x, y, gamma):
    return np.exp(-gamma*np.linalg.norm(x-y)**2)
