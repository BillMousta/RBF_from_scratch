import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

class KMeans(object):
    def __init__(self, iterations, tol):
        # self.k = k
        self.iterations = iterations
        self.tol = tol
        self.clustering = {}

    """
    Initializing centers randomly
    """
    def initial_centers(self, data, k):
        centers = np.zeros((k, data.shape[1]))
        while True:
            # get random int numbers with range of size of data
            rand_num = [random.randint(0, data.shape[0]-1) for i in range(k)]

            # check if the number of the random centers are equal to k
            if len(set(rand_num)) == k:
                for i in range(k):
                    centers[i] = data[rand_num[i]]
                break
        return centers
    """
    The implementation of K means algorithm
    """
    def algorithm(self, data, k):
        centers = self.initial_centers(data, k)
        for i in range(self.iterations):
            self.clustering = {}

            # Create a k dictionary of list which will contain
            # the data for every cluster
            for j in range(k):
                self.clustering[j] = []

            for feature in data:
                # get the k  Euclidean distances for each center
                distances = [np.linalg.norm(feature - c)**2 for c in centers]
                # get the min distance from center
                min_dist = distances.index(min(distances))
                # adding in nearest cluster
                self.clustering[min_dist].append(feature)

            previous_centers = np.array(centers)
            for clf in self.clustering:
                """
                update centers according with mean of features
                for each cluster
                """
                centers[clf] = np.average(self.clustering[clf], axis=0)

            if self.stop_iteration(previous_centers, centers):
                # print(i)
                for j in range(k):
                    self.clustering[j] = np.array(self.clustering[j])
                break

        return centers, self.clustering

    """
    Stop K means algorithm if after 1 iteration the percent change
    of new centers are smaller than a tolerance
    """
    def stop_iteration(self, prev, cur):
        for i in range(cur.shape[0]):
            if np.sum((cur[i] - prev[i]) / prev[i]) > self.tol:
                return False
        return True


if __name__ == '__main__':
    arr = np.array([[1,1],
                    [2,3],
                    [10,4],
                    [15,16],
                    [7,1],
                    [6,3],
                    [4,7],
                    [20,10],
                    [15.2,4],
                    [3,3],
                    [4.5,20],
                    [0, 0],
                    [20,20],
                    [19,18],
                    [3,17],
                    [12,15],
                    [1,16]
                    ])

    k = KMeans(100, 0.0001)
    centers, data = k.algorithm(arr, 3)
    X = arr[:, 0]
    Y = arr[:, 1]
    # plt.plot(X, Y, 'o', label= 'class 1', color='b')

    print(centers)
    print(data)
    sns.set_theme()
    plt.figure()
    plt.title('3 Means Clustering')
    plt.ylabel('y-axis')
    plt.xlabel('x-axis')
    plt.plot(data[0][:, 0], data[0][:, 1], 'o', label= 'class 1', color='blue')
    plt.plot(centers[:, 0], centers[:, 1], 'x', color='blue')

    plt.plot(data[1][:, 0], data[1][:, 1], 'o', label='class 2', color='red')
    plt.plot(centers[:, 0], centers[:, 1], 'x', color='red')

    plt.plot(data[2][:, 0], data[2][:, 1], 'o', label='class 3', color='black')
    plt.plot(centers[:, 0], centers[:, 1], 'x', color='black')

    plt.legend()
    plt.show()
