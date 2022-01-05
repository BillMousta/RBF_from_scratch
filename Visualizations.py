import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

"""
Visualizing the number of each class in binary
classification problem
"""
def visualize_class_distribution(data,title):
    sns.set_theme()
    dict = {'Buy': data[data==1].size, 'Sell': data[data==-1].size}
    fig = plt.figure(figsize=(12, 8))

    plt.title(title)
    plt.bar('Sell', height=dict['Sell'], color='r', label='Sell')
    plt.bar('Buy', height=dict['Buy'], color='b', label='Buy')
    plt.xlabel('Movement')
    plt.ylabel('No of Labels')
    plt.legend(loc=(1, 0.92))
    plt.show()

def histogram(data, title):
    sns.set_theme()
    # fig = plt.figure(figsize=(12,10), dpi=200)
    plt.hist(data, bins=50)
    plt.title(title)
    plt.show()

def visualize_data(data, y, title):
    sns.set_theme()
    fig = plt.figure(figsize=(12,8), dpi=200)
    # X, Y = np.split(data,[-1],axis=1)
    X = data[0, 0:]
    Y = data[1, 0:]
    plt.plot(X, Y, 'o', label='Class 1', markevery=(y == 1))
    plt.plot(X, Y, 'o', label='Class -1', markevery=(y == -1))
    plt.title(title + " Data after standardization")
    plt.xlabel('Daily Close Price')
    plt.ylabel('Moving Average 7')
    plt.legend()
    plt.show()


"""
Plot the roc curve
"""
def roc_Curve(actual, predicted):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(actual.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(actual[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(actual.ravel(), predicted.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(
        fpr[1],
        tpr[1],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[1],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


"""
Visualization of results for all epochs
"""
def visualize_results_for_each_epoch(data, title, ylabel):
    # for i, res in enumerate(data):
    #     plt.plot(res, '-x', label=legend[i])
    sns.set_theme()
    plt.plot(data, '-x')
    # plt.plot(data)
    plt.xlabel('No. of epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()