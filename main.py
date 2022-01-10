import KMeans
import RBF
import DataProcessing
import numpy as np
from sklearn import preprocessing
import Visualizations
from sklearn.metrics import classification_report
import time

"""
Saving results of GridSearch into a txt file
"""
def save_scores(filename, score, method, k, l, loss):
    file = open(filename, 'a')
    if method == 'RLS':
        L = ['Method = ' + method, ', k = ' + str(k), ' lambda = ' + str(l), '\n']
        file.writelines(L)
        file.write('Score = ' + str(score) + '\n')
        file.write('Loss = ' + str(loss) + '\n')
    else:
        L = ['Method ' + method, ', k = ' + str(k), '\n']
        file.writelines(L)
        file.write('Score = ' + str(score) + '\n')
        file.write('Loss = ' + str(loss) + '\n')


"""
Using k folds algorithm in RBF
"""
def evaluate_algorithm(fold_data, fold_classes, n_folds, k, method, l):
    # fold_data, fold_classes = DataProcessing.cross_validation_split(fold_data, fold_classes,  n_folds)
    scores = list()
    losses = list()
    for i, fold in enumerate(fold_data):

        X_train = list(fold_data)
        X_train.pop(i)

        y_train = list(fold_classes)
        y_train.remove(fold_classes[i])

        # merge data into 1 list
        X_train = sum(X_train, [])
        y_train = sum(y_train, [])

        X_test = list()
        y_test = list(fold_classes[i])
        for row in fold:
            X_test.append(row)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        rbf = RBF.RBF(0.1)
        rbf.fit(X_train, y_train, k, l, method)
        y_pred = rbf.predict(X_test, method)
        accuracy = rbf.accuracy_metric(y_test, y_pred)
        loss = rbf.MSE(y_test, y_test.size)
        scores.append(accuracy)
        losses.append(loss)

    # average score for k folds
    return sum(scores)/n_folds, sum(losses)/n_folds


"""
Grid search
"""
def find_best_parameters(X_train, y_train):
    # cross validation split to 5 folds
    n_folds = 5
    fold_data, fold_classes = DataProcessing.cross_validation_split(X_train, y_train, n_folds)

    method = ['RLS', 'pseudo random']
    # find the best k for k means
    start = time.time()
    num_k = [2, 4, 8, 16, 32, 64, 128, 256]
    # num_k = [2]
    max_acc_ps = -1
    best_k_ps = 0

    max_acc_rls = -1
    best_k_rls = 0
    best_l = 0
    for k in num_k:
        for name in method:
            if name == 'pseudo random':
                acc, loss = evaluate_algorithm(fold_data, fold_classes, n_folds, k, name, 0)
                save_scores(name + ' Results', acc, name, k, 0, loss)
                if acc > max_acc_ps:
                    max_acc_ps = acc
                    best_k_ps = k
            else:
                l = [0.1, 0.01, 0.001, 0.0001]
                # l = [0.1]
                for parameter in l:
                    acc, loss = evaluate_algorithm(fold_data, fold_classes, n_folds, k, name, parameter)
                    save_scores(name + ' Results', acc, name, k, parameter, loss)
                    if acc > max_acc_rls:
                        max_acc_rls = acc
                        best_k_rls = k
                        best_l = parameter

    end = time.time()

    print("------- Pseudo random ------")
    print(f"Best k after cross validation with 5 folds is: k = {(best_k_ps):>d} with accuracy = {(max_acc_ps):>0.1f}%")

    print("------- RLS ------")
    print(f"Best k after cross validation with 5 folds is: k = {(best_k_rls):>d} with accuracy = {(max_acc_rls):>0.1f}%")
    print(f"λ = {(best_l):>}")

    print("Time execution is: {} sec".format(end - start))

    return best_k_ps, best_k_rls, best_l

"""
Training data from sp 500 for 2000-2021
Testing data from sp 500 for 2021 rest
see results for best parameter k
"""
def test_pseudo(X_train, y_train, X_test, y_test, k, l):
    print("--------------------- Pseudo Inverse ---------------------")
    start = time.time()
    rbf = RBF.RBF(0.1)
    rbf.fit(X_train, y_train, k, 0, 'pseudo_inverse')
    y_pred = rbf.predict(X_test, 'pseudo_inverse')
    end = time.time()
    target_names = ['SELL', 'BUY']
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"Accuracy: {(rbf.accuracy_metric(y_test, y_pred)):>0.1f}%")
    print(f"Time execution for pseudo inverse method is: {(end - start):>0.2f} sec")


"""
Testing rls algorithm instead pseudo inverse
"""
def test_rls(X_train, y_train, X_test, y_test, k, l):
     print("--------------------- RLS ---------------------")
     start = time.time()

     rbf = RBF.RBF(0.1)
     rbf.fit(X_train, y_train, k, 0.01, 'RLS')
     y_pred = rbf.predict(X_test, 'RLS')

     end = time.time()

     target_names = ['SELL', 'BUY']
     print(classification_report(y_test, y_pred, target_names=target_names))
     print(f"Accuracy: {(rbf.accuracy_metric(y_test, y_pred)):>0.1f}%")
     print(f"Time execution for RLS method is: {(end - start):>0.2f} sec")


def run():
    processing = DataProcessing
    # Get train data 2000-2020 and testing data 2021-now for SP500
    X_train, y_train = processing.processing_data('SPY', start='2000-01-01', end='2020-12-31', interval='1d')
    X_test, y_test = processing.processing_data('SPY test', start='2021-01-01', end='2021-12-01', interval='1d')

    # best_k_ps, best_k_rls, best_l = find_best_parameters(X_train, y_train)

    # Real testing
    test_pseudo(X_train, y_train, X_test, y_test, 2, 0)
    test_rls(X_train, y_train, X_test, y_test, 16, 0.001)


if __name__ == '__main__':

    run()
