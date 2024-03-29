#################################
# Your name: Ido Abulafya
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    C = 1000
    support_vectors = []
    models = ['linear', 'poly', 'rbf']
    for model in models:
        clf = svm.SVC(C=C, kernel=model, degree=2)
        clf.fit(X_train, y_train)
        create_plot(X_train, y_train, clf) 
        plt.title("{} model classifer".format(model)) 
        # plt.show()
        support_vectors.append(clf.n_support_)
    return np.array(support_vectors)

def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    Cs = [10 ** i for i in range(-5, 6)]
    test_accuracies = []
    train_accuracies = []
    for C in Cs:
        clf = svm.SVC(C=C, kernel='linear')
        clf.fit(X_train, y_train)
        test_accuracies.append(clf.score(X_val, y_val))
        train_accuracies.append(clf.score(X_train, y_train))

    # plot accurecy as function of C
    plt.clf()
    plt.title('linear accuracy per C')
    plt.plot(Cs, train_accuracies, label='accuracy on train samples', marker='o') 
    plt.plot(Cs, test_accuracies, label='accuracy on validation samples', marker='o')
    plt.xscale("log")
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.legend()
    # plt.show()

    return np.array(test_accuracies)


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C = 10
    gammas = [10 ** i for i in range(-5, 6)]
    test_accuracies = []
    train_accuracies = []
    for gamma in gammas:
        clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        clf.fit(X_train, y_train)
        test_accuracies.append(clf.score(X_val, y_val))
        train_accuracies.append(clf.score(X_train, y_train))

    # plot accuracy as function of gamma
    plt.clf()
    plt.title('rbf accuracy per gamma')
    plt.plot(gammas, train_accuracies, label='accuracy on train samples', marker='o') 
    plt.plot(gammas, test_accuracies, label='accuracy on validation samples', marker='o')
    plt.xscale("log")
    plt.xlabel('gamma')
    plt.ylabel('accuracy')
    plt.legend()
    # plt.show()

    return np.array(test_accuracies)

# X_train, y_train, X_val, y_val = get_points()
# train_three_kernels(X_train, y_train, X_val, y_val)
# linear_accuracy_per_C(X_train, y_train, X_val, y_val)
# rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)
