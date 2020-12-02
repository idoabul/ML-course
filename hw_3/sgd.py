#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import tmp
import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

L = 10

def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000+test_idx, :].astype(float)
    test_labels = labels[8000+test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    @return: w_t
    """
    wt = np.zeros(data[0].size)
    for t in range(1, T + 1):
        random_sample = np.random.randint(0, len(data))
        x, y = data[random_sample], labels[random_sample]
        gradient = wt - (C * y * x if (y * np.dot(x, wt) < 1) else 0)
        wt = sgd_step(wt, eta_t=eta_0 / t, gradient=gradient)
    return wt

def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    @return: w_t
    """
    n = data[0].size
    wt = np.zeros((L, n))
    for t in range(1, T + 1):
        random_sample = np.random.randint(0, len(data))
        x, y = data[random_sample], int(labels[random_sample])
        dot_product = np.dot(wt, x)
        exp_dots = np.exp(dot_product - np.max(dot_product))
        gradient = np.divide([exp_dot * x for exp_dot in exp_dots], np.sum(exp_dots))
        gradient[y] -= x
        wt = sgd_step(wt, eta_t=eta_0, gradient=gradient)
    return wt

#################################

# Place for additional code

#################################

def sgd_step(w_t, eta_t, gradient):
    return w_t - np.multiply(eta_t, gradient)

def average_SGD_accuracy(
    SGD_algorithm, 
    accuracy_calculator, 
    train_data, 
    train_labels, 
    validation_data, 
    validation_labels, 
    runs, 
    **kwargs
):
    return np.average([
        accuracy_calculator(
            SGD_algorithm(train_data, train_labels, **kwargs), 
            validation_data, 
            validation_labels
        ) for run in range(runs)
    ])

def eta_to_average_accuracy(
    train_data, 
    train_labels, 
    validation_data, 
    validation_labels, 
    SGD_algorithm, 
    accuracy_calculator, 
    **kwargs
):
    average_cv_accuracy = []
    eta_0s = [np.float_power(10, k) for k in np.arange(-5.0, 3, 0.25)]
    eta_0s += [np.float_power(10, k) for k in np.arange(-1, 1, 0.1)]
    eta_0s.sort()
    for eta_0 in eta_0s:
        average_cv_accuracy.append(
            average_SGD_accuracy(
                SGD_algorithm,
                accuracy_calculator,
                train_data,
                train_labels,
                validation_data,
                validation_labels,
                eta_0=eta_0,
                **kwargs
            )
        )
    return eta_0s, average_cv_accuracy

# hinge

def hinge_calc_accuracy(wt, data, labels):
    return np.average([(wt.dot(x) * y) >= 0 for x, y in zip(data, labels)])

def part_1_b(train_data, train_labels, validation_data, validation_labels, **kwargs):
    average_cv_accuracy = []
    Cs = [np.float_power(10, k) for k in np.arange(-5.0, 3.25, 0.2)]
    for C in Cs:
        average_cv_accuracy.append(average_SGD_accuracy(
            SGD_hinge,
            hinge_calc_accuracy,
            train_data,
            train_labels,
            validation_data,
            validation_labels,
            C=C,
            **kwargs
        ))
    return Cs, average_cv_accuracy

def q1():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    # a
    etas, etas_cv_accuracy = eta_to_average_accuracy(
        train_data, 
        train_labels, 
        validation_data, 
        validation_labels, 
        SGD_hinge, 
        hinge_calc_accuracy,
        C=1, T=1000, runs=10
    )
    best_eta = etas[np.argmax(etas_cv_accuracy)]

    # b
    Cs, b_cv_accuracy = part_1_b(
        train_data, train_labels, validation_data, validation_labels,
        eta_0=best_eta, T=1000, runs=10
    )
    best_C = Cs[np.argmax(b_cv_accuracy)]

    # c
    best_wt = SGD_hinge(train_data, train_labels, eta_0=best_eta, C=best_C, T=20000)

    # d
    test_accuracy = hinge_calc_accuracy(best_wt, test_data, test_labels)

	# a
    plt.clf()
    plt.plot(etas, etas_cv_accuracy, label='etas cv error', marker='o')
    plt.xlabel('eta')
    plt.ylabel('average accuracy')
    plt.xscale("log")
    # plt.savefig("a")

	# b
    plt.clf()
    plt.plot(Cs, b_cv_accuracy, label='Cs cv error', marker='o')
    plt.xscale("log")
    plt.xlabel('C')
    plt.ylabel('average accuracy')
    # plt.savefig("b")

	#c
    plt.clf()
    plt.imshow(best_wt.reshape(28, 28))
    # plt.savefig("c")

# CE
def ce_calc_accuracy(wt, data, labels):
    return np.average([(np.argmax(wt.dot(x)) == int(y)) for x, y in zip(data, labels)])

def q2():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    # a
    etas, etas_cv_accuracy = eta_to_average_accuracy(
        train_data, train_labels, validation_data, validation_labels, SGD_ce, ce_calc_accuracy,
        T=1000, runs=10
    )
    best_eta = etas[np.argmax(etas_cv_accuracy)]

    # b
    best_wt = SGD_ce(train_data, train_labels, eta_0=best_eta, T=20000)

    # c
    test_accuracy = ce_calc_accuracy(best_wt, test_data, test_labels)

    plt.clf()
    plt.plot(etas, etas_cv_accuracy, label='etas cv error', marker='o')
    plt.xscale("log")
    plt.legend()
    # plt.savefig("part2_a")
    plt.clf()
    fig = plt.figure()
    for i in range(L):
        fig.add_subplot(2, 5, i + 1)
        plt.imshow(best_wt[i].reshape(28, 28))
    # plt.savefig("part_2_b")

if __name__ == '__main__':
    q1()
    q2()
