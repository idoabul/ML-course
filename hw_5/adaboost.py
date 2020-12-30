#################################
# Your name: Ido Abulafya
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)

def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    assert len(X_train) == len(y_train)
    hypotheses = []
    alpha_vals = []
    n = len(X_train)
    D = np.array([1. / n for _ in range(n)])
    for t in range(T):
        h, error = WL(D, X_train, y_train)
        weight = 0.5 * np.log((1 - error) / error)
        exponents = np.exp(
            np.array([-y*weight*h(x) for x, y in zip(X_train, y_train)])
        )
        D = np.multiply(D, exponents) / np.dot(D, exponents)
        hypotheses.append(h)
        alpha_vals.append(weight)
    return hypotheses, alpha_vals


##############################################
# You can add more methods here, if needed.

class Stump:
    def __init__(self, pred, index, theta):
        self.pred = pred
        self.index = index
        self.theta = theta
    
    def __call__(self, x):
        return self.pred if (x[self.index] <= self.theta) else -self.pred

    def __repr__(self):
        if self.pred == 1:
            return "x <= {}".format(self.theta)
        return "{} <= x".format(self.theta)

def WL(D, X_train, y_train):
    errors = []
    related_classifers = []
    best_h, min_error = None, np.Infinity
    for j in range(len(X_train[0])):
        h_j, error = get_best_hypothesis_per_j(D, X_train, y_train, j) 
        if error < min_error:
            best_h, min_error = h_j, error
    return best_h, min_error

def get_probs_of_zeros_and_ones(D, xs, ys):
    x_to_probs = dict()
    for d, x, y in zip(D, xs, ys):
        x_to_probs.setdefault(x, [0, 0])[int((y + 1) // 2)] += d
    return x_to_probs

def get_best_hypothesis_per_j(D, X_train, y_train, j):
    xs = [x[j] for x in X_train]
    thetas = set(xs)
    xj_to_probs = get_probs_of_zeros_and_ones(D, xs, y_train)
    best_h, min_error = None, np.Infinity
    for theta in thetas:
        for pred in (-1, 1):
            for neighbor in (-0.5, 0.5):
                h = Stump(pred, j, theta-neighbor)
                error = dist_error(h, xj_to_probs)
                if error < min_error:
                    best_h, min_error = h, error
    return best_h, min_error
        
def dist_error(h, xj_to_probs):
    sign = lambda x: h.pred if (x <= h.theta) else -h.pred
    return sum(errors[sign(x) == -1] for x, errors in xj_to_probs.items())

def empirical_error(h, xs, ys):
    return sum((h(x) != y) for x, y in zip(xs, ys)) / len(xs)

def get_weighted_classifiers(classifiers, alpha_vals):
    return lambda x: sum(h(x) * alpha for h, alpha in zip(classifiers, alpha_vals))

def get_classification_from_weighted_classifiers(classifiers, alpha_vals):
    return lambda x: np.sign(get_weighted_classifiers(classifiers, alpha_vals)(x))

def calc_loss(h, xs, ys):
    n = len(xs)
    return (1 / n) * sum(np.exp(
        np.array([-y*h(x) for x, y in zip(xs, ys)])
    ))

##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    # section a
    # T = 80
    # hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    # train_errors = []
    # test_errors = []
    # for t in range(1, T + 1):
        # h = get_classification_from_weighted_classifiers(hypotheses[:t], alpha_vals[:t])
        # train_errors.append(empirical_error(h, X_train, y_train))
        # test_errors.append(empirical_error(h, X_test, y_test))

    # ts = np.arange(1,T+1)
    # plt.clf()
    # plt.title('accuracy per t')
    # plt.plot(ts, train_errors ,marker='o', label = 'error of train set')
    # plt.plot(ts, test_errors ,marker='o', label = 'error of test set')
    # plt.xlabel('t')
    # plt.legend()
    # plt.show()

    # section b
    T = 10
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    # print([vocab[h.index] for h in hypotheses])

    # section c
    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    train_errors = []
    test_errors = []
    for t in range(1, T + 1):
        h = get_weighted_classifiers(hypotheses[:t], alpha_vals[:t])
        train_errors.append(calc_loss(h, X_train, y_train))
        test_errors.append(calc_loss(h, X_test, y_test))
    ts = np.arange(1,T+1)
    plt.clf()
    plt.title('accuracy per t')
    plt.plot(ts, train_errors ,marker='o', label = 'loss of train set')
    plt.plot(ts, test_errors ,marker='o', label = 'loss of test set')
    plt.xlabel('t')
    plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()

