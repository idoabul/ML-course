from collections import Counter
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
from numpy.linalg import norm
import numpy.random
import functools


class MLData:
    def __init__(self, data, labels):
        self.data = data
        self.label = labels

TRAIN_LENGTH = 10000
TEST_LENGTH = 1000
ml = fetch_openml("mnist_784")
ml_data = [MLData(image, label) 
              for image, label in zip(ml['data'], ml['target'])]
random_idx = numpy.random.RandomState(0).choice(
    len(ml_data), TRAIN_LENGTH + TEST_LENGTH
)
train_ml_data = [ml_data[random_idx[idx]] 
					for idx in range(TRAIN_LENGTH)]
test_ml_data = [ml_data[random_idx[TRAIN_LENGTH + idx]]
                    for idx in range(TEST_LENGTH)]

cache = dict()
def k_nearest_neighbors(train_ml_data, query_image, k):
	if k == 1:
		return min(train_ml_data, key=lambda x: norm(x.data - query_image)).label
	nearest = sorted(train_ml_data, key=lambda x: norm(x.data - query_image))
	key = hash(query_image.tobytes())
	if key not in cache:
		cache[key] = nearest
	k_nearest = cache[key][:k]
	return Counter(map(lambda x: x.label, k_nearest)).most_common()[0][0]

def k_nearest_accuracy(train_ml_data, test_ml_data, k):
    correct = sum(
        test.label == k_nearest_neighbors(train_ml_data, test.data, k) 
        for test in test_ml_data
    )
    print("{} / {}, k={} n={}".format(
        correct, len(test_ml_data), k, len(train_ml_data)
    ))
    return 100 * float(correct) / len(test_ml_data)

def plot_as_function_of_k(ml):
    ks = range(1, 101)
    train_length = 1000
    train_sliced_data = train_ml_data[:train_length]
    y = [k_nearest_accuracy(train_sliced_data, test_ml_data, k) for k in ks]
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.plot(ks, y)
    plt.show()

def plot_as_function_of_n(ml):
    k = 1
    ns = range(100, 5100, 100)
    y = [k_nearest_accuracy(train_ml_data[:n], test_ml_data, k)
            for n in ns]
    plt.xlabel("train_length")
    plt.ylabel("accuracy")
    plt.plot(ns, y)
    plt.show()

if __name__ == "__main__":
    plot_as_function_of_n(ml)
