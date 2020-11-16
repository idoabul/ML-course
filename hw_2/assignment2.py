#################################
# Your name: Ido Abulafya
#################################

import numpy as np
import math
import matplotlib.pyplot as plt
import intervals
import numpy.random


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        MY NOTE: Type in the doc. Should be (m, 2)
        Returns: np.ndarray of shape (m,1) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        xs = np.random.uniform(0, 1, m)
        xs.sort()
        probabilities = numpy.random.random(len(xs))
        A = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        sample = np.ndarray((m, 2))
        for idx, x in enumerate(xs):
            if any(interval[0] <= x <= interval[1] for interval in A):
                sample[idx] = (x, numpy.random.choice([0, 1], size=1, p=[0.2,0.8]))
            else:
                sample[idx] = (x, numpy.random.choice([0, 1], size=1, p=[0.9,0.1]))
        return sample

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        # plt.clf()
        sample = self.sample_from_D(m)
        xs, ys = sample[:, 0], sample[:, 1]
        plt.ylim((-0.1, 1.1))
        plt.plot(xs, ys, "o", markersize=1)
        for x in [0.2, 0.4, 0.6, 0.8]:
            plt.axvline(x, color='red')
        inters, errors = intervals.find_best_interval(xs, ys, k)
        for inter in inters:
            plt.plot(inter, [-0.1] * 2, linewidth=6)
        # plt.savefig("a")

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        ms = range(m_first, m_last + 1, step)
        true_error_avarage = []
        empirical_error_avarage = []
        for m in ms:
            empirical_errors = []
            true_errors = []
            for i in range(T):
                sample = self.sample_from_D(m)
                xs, ys = sample[:, 0], sample[:, 1]
                inters, errors = intervals.find_best_interval(xs, ys, k)
                empirical_errors.append(errors / m)
                true_errors.append(self.calc_true_error(inters))
            empirical_error_avarage.append(sum(empirical_errors) / len(empirical_errors))
            true_error_avarage.append(sum(true_errors) / len(true_errors))
        plt.clf()
        plt.plot(ms, empirical_error_avarage, label='empirical_error', marker='o')
        plt.plot(ms, true_error_avarage, label='true_error', marker='o')
        plt.xlabel("m")
        plt.legend()
        # plt.savefig("c")


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        ks = range(k_first, k_last + 1, step)
        sample = self.sample_from_D(m)
        empirical_errors = []
        true_errors = []
        xs, ys = sample[:, 0], sample[:, 1]
        for k in ks:
            inters, errors = intervals.find_best_interval(xs, ys, k)
            empirical_errors.append(errors / m)
            true_errors.append(self.calc_true_error(inters))
        plt.clf()
        plt.plot(ks, empirical_errors, label='empirical_error', marker='o')
        plt.plot(ks, true_errors, label='true_error', marker='o')
        plt.xlabel("k")
        plt.legend()
        # plt.savefig("d")

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        ks = range(k_first, k_last + 1, step)
        sample = self.sample_from_D(m)
        empirical_errors = []
        true_errors = []
        panelties = []
        empirical_error_plus_panelties = []
        xs, ys = sample[:, 0], sample[:, 1]
        for k in ks:
            inters, errors = intervals.find_best_interval(xs, ys, k)
            empirical_errors.append(errors / m)
            true_errors.append(self.calc_true_error(inters))
            panelties.append(self.panelty(m, k))
            empirical_error_plus_panelties.append(panelties[-1] + empirical_errors[-1])
        plt.clf()
        plt.plot(ks, empirical_errors, label='empirical_error', marker='o')
        plt.plot(ks, true_errors, label='true_error', marker='o')
        plt.plot(ks, panelties, label='empirical_error', marker='o')
        plt.plot(ks, empirical_error_plus_panelties, label='empirical_error + penalty', marker='o')
        plt.xlabel("k")
        plt.legend()
        # plt.savefig("e")

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        ks = range(1, 11)
        training_set_length = int(0.8 * m)
        best_k, min_error, best_hypothesis = None, None, None
        for i in range(T):
            sample = self.sample_from_D(m)
            np.random.shuffle(sample)
            training_data = np.array(sorted(sample[:training_set_length], key=lambda x: x[0]))
            testing_data = np.array(sorted(sample[training_set_length:], key=lambda x: x[0]))
            training_xs, training_ys = training_data[:,0], training_data[:,1]
            testing_xs, testing_ys = testing_data[:,0], testing_data[:,1]
            for k in ks:
                inters, error = intervals.find_best_interval(training_xs, training_ys, k)
                testing_set_error = self.calc_empirical_error(testing_xs, testing_ys, inters)
                if (min_error is None) or (testing_set_error < min_error):
                    min_error = testing_set_error
                    best_k = k
                    best_hypothesis = inters
        print("The best k is: ", best_k)
        print("The best hypotesis is: ", best_hypothesis)
        return k


    #################################
    # Place for additional methods

    def calc_empirical_error(self, xs, ys, inters):
        h = lambda x: any(inter[0] <= x <= inter[1] for inter in inters)
        return sum(h(x) != y for x, y in zip(xs, ys)) / len(xs)

    def calc_true_error(self, inters):
        A = [(0., 0.2), (0.4, 0.6), (0.8, 1.)]
        B = [(0.2, 0.4), (0.6, 0.8)]
        A_length = sum(a[1] - a[0] for a in A)
        B_length = sum(b[1] - b[0] for b in B)
        expectation_depends_interval_A_h_1 = 0.2
        expectation_depends_interval_A_h_0 = 0.8
        expectation_depends_interval_B_h_1 = 0.9
        expectation_depends_interval_B_h_0 = 0.1
        # P(x in A, h(x) == 1) = p(x in A) * p(h(x) == 1 | x in A)
        # p(h(x) == 1 | x in A) = |(h's inters) intersection A|
        h_1_depends_A = sum(
            self.intersection_size(inter, h_inter) for h_inter in inters for inter in A
        ) / A_length
        h_0_depends_A = 1 - h_1_depends_A
        h_1_depends_B = sum(
            self.intersection_size(inter, h_inter) for h_inter in inters for inter in B
        ) / B_length
        h_0_depends_B = 1 - h_1_depends_B
        return A_length * h_1_depends_A * expectation_depends_interval_A_h_1 + \
               A_length * h_0_depends_A * expectation_depends_interval_A_h_0 + \
               B_length * h_1_depends_B * expectation_depends_interval_B_h_1 + \
               B_length * h_0_depends_B * expectation_depends_interval_B_h_0

    def panelty(self, m, k):
        vc_dim_part = 2 * k * math.log((math.e * m) / k)
        return math.sqrt((8 / m) * (math.log(40) + vc_dim_part))

    # For intervals (a, b), (c, d):
    # |[a, b] intersection [c, d]| = max(min(b, d) - max(a, c)) if the intersection
    # is not the empty group. otherwise, 0.
    @classmethod
    def intersection_size(cls, interval_1, interval_2):
        return max(min(interval_1[1], interval_2[1]) - max(interval_1[0], interval_2[0]), 0)

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)


