# Lorenzo Bonasera 2023

import load_data
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from load_data import *
from train_test_split import cross_validate
from abide import *
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from scipy.stats import mode
from pymoo.core.duplicate import ElementwiseDuplicateElimination
import time


### Genetic algorithm problem
class MyProblem(ElementwiseProblem):

    def __init__(self, sequences, target, sigma, duration, alpha, num_queries):
        self.sequences = sequences
        self.num_queries = num_queries
        self.target = target
        self.sigma = sigma
        self.duration = duration
        self.alpha = alpha
        self.abide_time = []
        self.clf_time = []
        self.baseline = np.sum(target == mode(target, keepdims=True))
        super().__init__(n_var=self.sigma*self.num_queries*2, n_obj=1, n_constr=0, xl=0.0, xu=1.0)


    def _evaluate(self, x, out, *args, **kwargs):
        # Adjust chromosome
        query = decoder(x, self.duration, self.sigma, self.num_queries)
        # Compute fitness value
        out["F"] = fitnessModel(query, self.num_queries, self.sequences, self.target, self.baseline, self.alpha, self.abide_time, self.clf_time)


def findStart(start, duration):
    step = 1/duration
    for i in range(duration):
        if i*step <= start < (i+1)*step:
            # check if i or i+1
            return i


def decoder(x_sol, duration, sigma, num_queries):
    sequence = []
    for k in range(num_queries):
        query = []
        for i in range(sigma):
            start = findStart(x_sol[i*2 + k*sigma*2], duration)
            length = round(x_sol[i*2+1 + k*sigma*2] * duration)
            if length < 1:
                query.append([])
            else:
                query.append([[start, start + length, 1]])
        event_table = get_event_table(query)
        sequence.append(event_table)
    return sequence


def fitnessModel(query, num_queries, sequences, target, baseline, alpha, abide_time, clf_time):
    for q in query:
        if q is None:
            return 1
    start = time.time()
    feature_matrix = abide_features_test(sequences, query)
    abide_time.append(time.time() - start)
    #clf = SVC()
    start = time.time()
    clf = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target, test_size=0.33, random_state=1, stratify=target)
    clf.fit(X_train, y_train)
    err = 1 - clf.score(X_test, y_test)
    clf_time.append(time.time() - start)

    reg = 0
    for q in query:
        reg += np.count_nonzero(np.sum(q, axis=0))

    baseline = np.sum(y_train == mode(y_train, keepdims=True))
    return err/baseline + alpha * reg


def newfitnessModel(query, num_queries, sequences, target, baseline, alpha):
    for q in query:
        if q is None:
            return 1
    feature_matrix = abide_features_test(sequences, query)
    reg = 0
    for q in query:
        reg += np.count_nonzero(np.sum(q, axis=0))
    return -sum(np.var(feature_matrix, axis=0)) + alpha * reg


if __name__ == '__main__':
    np.random.seed(1)

    ### Load data
    sequences, ys = load_data.load_pioneer(cast_to_int=True)
    max_query_duration = min([max([intvs[-1][1] for intvs in evts if len(intvs) > 0]) for evts in sequences])
    #print(np.sort([max([intvs[-1][1] for intvs in evts if len(intvs) > 0]) for evts in sequences]))

    # Parameters
    sigma = len(sequences[0])
    duration = round(max_query_duration/4)
    print("duration:", duration)
    par = 2
    n_folds = 10
    alpha = 1e-3
    K = 5
    chromosome_length = sigma * 2 * K
    print("Chromosome length:", chromosome_length)

    # Cross-validation
    cv_folds = cross_validate(sequences, ys, n_folds)
    cv_scores = []
    cv_queries = []
    for train_seqs, train_ys, test_seqs, test_ys in cv_folds:
        start = time.time()

        ### Initialize BRKGA
        pop = 80
        pope = 0.25
        popm = 0.20
        algo = BRKGA(n_elites=int(pope * pop), n_offsprings=pop, n_mutants=int(popm * pop), bias=0.7)
        termination = get_termination("n_gen", 20)

        problem = MyProblem(train_seqs, train_ys, sigma, duration, alpha, K)
        res = minimize(problem, algo, termination, seed=1, verbose=True)
        queries = decoder(res.X, duration, sigma, K)

        feature_matrix = abide_features_test(train_seqs, queries)
        #clf = SVC()
        clf = DecisionTreeClassifier()
        clf.fit(feature_matrix, train_ys)

        feature_matrix = abide_features_test(test_seqs, queries)
        acc = clf.score(feature_matrix, test_ys)
        print("Accuracy:", acc)
        cv_scores.append(acc)
        cv_queries.append(queries)

        end = time.time()
        print("Elapsed time:", end-start)
        print("Abide time: {}%".format(round(sum(problem.abide_time)*100/(end-start)), 2))
        print("Clf time: {}%".format(round(sum(problem.clf_time)*100/(end-start)), 2))

        ######## Plot
        # Define the height of each row in the plot
        row_height = 0.5

        for (idx, ibsm_event_matrix) in enumerate(queries):
            fig, ax = plt.subplots()
            num_event_labels = ibsm_event_matrix.shape[1]
            event_colors = plt.cm.tab20(np.linspace(0, 1, num_event_labels))

            # Loop over each row of the IBSM event matrix
            for i in range(ibsm_event_matrix.shape[0]):
                # Get the indices of the non-zero elements in the row
                nonzero_indices = np.where(ibsm_event_matrix[i, :] != 0)[0]

                # Loop over each non-zero element in the row
                for j in nonzero_indices:
                    # Compute the x and y coordinates of the event
                    x = [i, i + 1]
                    y = [j * row_height, j * row_height]

                    # Plot the event as a horizontal line
                    ax.plot(x, y, color=event_colors[j], linewidth=2)

            # Set the y-ticks and labels
            ax.set_yticks(np.arange(ibsm_event_matrix.shape[1]) * row_height)
            ax.set_yticklabels(['e_{}'.format(i) for i in range(ibsm_event_matrix.shape[1])])

            # Set the x-ticks and labels
            ax.set_xticks(np.arange(ibsm_event_matrix.shape[0]))
            ax.set_xticklabels(['t_{}'.format(i) for i in range(ibsm_event_matrix.shape[0])])

            # Set the axis labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Event')
            ax.set_title('Event-Interval Pattern {}'.format(idx))

            plt.tight_layout()
            plt.show()
        #############

    print("final score:", sum(cv_scores)/n_folds)

    # Define the height of each row in the plot
    row_height = 0.5




