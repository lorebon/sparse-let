# Lorenzo Bonasera 2023

import multiprocessing
import load_data
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from load_data import *
from train_test_split import cross_validate
from abide import *
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.problem import ElementwiseProblem, Problem, StarmapParallelization, LoopedElementwiseEvaluation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from scipy.stats import mode
import time
from pymoo.core.termination import Termination, TerminateIfAny
from pymoo.termination.max_gen import MaximumGenerationTermination


### Termination
class BaseTarget(Termination):

    def __init__(self, target=None) -> None:
        super().__init__()
        if target is None:
            target = 0.0
        self.target = target

    def _update(self, algorithm):
        F = float("inf")
        if algorithm.opt is not None:
            F = algorithm.opt.get("F")
        if F <= self.target:
            return 1.0
        else:
            return 0.0


class TargetTermination(TerminateIfAny):

    def __init__(self, n_gen=float("inf"), target=None) -> None:
        super().__init__()
        self.max_gen = MaximumGenerationTermination(n_gen)
        self.target = BaseTarget(target)
        self.criteria = [self.max_gen, self.target]


### Genetic algorithm problem
class MyProblem(Problem):

    def __init__(self, sequences, target, sigma, duration, bootstrap, runner=None):
        self.event_tables, self.event_tables_s = preprocessData(sequences)
        self.num_queries = bootstrap.shape[0]
        self.boot_size = bootstrap.shape[1]
        self.target = target
        self.sigma = sigma
        self.duration = duration
        self.abide_time = []
        self.clf_time = []
        self.bootstrap = bootstrap
        super().__init__(n_var=self.boot_size*self.num_queries*2, n_obj=1, n_constr=0, xl=0.0, xu=1.0, elementwise=True, elementwise_runner=runner)


    def _evaluate(self, x, out, *args, **kwargs):
        # Adjust chromosome
        query = decoder(x, self.duration, self.sigma, self.num_queries, self.bootstrap, self.boot_size)
        # Compute fitness value
        out["F"] = fitnessModel(query, self.event_tables, self.event_tables_s, self.target, self.abide_time, self.clf_time)


def decoder(x_sol, duration, sigma, num_queries, bootstrap, boot_size):
    sequence = []
    min_length = np.ceil(duration/2)

    for k in range(num_queries):
        query = np.zeros((duration, sigma), dtype=np.int32)
        min_start = 1

        for (j, ev) in enumerate(bootstrap[k]):
            length = int(np.ceil(x_sol[j*2 + k*boot_size*2] * duration))
            if length > min_length:
                start = int(np.ceil(x_sol[j*2+1 + k*boot_size*2] * (duration - length + 1)))
                query[start:start+length, ev] = 1
                if start > min_start:
                    min_start = start

        if min_start > 1:
            query = query[min_start:, :]

        sequence.append(query)
    return sequence


def fitnessModel(query, event_tables, event_tables_s, target, abide_time, clf_time):
    #for q in query:
    #    if q is None:
    #        print("here")
    #        return 1
    start = time.time()
    feature_matrix = np.column_stack([abide(event_tables, event_tables_s, q) for q in query])
    abide_time.append(time.time() - start)
    #clf = SVC(C=100)
    #clf = DecisionTreeClassifier()
    start = time.time()
    clf = RandomForestClassifier(oob_score=True)
    #X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target, test_size=0.33, stratify=target)
    #clf.fit(X_train, y_train)
    #err = 1 - clf.score(X_test, y_test)
    clf.fit(feature_matrix, target)
    err = 1 - clf.oob_score_
    #err = 1 - clf.score(feature_matrix, target)
    clf_time.append(time.time() - start)
    return err


def preprocessData(sequences):
    event_tables = [get_event_table(s) for s in sequences]
    event_tables_s = [0]
    for et in event_tables:
        event_tables_s.append(event_tables_s[-1] + et.shape[0])
    event_tables_s = np.asarray(event_tables_s, dtype=np.int32)
    event_tables = np.concatenate(event_tables, axis=0)
    return event_tables, event_tables_s


if __name__ == '__main__':
    np.random.seed(2)

    ### Load data
    sequences, ys = load_data.load_auslan2(cast_to_int=True)
    max_query_duration = min([max([intvs[-1][1] for intvs in evts if len(intvs) > 0]) for evts in sequences])
    # print(np.sort([max([intvs[-1][1] for intvs in evts if len(intvs) > 0]) for evts in sequences]))
    sigma = len(sequences[0])
    n_folds = 10

    ### Parameters
    duration = round(max_query_duration/1)
    print("duration:", duration)
    alpha = 0.8
    K = 4
    n_gen = 40
    boot_size = round(alpha*sigma)
    chromosome_length = boot_size * 2 * K
    print("Chromosome length:", chromosome_length)

    ### Cross-validation
    cv_folds = cross_validate(sequences, ys, n_folds)
    cv_scores = []
    cv_queries = []
    for train_seqs, train_ys, test_seqs, test_ys in cv_folds:
        start = time.time()

        ### Bootstrap
        bootstrap = np.stack([np.random.choice(sigma,size=boot_size,replace=False) for i in range(K)])

        ### Initialize BRKGA
        pop = round(chromosome_length * 2)
        pope = 0.15
        popm = 0.20
        algo = BRKGA(n_elites=int(pope * pop), n_offsprings=pop, n_mutants=int(popm * pop), bias=0.70)
        termination = TargetTermination(n_gen)

        ### Parallelize
        n_proccess = 6
        pool = multiprocessing.Pool(n_proccess)
        runner = StarmapParallelization(pool.starmap)
        problem = MyProblem(train_seqs, train_ys, sigma, duration, bootstrap, runner)
        res = minimize(problem, algo, termination, seed=1, verbose=True)
        queries = decoder(res.X, duration, sigma, K, bootstrap, boot_size)

        feature_matrix = abide_features_test(train_seqs, queries)
        #clf = SVC(C=100)
        clf = RandomForestClassifier()
        #clf = DecisionTreeClassifier()
        clf.fit(feature_matrix, train_ys)

        feature_matrix = abide_features_test(test_seqs, queries)
        acc = clf.score(feature_matrix, test_ys)
        print("Accuracy: {}%".format(100 * round(acc, 4)))
        cv_scores.append(acc)
        cv_queries.append(queries)

        end = time.time()
        print("Elapsed time:", end-start)
        print("Abide time: {}%".format(round(sum(problem.abide_time)*100/(end-start)), 2))
        print("Clf time: {}%".format(round(sum(problem.clf_time)*100/(end-start)), 2))

        '''
        ######## Plot
        # Define the height of each row in the plot
        row_height = 0.5

        for (idx, ibsm_event_matrix) in enumerate(queries):
            ibsm_event_matrix = ibsm_event_matrix[1:, :]
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
            '''
        #############

    print("final score: {}%".format(100 * round(sum(cv_scores)/n_folds, 4)))





