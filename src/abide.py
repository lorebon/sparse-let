import numpy as np
from numba import njit, prange
from tqdm import tqdm
import time


def get_event_table(channels):
    not_instant = [[e for e in events if e[0] != e[1]] for events in channels]
    event_table = []

    # exception with high alpha
    if len([intvs[-1][1] for intvs in not_instant if len(intvs) > 0]) < 1:
        return None

    end = max([intvs[-1][1] for intvs in not_instant if len(intvs) > 0])
    focuses = [0 for e in not_instant]
    mask = [0 for _ in range(len(not_instant))]
    something_happens = [False for _ in range(end+1)]
    for intvls in not_instant:
        for intvl in intvls:
            something_happens[intvl[0]] = True
            something_happens[intvl[1]] = True

    for t in range(end):
        if something_happens[t]:
            for i in range(len(focuses)):
                f = focuses[i]
                if mask[i] == 1 and f < len(not_instant[i]) and not_instant[i][f][1] == t:
                    mask[i] = 0
                    focuses[i] = f+1
                if mask[i] == 0 and f < len(not_instant[i]) and not_instant[i][f][0] == t:
                    mask[i] = 1
        event_table.append(mask.copy())
    return np.asarray(event_table, dtype=np.int32)


@njit("float32[:](int32[:,:],int32[:],int32[:,:])", fastmath = True, parallel = True, cache = True)
def abide_old(event_tables, event_tables_s, query):
    num_examples = event_tables_s.shape[0] - 1
    feature_vector = np.zeros((num_examples), dtype=np.float32)

    for example_index in prange(num_examples):
        start_index = event_tables_s[example_index]
        end_index = event_tables_s[example_index+1]
        query_duration = query.shape[0]
        best_distance = np.sum(np.abs(event_tables[:query_duration,:] - query))
        for j in range(start_index, end_index-query_duration):
            distance = np.sum(np.abs(event_tables[j:j+query_duration,:] - query))
            if not np.isnan(distance) and (distance < best_distance or np.isnan(best_distance)):
                best_distance = distance
        feature_vector[example_index] = best_distance
    return feature_vector


@njit("float32[:](int32[:,:],int32[:],int32[:,:])", fastmath = True, parallel = True, cache = True)
def abide(event_tables, event_tables_s, query):
    num_examples = event_tables_s.shape[0] - 1
    feature_vector = np.zeros((num_examples), dtype=np.float32)
    query_duration = query.shape[0]

    # AB_LB speedup
    c_sigma_q = np.sum(query, axis=0, dtype=np.int32)
    c_sigma_w = np.zeros(query.shape[1], dtype=np.int32)

    for example_index in prange(num_examples):
        start_index = event_tables_s[example_index]
        end_index = event_tables_s[example_index+1]
        best_distance = np.sum(np.abs(event_tables[:query_duration,:] - query))
        for j in range(start_index, end_index-query_duration):
            if j == start_index:
                c_sigma_w = np.sum(event_tables[j:j+query_duration,:], axis=0, dtype=np.int32)
            else:
                c_sigma_w = np.abs(event_tables[j-1, :] - c_sigma_w)
                c_sigma_w = np.abs(event_tables[j+query_duration, :] + c_sigma_w)

            # O(1) pruning
            lb_o1 = np.abs(np.sum(c_sigma_q) - np.sum(c_sigma_w))
            if lb_o1 < best_distance:

                # ab_lb pruning
                lb = np.sum(np.abs(np.expand_dims(c_sigma_q, axis=0) - np.expand_dims(c_sigma_w, axis=0)))
                if lb < best_distance:

                    # Euclidean distance
                    distance = np.sum(np.abs(event_tables[j:j+query_duration,:] - query))
                    if not np.isnan(distance) and (distance < best_distance or np.isnan(best_distance)):
                        best_distance = distance

        feature_vector[example_index] = best_distance
    return feature_vector


@njit("float32[:](int32[:,:],int32[:],int32[:,:])", fastmath = True, parallel = True, cache = True)
def abide_full(event_tables, event_tables_s, query):
    num_examples = event_tables_s.shape[0] - 1
    feature_vector = np.zeros((num_examples), dtype=np.float32)
    query_duration = np.int32(query.shape[0])

    # AB_LB speedup
    c_sigma_q = np.sum(query, axis=0, dtype=np.int32)
    c_sigma_w = np.zeros(query.shape[1], dtype=np.int32)

    # AiDE data
    r_sigma_q = c_sigma_q / query_duration
    test = query_duration * np.ones(c_sigma_q.shape[0])

    for example_index in prange(num_examples):
        start_index = event_tables_s[example_index]
        end_index = event_tables_s[example_index+1]
        best_distance = np.sum(np.abs(event_tables[:query_duration,:] - query))
        for j in range(start_index, end_index-query_duration):
            if j == start_index:
                c_sigma_w = np.sum(event_tables[j:j+query_duration,:], axis=0, dtype=np.int32)
            else:
                c_sigma_w = np.abs(event_tables[j-1, :] - c_sigma_w).astype(np.int32)
                c_sigma_w = np.abs(event_tables[j+query_duration, :] + c_sigma_w).astype(np.int32)

            r_sigma_w = np.divide(c_sigma_w.astype(np.float64), test)

            # O(1) pruning
            lb_o1 = np.abs(np.sum(c_sigma_q) - np.sum(c_sigma_w))
            if lb_o1 < best_distance:

                # ab_lb pruning
                lb = np.sum(np.abs(np.expand_dims(c_sigma_q, axis=0) - np.expand_dims(c_sigma_w, axis=0)))
                if lb < best_distance:

                    # AiDE pruning
                    flag = 0
                    if j == start_index:
                        diff = np.abs(r_sigma_q - r_sigma_w)
                        order = np.argsort(-diff)
                    elif (j - start_index) % 10 == 0:
                        diff = np.abs(r_sigma_q - r_sigma_w)
                        order = np.argsort(-diff)
                    cumsum = 0
                    for k in order:
                        cumsum += np.sum(np.abs(event_tables[j:j+query_duration, k] - query[:, k]))
                        if cumsum >= best_distance:
                            flag = 1
                            break

                    # Euclidean distance
                    if flag == 0:
                        best_distance = cumsum

        feature_vector[example_index] = best_distance
    return feature_vector


def abide_features_test(sequences, queries):
    event_tables = [get_event_table(s) for s in sequences]
    event_tables_s = [0]
    for et in event_tables:
        event_tables_s.append(event_tables_s[-1] + et.shape[0])
    event_tables_s = np.asarray(event_tables_s, dtype=np.int32)
    event_tables = np.concatenate(event_tables, axis=0)
    feature_matrix = np.column_stack([abide(event_tables, event_tables_s, q) for q in queries])
    return feature_matrix

