#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import utils

import time
# import random
import numpy as np
import argparse
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import fileinput

import ot
import gromov
import S_GWL_Toolkit as GwGt
from COT import cot_numpy


def load_change_name1(path_pickle):
    with open(path_pickle, 'rb') as handle:
        dict_change_name = pickle.load(handle)
    for key in dict_change_name:
        for algo in dict_change_name[key]["result"]:
            if algo == "GOT_t_init":
                dict_change_name[key]["result"]["OTT"] = dict_change_name[key]["result"]["GOT_t_init"]
                del dict_change_name[key]["result"]["GOT_t_init"]
            # elif "GOT" in algo:
            #     dict_change_name[key]["result"][algo.replace("GOT", "OTT")] = dict_change_name[key]["result"][algo]
            #     del dict_change_name[key]["result"][algo]
    return dict_change_name

def pred_with_T(T, Sy, type_pred="max"):
    if type_pred == "max":
        label_influence = np.zeros((len(np.unique(Sy)), T.shape[1]))
        for label in np.unique(Sy):
            label_influence[label] = T[Sy == label, :].sum(axis=0)
        return np.argmax(label_influence, axis=0)
    elif type_pred == "mean":
        label_influence = np.zeros((len(np.unique(Sy)), T.shape[1]))
        for label in np.unique(Sy):
            label_influence[label] = T[Sy == label, :].sum(axis=0)
        return np.argmax(label_influence, axis=0)


def adaptData(algo, Sx, Sy, Sy_available, Tx, Ty, Ty_available, T_pos, param=None, da_pos=1):
    """
    Main function of the code that launch a method.
    :param algo: Name of the method to use.
    :param Sx: Source features.
    :param Sy: Source labels.
    :param Tx: Target features.
    :param Ty: Target labels.
    :param param: List of parameters needed for each method.
    :return: The adapted data source and target. It also return the labels unchanged.
    """
    pred_user = None
    T = None
    ot_dist = 10000000
    if algo == "random":
        target_label = np.random.choice(np.unique(Sy[da_pos]), size=len(Ty[da_pos]), replace=True)
    elif algo == "smart_random":
        target_label = np.random.choice(np.unique(Sy[da_pos]), size=len(Ty[da_pos]), replace=True)
        target_label[Ty_available[da_pos] != -1] = Ty_available[da_pos][Ty_available[da_pos] != -1]
    elif algo == "target_classifier":
        clf = make_pipeline(StandardScaler(), SVC(C=1, gamma='auto'))
        x_train = Tx[0].mean(axis=0)[Ty_available[da_pos] != -1, :]
        y_train = Ty_available[da_pos][Ty_available[da_pos] != -1]
        if len(Ty_available[da_pos][Ty_available[da_pos] != -1]) == 1:
            target_label = np.ones_like(Ty[da_pos]) * (Ty_available[da_pos][Ty_available[da_pos] != -1])[0]
        elif len(Ty_available[da_pos][Ty_available[da_pos] != -1]) == 0:
            target_label = np.random.choice(np.unique(Sy[da_pos]), size=len(Ty[da_pos]), replace=True)
            target_label[Ty_available[da_pos] != -1] = Ty_available[da_pos][Ty_available[da_pos] != -1]
        else:
            clf.fit(x_train, y_train)
            target_label = clf.predict(Tx[0].mean(axis=0))
    elif algo == "gromov":

        Sx_, Tx_ = Sx[0].mean(axis=0), Tx[0].mean(axis=0)

        if param["p"] == "uniform":
            p = ot.unif(len(Sx_))
        if param["q"] == "uniform":
            q = ot.unif(len(Tx_))

        if param["loss_fun"] == "L2":
            loss_fun = "square_loss"
        T, log = gromov.entropic_gromov_wasserstein(C1=Sx_, C2=Tx_,
                                                    p=p, q=q,
                                                    loss_fun=loss_fun,
                                                    epsilon=param["epsilon"],
                                                    eta=param["eta"],
                                                    labels_s=Sy_available[da_pos],
                                                    labels_t=Ty_available[da_pos],
                                                    log=True,
                                                    max_iter=1000)

        target_label = pred_with_T(T, Sy[da_pos])
        ot_dist = log["gw_dist"]
    elif algo == "COT":

        Sx_, Tx_ = Sx[0].mean(axis=1), Tx[0].mean(axis=1)

        Ts, Tv, ot_dist = cot_numpy(X1=Sx_, X2=Tx_, w1=None, w2=None, v1=None, v2=None,
                                    labels_s=Sy_available,
                                    labels_t=Ty_available,
                                    niter=10, algo='supervised', reg=param["epsilon"], algo2='supervised',
                                    reg2=param["epsilon"],
                                    eta=param["eta"], verbose=False, log=False, random_init=False, C_lin=None)
        T = Tv
        target_label = pred_with_T(T, Sy[da_pos])

    elif algo == "OTT" or algo == "GOT" or algo == "GOT_t_full" or algo == "GOT_no_t":
        if param["p"] == "uniform":
            p = [ot.unif(len(i)) for i in Sy]
        if param["q"] == "uniform":
            q = [ot.unif(len(i)) for i in Ty]

        if param["loss_fun"] == "L2":
            def f(C1, C2):
                return (C1 - C2) ** 2

            loss_fun = [f for _ in range(len(Sx))]

        def make_f(sx, i):
            def f(*x):
                return sx[x]

            return f

        Sx_ = []
        for sx in Sx:
            Sx_.append(make_f(sx, len(sx.shape)))
        Tx_ = []
        for tx in Tx:
            Tx_.append(make_f(tx, len(tx.shape)))

        if param["T"] == "uniform":
            T_ = []
            for i in range(len(Sy)):
                T_.append(p[i][:, np.newaxis] * q[i][np.newaxis, :])

        if algo == "OTT":
            sample_t_only_init = True
        else:
            sample_t_only_init = False

        if algo == "GOT_t_full":
            sample_t_init_and_iteration = True
        else:
            sample_t_init_and_iteration = False

        if algo == "GOT_no_t":
            nb_samples_t = None
        else:
            nb_samples_t = param["nb_samples_t"]

        T, log = gromov.OTT(p=p, q=q,
                            loss_fun=loss_fun,
                            X1=Sx_, X2=Tx_,
                            M=param["M"],
                            T=T_,
                            T_pos=T_pos,
                            alpha=param["alpha"],
                            nb_iter=param["nb_iter"],
                            nb_samples=param["nb_samples"],
                            nb_samples_t=nb_samples_t,
                            epsilon=param["epsilon"],
                            L2=param["L2"],
                            labels_s=Sy_available,
                            labels_t=Ty_available,
                            eta=param["eta"],
                            KL=param["KL"],
                            threshold=1e-15,
                            sliced=False,
                            log=True,
                            sparse_T=False,
                            verbose=False,
                            sample_t_only_init=sample_t_only_init,
                            sample_t_init_and_iteration=sample_t_init_and_iteration)
        target_label = pred_with_T(T[da_pos], Sy[da_pos])

        pred_user = np.sum(T[0][np.arange(len(T[0])), np.arange(len(T[0]))])
        ot_dist = log["gw_dist_estimated"]
    elif algo == "graph":
        pass  # Doesn't work well with our application
    elif algo == "ScalableGW":
        Sx_, Tx_ = Sx[0].mean(axis=0), Tx[0].mean(axis=0)

        if param["loss_fun"] == "L2":
            loss_fun = "square_loss"
        num_iters = 4000
        ot_dict = {'loss_type': loss_fun,  # the key hyperparameters of GW distance
                   'ot_method': 'proximal',
                   'beta': param["epsilon"],
                   'outer_iteration': num_iters,
                   # outer, inner iteration, error bound of optimal transport
                   'iter_bound': 1e-30,
                   'inner_iteration': 2,
                   'sk_bound': 1e-30,
                   'node_prior': 1e3,
                   'max_iter': 4,  # iteration and error bound for calcuating barycenter
                   'cost_bound': 1e-26,
                   'update_p': True,  # optional updates of source distribution
                   'lr': 0,
                   'alpha': 0}

        idx2node_s = {}
        idx2node_t = {}
        for i in range(len(Sx_)):
            idx2node_s[i] = i
        for i in range(len(Tx_)):
            idx2node_t[i] = i
        try:
            pairs_idx, _, _ = GwGt.recursive_direct_graph_matching(
                cost_s=Sx_, cost_t=Tx_,
                p_s=ot.unif(len(Sx_))[:, np.newaxis],
                p_t=ot.unif(len(Tx_))[:, np.newaxis],
                idx2node_s=idx2node_s, idx2node_t=idx2node_t, ot_hyperpara=ot_dict,
                weights=None, predefine_barycenter=False, cluster_num=2,
                partition_level=3, max_node_num=0)
            pairs_idx = np.array(pairs_idx)
            T = np.zeros((Sx_.shape[0], Tx_.shape[0]))
            T[pairs_idx[:, 0], pairs_idx[:, 1]] = 1 / pairs_idx.shape[0]

            target_label = pred_with_T(T, Sy[da_pos])

            def f(C1, C2):
                return (C1 - C2) ** 2

            ot_dist = np.sum(T * gromov.compute_L(C1=Sx_, C2=Tx_, loss_fun=f, T=T))

        except:
            print("Error")
            target_label = np.array([0] * len(Ty[da_pos]))

    else:
        print(algo)
        assert False  # Not the right algo

    return target_label, T, pred_user, ot_dist


def getAccuracy(prediction, testLabels):
    """
    :param prediction:
    :param testLabels:
    :return: The accuracy of the tested data.
    """
    return 100 * float(sum(prediction == testLabels)) / len(testLabels)


def load_features(path_S, path_T):
    with open(path_S, 'rb') as handle:
        dict_s = pickle.load(handle)
    Sx, Sy, T_pos_S = dict_s["C"], dict_s["True_label"], dict_s["T_pos"]
    with open(path_T, 'rb') as handle:
        dict_t = pickle.load(handle)
    Tx, Ty, T_pos_T = dict_t["C"], dict_t["True_label"], dict_t["T_pos"]
    dict_s["C"], dict_t["C"] = None, None
    assert T_pos_T == T_pos_S
    return Sx, Sy, Tx, Ty, T_pos_T, dict_s, dict_t


def run_adaptation(param, da_pos=1, rdm_seed=12345, param_to_keep=None):
    """
    Main function of the code that is launch at the beginning of the code.
    :return: Print result and save files in pickle format if needed.
    """
    np.random.seed(rdm_seed)
    total_dict = {}
    save_path_S, save_path_T = param["path_S"], param["path_T"]
    for d in range(len(param["dataset"])):
        np.random.seed(rdm_seed + d)  # Test
        print(param["dataset"][d])
        param["path_S"] = save_path_S + param["dataset"][d] + "1.pickle"
        param["path_T"] = save_path_T + param["dataset"][d] + "2.pickle"
        Sx, Sy, Tx, Ty, T_pos, dict_s, dict_t = load_features(param["path_S"], param["path_T"])

        # Sy_available = [None, Sy[1]]
        label = np.unique(Sy[1])
        Ty_available_1 = np.zeros_like(Ty[da_pos]) - 1
        for i in label:
            size = min(param["semi_supervised"][da_pos][i], len(np.where(Ty[da_pos] == i)[0]))
            Ty_available_1[np.random.choice(np.where(Ty[da_pos] == i)[0], size=size, replace=False)] = i
        # Ty_available = [None, Ty_available_1]

        keys_user, count = np.unique(Sy[0], return_counts=True)
        if (keys_user == np.unique(Ty[0])).all():
            transdict_user = dict(zip(keys_user, np.arange(len(keys_user))))
            Sy[0] = np.array([transdict_user[index_user] for index_user in Sy[0]])
            Ty[0] = np.array([transdict_user[index_user] for index_user in Ty[0]])
            Ty_available_0 = Ty[0].copy()
            Ty_available_0[np.random.choice(len(Ty_available_0),
                                            size=len(Ty_available_0) - param["semi_supervised"][0][0],
                                            replace=False)] = -1
            Sy_available_0 = Sy[0]
        else:
            Ty_available_0 = None
            Sy_available_0 = None
        Sy_available = [Sy_available_0, Sy[1]]
        Ty_available = [Ty_available_0, Ty_available_1]
        # print(Sy_available)
        # print(Ty_available)
        meansAcc = {}
        stdsAcc = {}
        totalTime = {}

        for name in param["adaptationAlgoUsed"]:
            meansAcc[name] = []
            stdsAcc[name] = []
            totalTime[name] = 0

        return_results = {}

        numberIteration_temp = param["numberIteration"]
        results = {}
        ot_dists = {}
        times = {}
        for name in param["adaptationAlgoUsed"]:
            # print(param["best_param"])
            if param["best_param"]:
                # print(param["pickle_path"] + "best_param/" + name + param["pickle_name"].split("/")[-1] + ".pickle")
                # print(param["pickle_path"] + param["pickle_name"] + param["param_pickle"] + ".pickle")
                if param["cheat"]:
                    cheat = "cheat"
                else:
                    cheat = ""
                if param["different"]:
                    with open(param["pickle_path"] + "best_param/" + name + param["pickle_name"].split("/")[-1] + "_" +
                              param["dataset"][d] + cheat + ".pickle",
                              "rb") as handle:
                        dict_name = pickle.load(handle)
                else:
                    with open(param["pickle_path"] + "best_param/" + name + param["pickle_name"].split("/")[
                        -1] + cheat + ".pickle",
                              "rb") as handle:
                        dict_name = pickle.load(handle)
                for e in ["eta", "epsilon"]:
                    if param_to_keep is None or not (e in param_to_keep):
                        param[e] = dict_name["param"][e]
                        # print("algo", name)

            results[name] = []
            times[name] = []
            ot_dists[name] = []

            # list of method that are not random
            if name in []:
                param["numberIteration"] = 1
            else:
                param["numberIteration"] = numberIteration_temp

            for iteration in range(param["numberIteration"]):
                startTime = time.time()
                np.random.seed(rdm_seed + iteration * 45 + 498861)

                # Adapt the data
                prediction, T, pred_user, ot_dist = adaptData(algo=name,
                                                              Sx=Sx, Sy=Sy, Sy_available=Sy_available,
                                                              Tx=Tx, Ty=Ty, Ty_available=Ty_available,
                                                              T_pos=T_pos, param=param, da_pos=da_pos)

                results[name].append(getAccuracy(prediction, Ty[da_pos]))
                ot_dists[name].append(ot_dist)
                times[name].append(time.time() - startTime)
                if param["test"]:
                    break
            if param["best_param"]:
                for e in ["eta", "epsilon"]:
                    if param_to_keep is None or not (e in param_to_keep):
                        param[e] = -1

        for name in param["adaptationAlgoUsed"]:
            meanAcc = np.mean(results[name])
            stdAcc = np.std(results[name])
            meansAcc[name].append(meanAcc)
            stdsAcc[name].append(stdAcc)
            totalTime[name] += sum(times[name])
            print("     {:4.1f}".format(meanAcc) + "  {:3.1f}".format(stdAcc) +
                  "   {:3.6f}".format(np.mean(ot_dists[name])) +
                  "  {:6}".format(name) + " {:6.2f}s".format(sum(times[name])))
            return_results[name] = {"mean": meanAcc, "std": stdAcc, "ot_dists": np.mean(ot_dists[name]),
                                    "ot_dists_std": np.std(ot_dists[name])}

        param["path_S"], param["path_T"] = save_path_S, save_path_T
        my_dict = {"result": return_results,
                   "param": param,
                   "dict_s": dict_s,
                   "dict_t": dict_t}
        total_dict[param["dataset"][d]] = my_dict

    if param["test"]:
        return None
    if param["save_pickle"]:
        if param["cheat"]:
            with open(param["pickle_path"] + param["pickle_name"] + "cheat" + param["param_pickle"] + ".pickle",
                      "wb") as handle:
                pickle.dump(total_dict, handle)
        else:
            with open(param["pickle_path"] + param["pickle_name"] + param["param_pickle"] + ".pickle", "wb") as handle:
                pickle.dump(total_dict, handle)

    return total_dict


def param_experiment(param, da_pos=1, rdm_seed=123456):
    create_new_dataset = param["new_dataset"]
    param_to_modified = [param["parameters_analysed"]]
    if not (param["parameters_analysed_values"] is None):
        param["parameters_analysed_values"] = param["parameters_analysed_values"].split(",")
    if param["parameters_analysed"] == "epsilon":
        if param["parameters_analysed_values"] is None:
            parameters_analysed = [[0], [0.00001], [0.0001], [0.001], [0.005], [0.01], [0.05], [0.1], [0.5], [1], [5],
                                   [10], [50], [100]]
        else:
            parameters_analysed = [[float(i)] for i in param["parameters_analysed_values"]]
        param_for_pickle = parameters_analysed
    elif param["parameters_analysed"] == "eta":
        if param["parameters_analysed_values"] is None:
            parameters_analysed = [[0], [0.0001], [0.0005], [0.001], [0.005], [0.01], [0.05], [0.1], [0.5], [1], [5],
                                   [10]]
        else:
            parameters_analysed = [[float(i)] for i in param["parameters_analysed_values"]]
        # [[0.5], [1], [5], [10]]
        param_for_pickle = parameters_analysed
    elif param["parameters_analysed"] == "noise":
        n_movies = [3500, 3500, 10000, 10000, 5000, 5000]
        # genres = [["Thriller_Crime_Drama", "Fantasy_Sci-Fi"],
        #           ["Thriller_Crime_Drama", "Children's_Animation"],
        #           ["Thriller_Crime_Drama", "War_Western"],
        #           ["Fantasy_Sci-Fi", "Children's_Animation"],
        #           ["Fantasy_Sci-Fi", "War_Western"],
        #           ["Children's_Animation", "War_Western"]]
        names = [d[1:] for d in param["dataset"]]
        noise = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        parameters_analysed = []
        for i, name in enumerate(names):
            if create_new_dataset:
                utils.create_movies_dataset(path="./Movies/subdataset/", n_user=100, n_movie=n_movies[i],
                                            n_per_class=[100, 100], name=name, genres=name,
                                            additional_path_save="noise/")
                sparsity = 1
                for a in range(1, 3):
                    [utils.df_to_numpy(path1="./Movies/subdataset/noise/movies_2D" + name + str(a) + ".csv",
                                       path2="./Movies/subdataset/noise/movies_info" + name + str(a) + ".csv",
                                       outpath="./Movies/subdataset/noise/C" + name + str(n) + "_1_" + str(
                                           a) + ".pickle",
                                       noise=n, sparsity=sparsity) for n in noise]

        param_to_modified = ["dataset"]
        for n in noise:
            parameters_analysed.append([["C" + name + str(n) + "_" + "1" + "_" for name in names]])
        # parameters_analysed = [parameters_analysed]
        param["path_S"] = "./Movies/subdataset/noise/"
        param["path_T"] = "./Movies/subdataset/noise/"

        param_for_pickle = [[n] for n in noise]
    elif param["parameters_analysed"] == "sparsity":
        assert False

    elif param["parameters_analysed"] == "nb_samples":
        param_to_modified = ["nb_samples", "nb_samples_t"]
        parameters_analysed = [[[i], [i]] for i in [1, 5, 10, 50, 100, 500, 1000]]
        # parameters_analysed = [[[1], [1]], [[5], 5], [10, 10], [50, 50], [100, 100], [500, 500], [1000, 1000]]
        param_for_pickle = [[i] for i in [1, 5, 10, 50, 100, 500, 1000]]
    elif param["parameters_analysed"] == "supervision_user":
        param_to_modified = ["semi_supervised"]
        nb_user_known = [[0], [5], [10], [50], [100]]
        parameters_analysed = [[[i, param["semi_supervised"][1]]] for i in nb_user_known]
        param_for_pickle = nb_user_known
        parameters_analysed = parameters_analysed
    elif param["parameters_analysed"] == "supervision_movie":
        param_to_modified = ["semi_supervised"]
        nb_movie_known = [[0], [1], [5], [10], [50]]
        parameters_analysed = [[[param["semi_supervised"][0], [i[0], i[0]]]] for i in nb_movie_known]
        parameters_analysed = parameters_analysed
        param_for_pickle = nb_movie_known
    else:
        assert False

    my_dict = {param["parameters_analysed"]: parameters_analysed, "param": param, "return_results": []}
    param["pickle_name"] = param["parameters_analysed"] + "/" + param["pickle_name"]
    # print("parameters_analysed", parameters_analysed)
    for i, parameter_analysed in enumerate(parameters_analysed):

        for p in range(len(param_to_modified)):
            param[param_to_modified[p]] = parameter_analysed[p]

        param["param_pickle"] = "_" + str(param_for_pickle[i][0]) + "__"
        # print(param["param_pickle"])
        dict_i = run_adaptation(param, da_pos=da_pos, rdm_seed=rdm_seed, param_to_keep=param_to_modified)
        for dataset in dict_i:
            for j in range(len(param["adaptationAlgoUsed"])):
                my_dict[str(param_for_pickle[i][0]) + "|" + dataset + "|" + param["adaptationAlgoUsed"][j]] = \
                    dict_i[dataset]["result"][param["adaptationAlgoUsed"][j]]
    # print(my_dict)
    print(param["pickle_path"] + param["pickle_name"] + "final" + ".pickle")
    if param["save_pickle"] and param["parameters_analysed_values"] is None:
        with open(param["pickle_path"] + param["pickle_name"] + "final" + ".pickle", "wb") as handle:
            pickle.dump(my_dict, handle)


def select_best_hyper(param, da_pos=1, rdm_seed=123456):
    assert len(param["adaptationAlgoUsed"]) == 1
    time_before_loop = time.time()
    print("\nTrain : ", param["adaptationAlgoUsed"])

    # This is the number of iteration to each set of hyperparameter to avoid randomness
    nb_train = param["numberIteration"]

    # The next lines will define the range of cross validation.
    epsilon = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    eta = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    number_iteration_cross_val = 0
    best_ot_dists = [np.inf] * len(param["dataset"])
    best_mean = [-np.inf] * len(param["dataset"])
    best_dict = [0] * len(param["dataset"])
    mean_dict = [0] * len(param["dataset"])
    while time.time() - time_before_loop < 3600 * param["time_cross_val"] and number_iteration_cross_val < 1000:
        np.random.seed(4896 * number_iteration_cross_val + 5272)
        param["epsilon"] = epsilon[np.random.randint(len(epsilon))]
        param["eta"] = eta[np.random.randint(len(eta))]

        print("Running :", number_iteration_cross_val,
              "epsilon", param["epsilon"],
              "eta", param["eta"], )
        dict_i = run_adaptation(param, da_pos=da_pos, rdm_seed=rdm_seed)
        for d in range(len(param["dataset"])):
            ot_dists = dict_i[param["dataset"][d]]["result"][param["adaptationAlgoUsed"][0]]["ot_dists"]
            mean_ = dict_i[param["dataset"][d]]["result"][param["adaptationAlgoUsed"][0]]["mean"]
            if ot_dists < best_ot_dists[d]:
                best_ot_dists[d] = ot_dists
                best_dict[d] = deepcopy(dict_i[param["dataset"][d]])
            if mean_ > best_mean[d]:
                best_mean[d] = mean_
                mean_dict[d] = deepcopy(dict_i[param["dataset"][d]])

        time.sleep(1.)  # Allow to stop the program with ctrl-C if in a try
        number_iteration_cross_val += 1
        # Special case were there is no hyperparameters to tune.
        if param["adaptationAlgoUsed"][0] in ["smart_random", "target_classifier"] and number_iteration_cross_val >= 1:
            print("No param to tune")
            break
    print("Time for the cross validation:", time.time() - time_before_loop, "s")
    # print(best_dict)
    # print(mean_dict)
    if param["save_pickle"]:
        for d in range(len(param["dataset"])):
            with open(param["pickle_path"] + "/best_param/" + param["adaptationAlgoUsed"][0] + param[
                "pickle_name"] + "_" + param["dataset"][d] + ".pickle", "wb") as handle:
                pickle.dump(best_dict[d], handle)
            with open(param["pickle_path"] + "/best_param/" + param["adaptationAlgoUsed"][0] + param[
                "pickle_name"] + "_" + param["dataset"][d] + "cheat.pickle", "wb") as handle:
                pickle.dump(mean_dict[d], handle)


def latex_table(path_pickle, list_exp=['C_T_F', 'C_T_C', 'C_T_W', 'C_F_C', 'C_F_W', 'C_C_W'],
                names_algo=None,
                list_exp_plot=None, list_algo_plot=None):
    dict_ = load_change_name1(path_pickle)
    if names_algo is None:
        names_algo = dict_[list_exp[0]]["param"]["adaptationAlgoUsed"]

    if list_exp_plot is None:
        list_exp_plot = list_exp
    if list_algo_plot is None:
        list_algo_plot = names_algo

    latex_tabular = {"Datasets": []}
    mean = np.zeros((len(list_exp), len(names_algo)))
    std = np.zeros((len(list_exp), len(names_algo)))
    # print(names_algo)
    for i in range(len(list_exp)):
        latex_tabular["Datasets"].append(list_exp_plot[i])
        for j, algo in enumerate(names_algo):
            mean[i, j] = dict_[list_exp[i]]["result"][algo]["mean"]
            std[i, j] = dict_[list_exp[i]]["result"][algo]["std"]
    mean_ = np.zeros((len(list_exp) + 1, len(names_algo)))
    mean_[np.ix_(np.arange(mean.shape[0]), np.arange(mean.shape[1]))] = mean
    for j, algo in enumerate(names_algo):
        if np.mean(std[:, j]) > 0:
            latex_tabular[list_algo_plot[j]] = [str(np.round(mean[i, j], 1)) + "$\pm$" + \
                                                str(np.round(std[i, j], 1)) for i in range(len(list_exp))]
            latex_tabular[list_algo_plot[j]].append(str(np.round(np.mean(mean[:, j]), 1)) + "$\pm$" +
                                                    str(np.round(np.mean(std[:, j]), 1)))
        else:
            latex_tabular[list_algo_plot[j]] = [str(np.round(mean[i, j], 1)) for i in range(len(list_exp))]
            latex_tabular[list_algo_plot[j]].append(str(np.round(np.mean(mean[:, j]), 1)))
        mean_[i + 1, j] = np.mean(mean[:, j])
    textbf = np.argmax(mean_, axis=1)
    for j, algo in enumerate(names_algo):
        for i in range(len(textbf)):
            if textbf[i] == j:
                a = latex_tabular[list_algo_plot[j]][i].split("$\pm$")
                if len(a) == 2:
                    latex_tabular[list_algo_plot[j]][i] = "\textbf{" + a[0] + '}' + "$\pm$" + a[1]
                elif len(a) == 1:
                    latex_tabular[list_algo_plot[j]][i] = "\textbf{" + a[0] + '}'
                else:
                    assert False
    latex_tabular["Datasets"].append("AVG")
    df = pd.DataFrame(latex_tabular)
    print("\\begin{table}")
    print("\centering")
    print("\\caption{Accuracy with the best hyperparameters sets on 6 DA tasks.(" + path_pickle.split("/")[-1][
                                                                                    2:-10] + ")}")
    print("\\label{tab:DA" + path_pickle.split("/")[-1][2:-10] + "}")
    print(df.to_latex(index=False, escape=False))
    print("\\end{table}")


def latex_hyper(path_pickle,
                list_param_plot=None, list_algo_plot=None, names_algo=None,
                plot=True, color=None, type=None, log=True, labelx=["", ""], labely=["", ""],
                loc="best", figsize=[(7, 8), (7, 8)], list_symbol_plot=[]):
    import tikzplotlib
    # with open(path_pickle, 'rb') as handle:
    #     dict_ = pickle.load(handle)
    dict_ = load_change_name1(path_pickle)

    if names_algo is None:
        names_algo = dict_["param"]["adaptationAlgoUsed"]
    list_exp = dict_["param"]["dataset"]
    parameters_analysed = path_pickle.split("/")[-2]
    if list_param_plot is None:
        list_param_plot = [x[0] for x in dict_[parameters_analysed]]
    if list_algo_plot is None:
        list_algo_plot = names_algo

    latex_tabular = {"dataset": []}
    mean = np.zeros((len(names_algo), len(dict_[parameters_analysed]) + 1))
    std = np.zeros((len(names_algo), len(dict_[parameters_analysed])))
    ot_dists = np.zeros((len(names_algo), len(dict_[parameters_analysed])))
    for k in range(len(dict_[parameters_analysed])):
        latex_tabular["dataset"].append(list_param_plot[k])
    for j, algo in enumerate(names_algo):
        for k in range(len(dict_[parameters_analysed])):
            mean_jk, std_jk, ot_dists_jk = [], [], []
            for i in range(len(list_exp)):
                key = str(list_param_plot[k])
                key = key + "|" + list_exp[i] + "|" + algo
                mean_jk.append(dict_[key]["mean"])
                std_jk.append(dict_[key]["std"])
                ot_dists_jk.append(dict_[key]["ot_dists"])
            mean[j, k] = np.mean(mean_jk)
            std[j, k] = np.mean(std_jk)
            ot_dists[j, k] = np.mean(ot_dists_jk)
            if ot_dists[j, k] >= 99999:
                ot_dists[j, k] = -1

    if plot:
        plt.figure(figsize=figsize[0])
        for j, algo in enumerate(names_algo):
            if color is None:
                plt.plot(list_param_plot, mean[j, 0:-1], label=list_algo_plot[j], marker=list_symbol_plot[j],
                         markersize=6)
            else:
                plt.plot(list_param_plot, mean[j, 0:-1], label=list_algo_plot[j],
                         color=color[j], marker=list_symbol_plot[j], markersize=6)
        plt.xlabel(labelx[0])
        plt.ylabel(labely[0])
        if log:
            plt.xscale("log")
        plt.legend(loc=loc)
        tikzplotlib.save("DA/fig/" + parameters_analysed + ".tex")
        plt.show()
        plt.figure(figsize=figsize[1])
        for j, algo in enumerate(names_algo):
            if color is None:
                plt.plot(list_param_plot, ot_dists[j, :], label=list_algo_plot[j], marker=list_symbol_plot[j],
                         markersize=6)
            else:
                plt.plot(list_param_plot, ot_dists[j, :], label=list_algo_plot[j],
                         color=color[j], marker=list_symbol_plot[j], markersize=6)
            plt.xlabel(labelx[1])
            plt.ylabel(labely[1])
        if log:
            plt.xscale("log")
        plt.legend(loc=loc)
        tikzplotlib.save("DA/fig/" + parameters_analysed + "_W_dist.tex")
        plt.show()


def latex_hyper_without_final(path_pickle,
                              list_param,
                              names_algo=None, list_param_plot=None, list_algo_plot=None,
                              plot=True, color=None, type=None, log=True, labelx=["", ""], labely=["", ""],
                              loc=["best", "best"], rescale=True, figsize=[(7, 8), (7, 8)],
                              list_symbol_plot=[], fontsize=10, value_cross=False, delete_yaxis=False,
                              legend=[True, True], name_scale="epsilon"):
    import tikzplotlib
    plt.rcParams.update({'font.size': fontsize})

    dict_list_param = []
    for i in range(len(list_param)):
        list_param[i] = str(list_param[i])
        # if list_param[i] == "1e-05":
        #     list_param[i] = 0.0000
    for p in list_param:
        # print(p)
        # with open(path_pickle + "DA_" + p + "__.pickle", 'rb') as handle:
        #     dict_list_param.append(pickle.load(handle))
        dict_list_param.append(load_change_name1(path_pickle + "DA_" + p + "__.pickle"))

    # print(dict_list_param[0])
    if names_algo is None:
        names_algo = dict_list_param[0]["C_T_F"]["param"]["adaptationAlgoUsed"]

    list_exp = dict_list_param[0]["C_T_F"]["param"]["dataset"]
    parameters_analysed = path_pickle.split("/")[-2]
    if list_param_plot is None:
        list_param_plot = list_param
    if list_algo_plot is None:
        list_algo_plot = names_algo

    latex_tabular = {"dataset": []}
    mean = np.zeros((len(names_algo), len(list_param) + 1))
    std = np.zeros((len(names_algo), len(list_param)))
    ot_dists = np.zeros((len(names_algo), len(list_param)))
    ot_dists_std = np.zeros((len(names_algo), len(list_param)))
    # print(names_algo)
    for k in range(len(list_param)):
        latex_tabular["dataset"].append(list_param_plot[k])
    for j, algo in enumerate(names_algo):
        for k, p in enumerate(list_param):
            mean_jk, std_jk, ot_dists_jk, ot_dists_jk_std = [], [], [], []
            for i in range(len(list_exp)):
                mean_jk.append(dict_list_param[k][list_exp[i]]["result"][algo]["mean"])
                std_jk.append(dict_list_param[k][list_exp[i]]["result"][algo]["std"])
                ot_dists_jk.append(dict_list_param[k][list_exp[i]]["result"][algo]["ot_dists"])
                ot_dists_jk_std.append(dict_list_param[k][list_exp[i]]["result"][algo]["ot_dists_std"])

            mean[j, k] = np.mean(mean_jk)
            std[j, k] = np.mean(std_jk)
            ot_dists[j, k] = np.mean(ot_dists_jk)
            ot_dists_std[j, k] = np.mean(ot_dists_jk_std)
            if ot_dists[j, k] >= 99999:
                ot_dists[j, k] = -1

    list_param_plot = np.array([float(i) for i in list_param_plot])
    if plot:
        plt.figure(figsize=figsize[0])
        for j, algo in enumerate(names_algo):
            # print(algo)
            if algo in ['target_classifier', 'smart_random']:
                ot_discard = np.array([True for _ in range(len(ot_dists[j, :]))])
            else:
                ot_discard = ot_dists[j, :] != -1

            if color is None:
                plt.plot(list_param_plot[ot_discard], mean[j, 0:-1][ot_discard],
                         label=list_algo_plot[j], marker=list_symbol_plot[j], markersize=6)
                plt.fill_between(list_param_plot[ot_discard],
                                 mean[j, :-1][ot_discard] + std[j, :][ot_discard],
                                 mean[j, :-1][ot_discard] - std[j, :][ot_discard],
                                 alpha=0.2)
            else:
                plt.plot(list_param_plot[ot_discard], mean[j, 0:-1][ot_discard], label=list_algo_plot[j],
                         color=color[j], marker=list_symbol_plot[j], markersize=6)
                plt.fill_between(list_param_plot[ot_discard],
                                 mean[j, :-1][ot_discard] + std[j, :][ot_discard],
                                 mean[j, :-1][ot_discard] - std[j, :][ot_discard],
                                 alpha=0.2,
                                 color=color[j],
                                 linestyle=type[j])
            if value_cross:
                if algo in ['target_classifier', 'smart_random']:
                    ot_discard = np.array([True for _ in range(len(ot_dists[j, :]))])
                else:
                    ot_discard = ot_dists[j, :] != -1
                # print(algo)
                # print(ot_dists[j, :])
                if "DA/DA_pickle/supervision_user/" == path_pickle and algo in ["gromov", "ScalableGW",
                                                                                "target_classifier", "smart_random"]:
                    continue

                plt.plot(list_param_plot[ot_discard][np.argmin(ot_dists[j, ot_discard])],
                         mean[j, :-1][ot_discard][np.argmin(ot_dists[j, ot_discard])], markersize=7,
                         marker=list_symbol_plot[j], color="black")

        plt.xlabel(labelx[0])
        name_save = ""
        if delete_yaxis:
            name_save = "_noy"
        else:
            plt.ylabel(labely[0])

        if log:
            plt.xscale("log")
        if legend[0]:
            plt.legend(loc=loc[0], fontsize=fontsize)
        # plt.savefig("DA/fig/" + parameters_analysed + ".png")
        if value_cross:
            tikzplotlib.save("DA/fig/" + parameters_analysed + name_save + "_c.tex")
            with fileinput.FileInput("./DA/fig/" + parameters_analysed + name_save + "_c.tex", inplace=True) as file:
                for line in file:
                    print(
                        line.replace("begin{tikzpicture}", "begin{tikzpicture}" + "[scale=\\scale" + name_scale + "]"),
                        end='')
        else:
            tikzplotlib.save("DA/fig/" + parameters_analysed + name_save + ".tex")
            with fileinput.FileInput("./DA/fig/" + parameters_analysed + name_save + ".tex", inplace=True) as file:
                for line in file:
                    print(
                        line.replace("begin{tikzpicture}", "begin{tikzpicture}" + "[scale=\\scale" + name_scale + "]"),
                        end='')
        plt.show()
        plt.figure(figsize=figsize[1])

        for j, algo in enumerate(names_algo):
            if algo in ['target_classifier', 'smart_random']:
                continue
            if algo in ['target_classifier', 'smart_random']:
                ot_discard = np.array([True for _ in range(len(ot_dists[j, :]))])
            else:
                ot_discard = ot_dists[j, :] != -1
            if rescale and np.max(ot_dists[j, :][ot_discard]) - np.min(ot_dists[j, :][ot_discard]) > 0:
                ot_dists[j, :] = (ot_dists[j, :] - np.min(ot_dists[j, :][ot_discard])) / (
                            np.max(ot_dists[j, :][ot_discard]) - np.min(ot_dists[j, :][ot_discard]))
                ot_dists_std[j, :] = (ot_dists_std[j, :] - np.min(ot_dists[j, :][ot_discard])) / (
                        np.max(ot_dists[j, :][ot_discard]) - np.min(ot_dists[j, :][ot_discard]))
            if color is None:
                plt.plot(list_param_plot[ot_discard],
                         ot_dists[j, :][ot_discard],
                         label=list_algo_plot[j], marker=list_symbol_plot[j], markersize=6)

                plt.fill_between(list_param_plot[ot_discard],
                                 ot_dists[j, :][ot_discard] + ot_dists_std[j, :][ot_discard],
                                 ot_dists[j, :][ot_discard] - ot_dists_std[j, :][ot_discard],
                                 alpha=0.2)
            else:
                plt.plot(list_param_plot[ot_discard], ot_dists[j, :][ot_discard], label=list_algo_plot[j],
                         color=color[j], marker=list_symbol_plot[j], markersize=6)
                plt.fill_between(list_param_plot[ot_discard],
                                 ot_dists[j, :][ot_discard] + ot_dists_std[j, :][ot_discard],
                                 ot_dists[j, :][ot_discard] - ot_dists_std[j, :][ot_discard],
                                 alpha=0.2,
                                 color=color[j],
                                 linestyle=type[j])
        plt.xlabel(labelx[1])
        plt.ylabel(labely[1])
        if log:
            plt.xscale("log")
        if legend[1]:
            plt.legend(loc=loc[1], fontsize=fontsize)
        # plt.savefig("DA/fig/" + parameters_analysed + "_W_dist.png")
        tikzplotlib.save("DA/fig/" + parameters_analysed + "_W_dist.tex")
        with fileinput.FileInput("DA/fig/" + parameters_analysed + "_W_dist.tex", inplace=True) as file:
            for line in file:
                print(line.replace("begin{tikzpicture}", "begin{tikzpicture}" + "[scale=\\scale" + name_scale + "W]"),
                      end='')
        plt.show()


def latex_hyper_without_final_separated(path_pickle,
                                        list_param,
                                        names_algo=None, list_param_plot=None, list_algo_plot=None,
                                        plot=True, color=None, type=None, log=True, labelx=["", ""], labely=["", ""],
                                        loc=["best", "best"], rescale=True, figsize=[(7, 8), (7, 8)],
                                        list_symbol_plot=[], legend=[True, True], fontsize=11):
    import tikzplotlib
    plt.rcParams.update({'font.size': fontsize})

    dict_list_param = []
    for i in range(len(list_param)):
        list_param[i] = str(list_param[i])
    for p in list_param:
        # with open(path_pickle + "DA_" + p + "__.pickle", 'rb') as handle:
        dict_list_param.append(load_change_name1(path_pickle + "DA_" + p + "__.pickle"))
    # print(dict_list_param[0])
    if names_algo is None:
        names_algo = dict_list_param[0]["C_T_F"]["param"]["adaptationAlgoUsed"]
    list_exp = dict_list_param[0]["C_T_F"]["param"]["dataset"]
    parameters_analysed = path_pickle.split("/")[-2]
    if list_param_plot is None:
        list_param_plot = list_param
    if list_algo_plot is None:
        list_algo_plot = names_algo

    mean = np.zeros((len(names_algo), len(list_param) + 1))
    std = np.zeros((len(names_algo), len(list_param)))
    ot_dists = np.zeros((len(names_algo), len(list_param)))
    ot_dists_std = np.zeros((len(names_algo), len(list_param)))
    list_param_plot = np.array([float(a) for a in list_param_plot])

    for i in range(len(list_exp)):
        for j, algo in enumerate(names_algo):
            for k, p in enumerate(list_param):
                mean[j, k] = dict_list_param[k][list_exp[i]]["result"][algo]["mean"]
                std[j, k] = dict_list_param[k][list_exp[i]]["result"][algo]["std"]
                # std[j, k] = np.std(mean_jk)
                ot_dists[j, k] = dict_list_param[k][list_exp[i]]["result"][algo]["ot_dists"]
                ot_dists_std[j, k] = np.mean(dict_list_param[k][list_exp[i]]["result"][algo]["ot_dists_std"])
                if ot_dists[j, k] >= 99999:
                    ot_dists[j, k] = -1

        if plot:
            plt.figure(figsize=figsize[0])
            for j, algo in enumerate(names_algo):
                if algo in ['target_classifier', 'smart_random']:
                    ot_discard = np.array([True for _ in range(len(ot_dists[j, :]))])
                else:
                    ot_discard = ot_dists[j, :] != -1
                # print(list_param_plot[ot_dists[j, :] == -1], )
                if color is None:
                    plt.plot(list_param_plot[ot_discard], mean[j, 0:-1][ot_discard],
                             label=list_algo_plot[j], marker=list_symbol_plot[j], markersize=6)
                    plt.fill_between(list_param_plot[ot_discard],
                                     mean[j, :-1][ot_discard] + std[j, :][ot_discard],
                                     mean[j, :-1][ot_discard] - std[j, :][ot_discard],
                                     alpha=0.2)
                else:
                    plt.plot(list_param_plot[ot_discard], mean[j, 0:-1][ot_discard],
                             label=list_algo_plot[j],
                             color=color[j], marker=list_symbol_plot[j], markersize=6)
                    plt.fill_between(list_param_plot[ot_discard],
                                     mean[j, :-1][ot_discard] + std[j, :][ot_discard],
                                     mean[j, :-1][ot_discard] - std[j, :][ot_discard],
                                     alpha=0.2,
                                     color=color[j],
                                     linestyle=type[j])
            plt.xlabel(labelx[0])
            plt.ylabel(labely[0])
            if log:
                plt.xscale("log")
            if legend[0]:
                plt.legend(loc=loc[0], fontsize=fontsize)
            # plt.savefig("DA/fig/" + parameters_analysed + ".png")
            name_scale = "epsilon"
            tikzplotlib.save("DA/fig/" + parameters_analysed + list_exp[i] + ".tex")
            with fileinput.FileInput("DA/fig/" + parameters_analysed + list_exp[i] + ".tex", inplace=True) as file:
                for line in file:
                    print(
                        line.replace("begin{tikzpicture}", "begin{tikzpicture}" + "[scale=\\scale" + name_scale + "W]"),
                        end='')
            plt.show()
            plt.figure(figsize=figsize[1])

            for j, algo in enumerate(names_algo):
                if algo in ['target_classifier', 'smart_random']:
                    continue
                if algo in ['target_classifier', 'smart_random']:
                    ot_discard = np.array([True for _ in range(len(ot_dists[j, :]))])
                else:
                    ot_discard = ot_dists[j, :] != -1
                if rescale and np.max(ot_dists[j, :][ot_discard]) - np.min(ot_dists[j, :][ot_discard]) > 0:
                    ot_dists[j, :] = (ot_dists[j, :] - np.min(ot_dists[j, :][ot_discard])) / (
                            np.max(ot_dists[j, :][ot_discard]) - np.min(ot_dists[j, :][ot_discard]))
                    ot_dists_std[j, :] = (ot_dists_std[j, :] - np.min(ot_dists[j, :][ot_discard])) / (
                            np.max(ot_dists[j, :][ot_discard]) - np.min(ot_dists[j, :][ot_discard]))
                if color is None:
                    plt.plot(list_param_plot[ot_discard],
                             ot_dists[j, :][ot_discard],
                             label=list_algo_plot[j], marker=list_symbol_plot[j], markersize=6)
                    plt.fill_between(list_param_plot[ot_discard],
                                     ot_dists[j, :][ot_discard] + ot_dists_std[j, :][ot_discard],
                                     ot_dists[j, :][ot_discard] - ot_dists_std[j, :][ot_discard],
                                     alpha=0.2)
                else:
                    plt.plot(list_param_plot[ot_discard], ot_dists[j, :][ot_discard], label=list_algo_plot[j],
                             color=color[j], marker=list_symbol_plot[j], markersize=6)
                    plt.fill_between(list_param_plot[ot_discard],
                                     ot_dists[j, :][ot_discard] + ot_dists_std[j, :][ot_discard],
                                     ot_dists[j, :][ot_discard] - ot_dists_std[j, :][ot_discard],
                                     alpha=0.2,
                                     color=color[j],
                                     linestyle=type[j])
            plt.xlabel(labelx[1])
            plt.ylabel(labely[1])
            if log:
                plt.xscale("log")
            if legend[1]:
                plt.legend(loc=loc[1], fontsize=fontsize)
            # plt.savefig("DA/fig/" + parameters_analysed + "_W_dist.png")
            tikzplotlib.save("DA/fig/" + parameters_analysed + list_exp[i] + "_W_dist.tex")
            name_scale = "epsilon"
            with fileinput.FileInput("DA/fig/" + parameters_analysed + list_exp[i] + "_W_dist.tex",
                                     inplace=True) as file:
                for line in file:
                    print(
                        line.replace("begin{tikzpicture}", "begin{tikzpicture}" + "[scale=\\scale" + name_scale + "W]"),
                        end='')
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OTT')
    parser.add_argument('--path_S', type=str, default="Movies/subdataset/")
    parser.add_argument('--path_T', type=str, default="Movies/subdataset/")
    parser.add_argument('--dataset', type=str, default="C_T_F,C_T_C,C_T_W,C_F_C,C_F_W,C_C_W")

    parser.add_argument('--param1', type=str, default="")
    parser.add_argument('--param2', type=str, default="")
    parser.add_argument('--param3', type=str, default="")
    parser.add_argument('--time_cross_val', type=float, default=24)
    parser.add_argument('--numberIteration', type=int, default=10,
                        help="Number of iterations of each method, this is usefull for random methods.")
    parser.add_argument('--adaptationAlgoUsed', type=str,
                        default='smart_random,ScalableGW,gromov,COT,target_classifier,OTT',
                        help="Name of the method that should be used. Method should be separate by comma if more than \
                             one method is needed (OTT,random)")
    parser.add_argument('--semi_supervised', type=str, default="0/1,1")

    parser.add_argument('--p', type=str, default="uniform")
    parser.add_argument('--q', type=str, default="uniform")
    parser.add_argument('--loss_fun', type=str, default="L2")
    parser.add_argument('--M', type=str, default="None")
    parser.add_argument('--T', type=str, default="uniform")
    parser.add_argument('--alpha', type=str, default="1")
    parser.add_argument('--nb_iter', type=int, default=1000, help="number max of iterations of OTT")
    parser.add_argument('--nb_samples', type=str, default="1000")
    parser.add_argument('--nb_samples_t', type=str, default="1000")
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--KL', type=float, default=1)
    parser.add_argument('--eta', type=float, default=0.001)
    parser.add_argument('--L2', type=float, default=0)
    parser.add_argument('--rdm', type=int, default=1234567)

    parser.add_argument('--pickle_path', type=str, default="./DA/DA_pickle/",
                        help="Pickle_name to save pickle. It will have different location depending if the cross \
                        validation or the -s option or the specific comparaison is used")
    parser.add_argument('--pickle_name', type=str, default="DA")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Will print more information about the run")
    parser.add_argument("-b", "--best_param", action="store_true")
    parser.add_argument("-d", "--different", action="store_true", help="We look at each dataset independently")
    parser.add_argument("-c", "--cheat", action="store_true")
    parser.add_argument("-s", "--save_pickle", action="store_true",
                        help="Will save the data adapted, this can be huge for some dataset.")
    parser.add_argument("-n", "--new_dataset", action="store_true")
    parser.add_argument('--specific_comparaison', type=str, choices=["None", "param_experiment", "best_param"],
                        default="None", help="Will launch specific analysis.")
    parser.add_argument('--parameters_analysed', type=str, choices=["epsilon", "eta", "noise", "nb_samples",
                                                                    "supervision_user", "supervision_movie"],
                        default="None", help="Will launch specific analysis.")
    parser.add_argument('--parameters_analysed_values', type=str, default=None)
    parser.add_argument("-t", "--test", action="store_true",
                        help="Will run only one iteration and stop without saving anything")

    args = parser.parse_args()
    args = vars(args)
    print(args)
    args["param_pickle"] = "_" + args["param1"] + "_" + args["param2"] + "_" + args["param3"]
    if args["save_pickle"]:
        print("Name of the pickle file associated with this run: ",
              args["pickle_path"] + args["pickle_name"] + args["param_pickle"] + ".pickle")

    args["alpha"] = [float(i) for i in args["alpha"].split(",")]
    args["nb_samples"] = [int(i) for i in args["nb_samples"].split(",")]
    args["nb_samples_t"] = [None if i == "None" else int(i) for i in args["nb_samples_t"].split(",")]
    args["adaptationAlgoUsed"] = args["adaptationAlgoUsed"].split(",")
    copy = args["semi_supervised"]
    args["semi_supervised"] = []
    for j in copy.split("/"):
        args["semi_supervised"].append([int(i) for i in j.split(",")])
    args["dataset"] = args["dataset"].split(",")
    # if args["cheat"]:
    #     args["pickle_name"] = args["pickle_name"] + "cheat"
    if args["specific_comparaison"] == "None":
        if args["new_dataset"]:
            utils.create_dataset(args["rdm"])
        run_adaptation(param=args, rdm_seed=args["rdm"])
    elif args["specific_comparaison"] == "param_experiment":
        param_experiment(param=args, da_pos=1, rdm_seed=args["rdm"])
    elif args["specific_comparaison"] == "best_param":
        select_best_hyper(param=args, da_pos=1, rdm_seed=args["rdm"])
