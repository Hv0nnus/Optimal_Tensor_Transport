import os

import argparse

import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import pairwise_distances, f1_score
import umap
import matplotlib.pyplot as plt
import pandas as pd


import gromov
import ot
import sys

# sys.path.insert(0, "./adds")
from addsclustering import SPUR, SDP_known_k
from addsclustering.oracle import OracleTriplets
from addsclustering.similarity import get_AddS_triplets

sys.path.insert(0, "./tste")
import cy_tste
import struct


def loadlocal_mnist(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def mnist_model(labels_selected=[0, 1], n_per_features=[10, 10], sigma=1, features="umap"):
    if args["pickle_name"] == "mnist":
        path = "./data/MNIST/raw/"
        X, y = loadlocal_mnist(images_path=path + 'train-images-idx3-ubyte',
                               labels_path=path + 'train-labels-idx1-ubyte')
    else:
        assert False
    if features == "umap":
        reducer = umap.UMAP(random_state=42)
        reducer.n_components = args["umap_dim"]
        embedding = reducer.fit_transform(X)
    elif features == "initial":
        embedding = X
    labels_init = np.array(y, dtype=int)
    features_init = embedding
    features, labels = None, None,
    for i, label_selected in enumerate(labels_selected):
        selections = np.isin(labels_init, label_selected)
        labels_, features_ = labels_init[selections], features_init[selections]
        selections_rdm = np.random.choice(len(labels_), size=n_per_features[i])
        labels_, features_ = labels_[selections_rdm], features_[selections_rdm]
        if features is None:
            features = features_
            labels = labels_
        else:
            features = np.concatenate((features, features_), axis=0)
            labels = np.concatenate((labels, labels_), axis=0)

    pairwise_similarity = pairwise_distances(features)
    if args["pickle_name"] == "mnist":
        pairwise_similarity = pairwise_similarity / np.mean(pairwise_similarity)
    pairwise_similarity = np.exp(- pairwise_similarity / sigma)  # This line is useless when we only check hardly if
    # P_ij is higher or lower that P_ik. Is it usefull only for smooth version.
    # The fact that we use a similarity matrix instead of distance matrix make no difference.
    # print(pairwise_similarity)
    return features, pairwise_similarity, labels


def pred(algo, oracle, args, original_clusters=None):
    T_, X2_, ot_dist_ = None, None, -1
    if algo == "SPUR":
        adds_similarities = get_AddS_triplets(oracle, args["n_examples"])
        X, n_clusters_estimated = SPUR(K=adds_similarities,
                                       N=args["n_examples"],
                                       T=len(args["n_per_features"]),
                                       n_observ=args["n_comparisons"])
        classes_pred = KMeans(n_clusters=args["n_clusters"], n_init=args["numberIteration"],
                              random_state=args["seed"]).fit_predict(X.T)
    elif algo == "SPUR_supervised":
        adds_similarities = get_AddS_triplets(oracle, args["n_examples"])
        embedding, n_clusters_estimated = SPUR(K=adds_similarities,
                                               N=args["n_examples"],
                                               T=len(args["n_per_features"]),
                                               n_observ=args["n_comparisons"])
        embedding = embedding.T
        best_cost = np.inf
        for i in range(args["numberIteration"]):
            X_init = np.random.rand(args["n_clusters"], embedding.shape[1])
            X = ot.lp.free_support_barycenter(measures_locations=[embedding],
                                              measures_weights=[ot.unif(args["n_examples"])],
                                              X_init=X_init,
                                              b=args["proportion"])

            M = ot.utils.dist(X, embedding)
            T = ot.emd(args["proportion"], ot.unif(args["n_examples"]), M)
            cost = (M * T).sum()
            if cost < best_cost:
                classes_pred = np.argmax(T, axis=0)
                best_cost = cost
        # classes_pred = KMeans(n_clusters=args["n_clusters"], n_init=args["numberIteration"],
        #                       random_state=args["seed"]).fit_predict(X.T)
    elif algo == "SDP":
        adds_similarities = get_AddS_triplets(oracle, args["n_examples"])
        embedding = SDP_known_k(K=adds_similarities,
                                N=args["n_examples"],
                                n_clusters=len(args["n_per_features"]),
                                verbose=False, eps=1e-3, warm=None, get_res=False)
        classes_pred = KMeans(n_clusters=args["n_clusters"], n_init=args["numberIteration"],
                              random_state=args["seed"]).fit_predict(embedding.T)
    elif algo == "SDP_supervised":
        adds_similarities = get_AddS_triplets(oracle, args["n_examples"])
        embedding = SDP_known_k(K=adds_similarities,
                                N=args["n_examples"],
                                n_clusters=len(args["n_per_features"]),
                                verbose=False, eps=1e-3, warm=None, get_res=False)
        embedding = embedding.T
        best_cost = np.inf
        for i in range(args["numberIteration"]):
            X_init = np.random.rand(args["n_clusters"], embedding.shape[1])
            X = ot.lp.free_support_barycenter(measures_locations=[embedding],
                                              measures_weights=[ot.unif(args["n_examples"])],
                                              X_init=X_init,
                                              b=args["proportion"])

            M = ot.utils.dist(X, embedding)
            T = ot.emd(args["proportion"], ot.unif(args["n_examples"]), M)
            cost = (M * T).sum()
            if cost < best_cost:
                classes_pred = np.argmax(T, axis=0)
                best_cost = cost
        # classes_pred = KMeans(n_clusters=args["n_clusters"], n_init=args["numberIteration"],
        #                       random_state=args["seed"]).fit_predict(X.T)
    elif algo == "tste_supervised":

        oracle.n_quadruplets = len(oracle.comparisons.row)
        oracle_triplet = oracle.get_tSTE_comparisons()


        embedding = cy_tste.tste(oracle_triplet,
                                 no_dims=args["dims_embed"],
                                 lamb=0,
                                 alpha=None,
                                 verbose=False,
                                 max_iter=1000,
                                 save_each_iteration=False,
                                 initial_X=None,
                                 static_points=np.array([]),
                                 ignore_zeroindexed_error=True,
                                 num_threads=None,
                                 use_log=False)
        best_cost = np.inf
        for i in range(args["numberIteration"]):
            X_init = np.random.rand(args["n_clusters"], args["dims_embed"])
            X = ot.lp.free_support_barycenter(measures_locations=[embedding],
                                              measures_weights=[ot.unif(args["n_examples"])],
                                              X_init=X_init,
                                              b=args["proportion"])

            M = ot.utils.dist(X, embedding)
            T = ot.emd(args["proportion"], ot.unif(args["n_examples"]), M)
            cost = (M * T).sum()
            if cost < best_cost:
                classes_pred = np.argmax(T, axis=0)
                best_cost = cost
        if args["plot"]:
            for i in np.unique(classes_pred):
                plt.plot(embedding[classes_pred == i, 0], embedding[classes_pred == i, 1], "+")
            plt.savefig("./Barycenter/picture/triplet_supervised.pdf")
            plt.show()
            for i in np.unique(classes_pred):
                plt.plot(embedding[original_clusters == i, 0], embedding[original_clusters == i, 1], "+")
            plt.savefig("./Barycenter/picture/triplet_supervisedGT.pdf")
            plt.show()

    elif algo == "tste":

        oracle.n_quadruplets = len(oracle.comparisons.row)
        oracle_triplet = oracle.get_tSTE_comparisons()  # Already implemented... no need for what is bellow

        embedding = cy_tste.tste(oracle_triplet,
                                 no_dims=args["dims_embed"],
                                 lamb=0,
                                 alpha=None,
                                 verbose=False,
                                 max_iter=1000,
                                 save_each_iteration=False,
                                 initial_X=None,
                                 static_points=np.array([]),
                                 ignore_zeroindexed_error=True,
                                 num_threads=None,
                                 use_log=False)
        kmeans = KMeans(n_clusters=args["n_clusters"], n_init=args["numberIteration"],
                        random_state=args["seed"]).fit(embedding)
        classes_pred = kmeans.labels_
        if args["plot"]:
            for i in np.unique(classes_pred):
                plt.plot(embedding[classes_pred == i, 0], embedding[classes_pred == i, 1], "+")
            plt.savefig("./Barycenter/picture/triplet.pdf")
            plt.show()
            for i in np.unique(classes_pred):
                plt.plot(embedding[original_clusters == i, 0], embedding[original_clusters == i, 1], "+")
            plt.savefig("./Barycenter/picture/tripletGT.pdf")
            plt.show()

    elif algo == "OTT":
        oracle_triplet = oracle.get_AddS_comparisons()
        oracle_triplet_reshape = oracle_triplet.toarray().reshape((args["n_examples"],
                                                                   args["n_examples"],
                                                                   args["n_examples"]))
        p = [ot.unif(args["n_examples"])]
        q = [args["proportion"]]

        def loss_fun(C1, C2):
            return (C1 - C2) ** 2

        loss_fun = [loss_fun]
        best_ot_dist = np.inf
        for i in range(args["numberIteration"]):
            X2, T, ot_dist = gromov.barycenter_OTT_L2(p=p, q=q,
                                                      loss_fun=loss_fun,
                                                      X1=[oracle_triplet_reshape],
                                                      X2=None,
                                                      M=[None],
                                                      T_pos=[[0, 0, 0]],
                                                      T=[None],
                                                      alpha=[1],
                                                      nb_iter=args["nb_iter"],
                                                      nb_samples=args["nb_samples"],
                                                      epsilon=args["epsilon"],
                                                      KL=args["KL"],
                                                      L2=0,
                                                      labels_s=None,
                                                      eta=0,
                                                      verbose=False,
                                                      threshold=1e-20,
                                                      sample_t_only_init=False,
                                                      sample_t_init_and_iteration=False,
                                                      nb_loop_barycenter=args["nb_loop_barycenter"])
            if ot_dist < best_ot_dist:
                classes_pred = np.argmax(T[0], axis=1)
                best_ot_dist = ot_dist
                T_, X2_, ot_dist_ = T, X2, ot_dist
    else:
        assert False
    return classes_pred, T_, X2_, ot_dist_




def main(args):
    n_per_features = args["n_per_features"]
    labels_selected = args["labels_selected"]
    args["n_examples"] = sum(n_per_features)
    args["n_clusters"] = len(labels_selected)
    noise = args["noise"]
    for i in range(10):
        # if args["new_dataset"]:
        if args["features"] == "umap" or args["features"] == "initial":
            np.random.seed(args["seed"] + i)
            features, original_similarities, original_clusters = mnist_model(labels_selected=labels_selected,
                                                                             n_per_features=n_per_features,
                                                                             features=args["features"])

        else:
            assert False

        args["n_comparisons"] = int(args["n_examples"] * np.log(args["n_examples"]) ** args["power"])
        np.random.seed(args["seed"] + i)
        oracle = OracleTriplets(original_similarities,
                                args["n_examples"],
                                n_triplets=args["n_comparisons"],
                                proportion_noise=noise / 2, seed=args["seed"])

        dict_return = {"original_clusters": original_clusters}
        for d in range(len(args["algo"])):
            np.random.seed(args["seed"] + i)

            classes_pred, T, X2, ot_dist = pred(args["algo"][d], oracle, args, original_clusters=original_clusters)

            score_adds = adjusted_rand_score(classes_pred, original_clusters)

            dict_return[args["algo"][d]] = {"classes_pred": classes_pred,
                                            "score_adds": score_adds,
                                            # "score_adds2": score_adds2,
                                            "T": None,
                                            # "T": T,
                                            "X2": X2,
                                            "ot_dist": ot_dist,
                                            }
            print(args["algo"][d], np.mean(score_adds))  # , np.mean(score_adds2))
        if args["save"]:
            if args["umap_dim"] == 2:
                with open("./Barycenter/pickle/" + args["features"] + "_" + args["dataset"] + "_" + str(i) + "_" + args[
                    "pickle_name"] + ".pickle", "wb") as f:
                    pickle.dump(dict_return, f)
            else:
                with open("./Barycenter/pickle/umap" + str(args["umap_dim"]) + "/" + args["features"] + "_" + args["dataset"] + "_" + str(i) + "_" + args[
                    "pickle_name"] + ".pickle", "wb") as f:
                    pickle.dump(dict_return, f)


def display_basic_only_one(args):
    latex_tabular = {"Datasets": []}
    latex_tabular_ = {"dataset": []}
    # data = [[] for _ in range(len(args["algo"]))]
    no_keep_list = [args["features"] + x for x in ["_20,20,20,20,20,20,20,20,20,20|0,1,2,3,4,5,6,7,8,9_" + args["pickle_name"] + ".pickle",
                                                    "_40,4,8,6,28,37,3,29,18,15|0,1,2,3,4,5,6,7,8,9_" + args["pickle_name"] + ".pickle",
                                                    "_30,3,3|0,1,2_" + args["pickle_name"] + ".pickle"]]
                                                    # "_30,3,1|0,1,2_" + args["pickle_name"] + ".pickle"]]
    for file in sorted(os.listdir("./Barycenter/pickle/")):
        if file.endswith("_" + args["pickle_name"] + ".pickle") and file.startswith(args["features"]) and not (
                file in no_keep_list):
            print(file)
            with open("./Barycenter/pickle/" + file, "rb") as f:
                dict_return = pickle.load(f)
            a = file.split(".")[0][len(args["features"]) + 1:]
            # print(a)
            a = a.split("_" + args["pickle_name"])[0]
            # print(a)
            a = a[:min(15, len(a) + 1)]
            a.replace("|", "_")
            latex_tabular["Datasets"].append(a)
            best = []
            # print(dict_return)
            for d in range(len(args["algo"])):
                # print(args["algo"][d])
                # print(np.round(dict_return[args["algo"][d]]["score_adds"]))
                if not (args["algo_display"][d] in latex_tabular):
                    latex_tabular[args["algo_display"][d]] = []
                    latex_tabular_[args["algo_display"][d]] = []
                latex_tabular[args["algo_display"][d]].append(np.round(dict_return[args["algo"][d]]["score_adds"], 2))
                latex_tabular_[args["algo_display"][d]].append(dict_return[args["algo"][d]]["score_adds"])
                best.append(dict_return[args["algo"][d]]["score_adds"])
            pos_bests = np.argwhere(best == np.max(best)).flatten()  # np.argmax(best)
            for pos_best in pos_bests:
                latex_tabular[args["algo_display"][pos_best]][-1] = "\textbf{" + \
                                                                    str(latex_tabular[args["algo_display"][pos_best]][
                                                                            -1]) + "}"
    # print(latex_tabular)
    # for key in latex_tabular:
    #     print(key, len(latex_tabular[key]))
    best = []
    for d in range(len(args["algo"])):
        # print(args["algo_display"][d])
        latex_tabular[args["algo_display"][d]].append(np.round(np.mean(latex_tabular_[args["algo_display"][d]]), 2))
        best.append(np.mean(latex_tabular_[args["algo_display"][d]]))
    # pos_best = np.argmax(best)
    pos_bests = np.argwhere(best == np.max(best)).flatten()  # np.argmax(best)
    for pos_best in pos_bests:
        latex_tabular[args["algo_display"][pos_best]][-1] = "\textbf{" + \
                                                            str(latex_tabular[args["algo_display"][pos_best]][-1]) + "}"
    latex_tabular["Datasets"].append("AVG")

    df = pd.DataFrame(latex_tabular)
    if args["pickle_name"] == "":
        args["pickle_name"] = "digit"
    print("\\begin{table}")
    print("\centering")
    print("\\caption{ARI for the " + args["pickle_name"] + " dataset with " + args["features"] + " features.}")
    print("\\label{tab:bary" + args["features"] + args["pickle_name"] + "}")
    print(df.to_latex(index=False, escape=False))
    print("\\end{table}")


def display_basic2(args):
    latex_tabular = {"Datasets": []}
    latex_tabular_ = {"dataset": []}
    latex_tabular_std = {"dataset": []}
    # data = [[] for _ in range(len(args["algo"]))]
    no_keep_list = ["_20,20,20,20,20,20,20,20,20,20|0,1,2,3,4,5,6,7,8,9_",
                    "_40,4,8,6,28,37,3,29,18,15|0,1,2,3,4,5,6,7,8,9_",
                    # "_30,3,3|0,1,2_",
                    "200,20|",
                    "300,10|"]
    basic_file = []
    for file in sorted(os.listdir("./Barycenter/pickle/")):
        if file.endswith("_" + args["pickle_name"] + ".pickle") and file.startswith(args["features"]) and file.split("_")[-2].isdigit():# and len(file.split("_")) == 4:
            continue_ = False
            for line in no_keep_list:
                if line in file:
                    continue_= True
            if continue_:
                continue
            file_ = file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[-1]
            basic_file.append(file_)

    basic_file = list(set(basic_file))
    basic_file = sorted(basic_file)
    dict_return = {}

    for file in basic_file:
        for d in range(len(args["algo"])):
            dict_return[args["algo"][d]] = []
        for i in range(10):
            with open("./Barycenter/pickle/" + file.split("_")[0] + "_" + file.split("_")[1] + "_" + str(i) + "_" +
                      file.split("_")[-1], "rb") as f:
                dict_return_i = pickle.load(f)
                for d in range(len(args["algo"])):
                    dict_return[args["algo"][d]].append(dict_return_i[args["algo"][d]]["score_adds"])
        a = file.split(".")[0][len(args["features"]) + 1:]
        a = a.split("_" + args["pickle_name"])[0]
        a = a[:min(15, len(a) + 1)]
        a.replace("|", "_")
        latex_tabular["Datasets"].append(a)
        best = []
        for d in range(len(args["algo"])):
            if not (args["algo_display"][d] in latex_tabular):
                latex_tabular[args["algo_display"][d]] = []
                latex_tabular_[args["algo_display"][d]] = []
                latex_tabular_std[args["algo_display"][d]] = []
            latex_tabular[args["algo_display"][d]].append(
                    str(np.round(np.mean(dict_return[args["algo"][d]]), 2)) + "$\pm$" + str(np.round(np.std(dict_return[args["algo"][d]]), 2)))
            latex_tabular_[args["algo_display"][d]].append(np.mean(dict_return[args["algo"][d]]))
            latex_tabular_std[args["algo_display"][d]].append(np.std(dict_return[args["algo"][d]]))
            best.append(np.mean(dict_return[args["algo"][d]]))

        pos_bests = np.argwhere(best == np.max(best)).flatten()
        for pos_best in pos_bests:
            latex_tabular[args["algo_display"][pos_best]][-1] = "\textbf{" + \
                                                                str(np.round(np.mean(dict_return[args["algo"][pos_best]]), 2)) + \
                                                                "}" + "$\pm$" + str(np.round(np.std(dict_return[args["algo"][pos_best]]), 2))

    best = []
    for d in range(len(args["algo"])):
        latex_tabular[args["algo_display"][d]].append(
            str(np.round(np.mean(latex_tabular_[args["algo_display"][d]]), 2)) + "$\pm$" +
            str(np.round(np.mean(latex_tabular_std[args["algo_display"][d]]), 2)))
        best.append(np.mean(latex_tabular_[args["algo_display"][d]]))

    pos_bests = np.argwhere(best == np.max(best)).flatten()  # np.argmax(best)
    for pos_best in pos_bests:
        latex_tabular[args["algo_display"][pos_best]][-1] = "\textbf{" + str(np.round(np.mean(latex_tabular_[args["algo_display"][pos_best]]), 2)) + \
                                                        "}" + "$\pm$" + str(np.round(np.mean(latex_tabular_std[args["algo_display"][pos_best]]), 2))
    latex_tabular["Datasets"].append("AVG")

    df = pd.DataFrame(latex_tabular)
    if args["pickle_name"] == "":
        args["pickle_name"] = "digit"
    print("\\begin{table}")
    print("\centering")
    print("\\caption{ARI for the " + args["pickle_name"] + " dataset with " + args["features"] + " features.}")
    print("\\label{tab:bary" + args["features"] + args["pickle_name"] + "}")
    print(df.to_latex(index=False, escape=False))
    print("\\end{table}")



def display_basic(args):
    latex_tabular = {"Datasets": []}
    latex_tabular_ = {"dataset": []}
    latex_tabular_std = {"dataset": []}
    # data = [[] for _ in range(len(args["algo"]))]
    # no_keep_list = [args["features"] + x for x in ["_20,20,20,20,20,20,20,20,20,20|0,1,2,3,4,5,6,7,8,9_" + args["pickle_name"] + ".pickle",
    #                                                 "_40,4,8,6,28,37,3,29,18,15|0,1,2,3,4,5,6,7,8,9_" + args["pickle_name"] + ".pickle",
    #                                                 "_30,3,3|0,1,2_" + args["pickle_name"] + ".pickle"]]

    lines = args["dataset_plot"].split("/")
    basic_file = []
    if args["umap_dim"] != 2:
        umap_index = "umap" + str(args["umap_dim"]) +"/"
    else:
        umap_index = ""
    # umap_index = "umap" + str(args["umap_dim"]) + "/"
    # print(umap_index)
    for file in sorted(os.listdir("./Barycenter/pickle/"+umap_index)):
        if file.endswith("_" + args["pickle_name"] + ".pickle") and file.startswith(args["features"]) and file.split("_")[-2].isdigit():# and len(file.split("_")) == 4:
            keep = False
            for line in lines:
                if line in file:
                    keep = True
            if keep:
                file_ = file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[-1]
                basic_file.append(file_)

    basic_file = list(set(basic_file))
    basic_file = sorted(basic_file)
    dict_return, dict_return_std = {}, {}
    for line in lines:
        latex_tabular["Datasets"].append(line[:-1])
        for d in range(len(args["algo"])):
            dict_return[args["algo"][d]] = []
            dict_return_std[args["algo"][d]] = []

        # print(basic_file)
        for file in basic_file:
            if not line in file:
                continue
            # print(file)
            std_temp = []
            for d in range(len(args["algo"])):
                std_temp.append([])
            for i in range(10):
                try:
                    with open("./Barycenter/pickle/" + umap_index + file.split("_")[0] + "_" + file.split("_")[1] + "_" + str(i) + "_" +
                              file.split("_")[-1], "rb") as f:
                        dict_return_i = pickle.load(f)
                        for d in range(len(args["algo"])):
                            dict_return[args["algo"][d]].append(dict_return_i[args["algo"][d]]["score_adds"])
                            std_temp[d].append(dict_return_i[args["algo"][d]]["score_adds"])
                except:
                    continue
            for d in range(len(args["algo"])):
                dict_return_std[args["algo"][d]].append(np.std(std_temp[d]))
        for d in range(len(args["algo"])):
            dict_return_std[args["algo"][d]] = np.mean(dict_return_std[args["algo"][d]])
        best = []
        for d in range(len(args["algo"])):
            if not (args["algo_display"][d] in latex_tabular):
                latex_tabular[args["algo_display"][d]] = []
                latex_tabular_[args["algo_display"][d]] = []
                latex_tabular_std[args["algo_display"][d]] = []
            latex_tabular[args["algo_display"][d]].append(
                str(np.round(np.mean(dict_return[args["algo"][d]]), 2)) + "$\pm$" + str(np.round(dict_return_std[args["algo"][d]], 2)))
            latex_tabular_[args["algo_display"][d]].append(np.mean(dict_return[args["algo"][d]]))
            latex_tabular_std[args["algo_display"][d]].append(dict_return_std[args["algo"][d]])
            best.append(np.round(np.mean(dict_return[args["algo"][d]]), 2))

        pos_bests = np.argwhere(best == np.max(best)).flatten()  # np.argmax(best)
        for pos_best in pos_bests:
            latex_tabular[args["algo_display"][pos_best]][-1] = "\textbf{" + \
                                                                str(np.round(np.mean(dict_return[args["algo"][pos_best]]), 2)) + \
                                                                "}" + "$\pm$" + str(np.round(dict_return_std[args["algo"][d]], 2))

    best = []
    for d in range(len(args["algo"])):
        latex_tabular[args["algo_display"][d]].append(
            str(np.round(np.mean(latex_tabular_[args["algo_display"][d]]), 2)) + "$\pm$" +
            str(np.round(np.mean(latex_tabular_std[args["algo_display"][d]]), 2)))
        best.append(np.mean(latex_tabular_[args["algo_display"][d]]))

    pos_bests = np.argwhere(best == np.max(best)).flatten()  # np.argmax(best)
    for pos_best in pos_bests:
        latex_tabular[args["algo_display"][pos_best]][-1] = "\textbf{" + str(np.round(np.mean(latex_tabular_[args["algo_display"][pos_best]]), 2)) + \
                                                        "}" + "$\pm$" + str(np.round(np.mean(latex_tabular_std[args["algo_display"][pos_best]]), 2))
    latex_tabular["Datasets"].append("AVG")
    # for d in range(len(args["algo"])):
    #     # print("SUMMM", A[d] / (len(latex_tabular["Datasets"]) - 1))
    #     print(np.mean(latex_tabular_[args["algo_display"][d]]))
    #     print(latex_tabular_[args["algo_display"][d]])


    df = pd.DataFrame(latex_tabular)
    if args["pickle_name"] == "":
        args["pickle_name"] = "digit"
    print("\\begin{table}")
    print("\centering")
    print("\\caption{ARI for the " + args["pickle_name"] + " dataset with " + args["features"] + " features.}")
    print("\\label{tab:bary" + args["features"] + args["pickle_name"] + "}")
    print(df.to_latex(index=False, escape=False))
    print("\\end{table}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Barycenter')
    parser.add_argument('--n_per_features', type=str, default="10,10,10")
    parser.add_argument('--proportion', type=str, default="None")
    parser.add_argument('--labels_selected', type=str, default="None")
    parser.add_argument('--algo', type=str, default="SPUR,SPUR_supervised,SDP,SDP_supervised,tste,tste_supervised,OTT")
    parser.add_argument('--algo_display', type=str,
                        default="\AddSpredcluster{},\AddSpredcluster{}$_{s}$,\AddS{},\AddS{}$_{s}$,t-STE,t-STE$_{s}$,OTT")
    parser.add_argument('--pickle_name', type=str, default="mnist")
    parser.add_argument('--features', type=str, default="umap")
    parser.add_argument('--dataset_plot', type=str, default="")
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--power', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nb_iter', type=int, default="500")
    parser.add_argument('--nb_samples', type=int, default="100")
    parser.add_argument('--epsilon', type=float, default="0.1")
    parser.add_argument('--KL', type=int, default="1")
    parser.add_argument('--nb_loop_barycenter', type=int, default="20")
    parser.add_argument('--numberIteration', type=int, default="10")
    parser.add_argument('--special_exp', type=str, default="None")
    parser.add_argument('--dims_embed', type=int, default=2)
    # parser.add_argument('--n_per_features', type=str, default="10,10")
    parser.add_argument("-n", "--new_dataset", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-d", "--display", action="store_true")
    parser.add_argument("--umap_dim", type=int, default="2")
    args = parser.parse_args()
    args = vars(args)
    args["algo"] = args["algo"].split(",")
    args["algo_display"] = args["algo_display"].split(",")
    args["dataset"] = args["n_per_features"]
    args["n_per_features"] = [int(x) for x in args["n_per_features"].split(",")]
    if args["labels_selected"] is "None":
        args["labels_selected"] = [int(i) for i in range(len(args["n_per_features"]))]
        args["dataset"] += "|0"
        for i in range(1, len(args["n_per_features"])):
            args["dataset"] += "," + str(i)
    else:
        args["dataset"] += "|" + args["labels_selected"]
        args["labels_selected"] = [int(x) for x in args["labels_selected"].split(",")]
    args["nb_samples"] = [args["nb_samples"]]
    if args["proportion"] == "None":
        # args["dataset"] += "|uniform"
        args["proportion"] = np.array(args["n_per_features"]) / np.sum(args["n_per_features"])
    else:
        # args["dataset"] += "|"
        args["proportion"] = np.array([float(x) for x in args["proportion"].split(",")])
        args["proportion"] = args["proportion"] / np.sum(args["proportion"])
    if not args["display"]:
        print("./Barycenter/pickle/" + args["features"] + "_" + args["dataset"] + "_" + args["pickle_name"] + ".pickle")
    # if args["special_exp"] == "prop":
    #     find_proportion(args)
    # else:
    if args["display"]:
        if args["dataset_plot"] == "":
            display_basic2(args)
        else:
            display_basic(args)
    else:
        main(args)
