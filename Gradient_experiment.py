import matplotlib.pyplot as plt

import HDA
import ot
import numpy as np
import gromov
import time
import argparse
import pickle
import sklearn.neighbors
import itertools
import tikzplotlib



def find_T(nb_iter, dataset_path="./gradient_experiment/dataset/", nb_samples=200):
    param = {"alpha": [1], "nb_iter": nb_iter, "nb_samples": [nb_samples], "epsilon": 1}
    with open(dataset_path, "rb") as handle:
        dict_dataset = pickle.load(handle)
    C1, C2, p, q = dict_dataset["C1"], dict_dataset["C2"], dict_dataset["p"], dict_dataset["q"]
    # print(np.allclose(C1, C1.T, rtol=1e-10, atol=1e-10) and np.allclose(C2, C2.T, rtol=1e-10, atol=1e-10))
    param["p"], param["q"] = [p], [q]

    def f(C1, C2):
        return (C1 - C2) ** 2

    loss_fun = [f]
    param["loss_fun"] = loss_fun
    def make_f(sx):
        def f(*x):
            return sx[x]

        return f

    Sx_ = [make_f(C1)]
    Tx_ = [make_f(C2)]
    T_ = [p[:, np.newaxis] * q[np.newaxis, :]]
    T_pos = [[0] * len(C1.shape)]

    T, log = gromov.CO_Generalisation_OT(p=[p], q=[q],
                                         loss_fun=loss_fun,
                                         X1=Sx_, X2=Tx_,
                                         T=T_,
                                         M=[None],
                                         T_pos=T_pos,
                                         alpha=param["alpha"],
                                         nb_iter=param["nb_iter"],
                                         nb_samples=param["nb_samples"],
                                         epsilon=param["epsilon"],
                                         KL=1,
                                         threshold=1e-15,
                                         log=True,
                                         sparse_T=False)

    return T, param, T_pos, Sx_, Tx_


def diff_grad(T, param, T_pos, X1, X2, nb_samples_list):
    p, q, loss_fun = param["p"], param["q"], param["loss_fun"]
    a = 0
    grad_index = 0
    grad = [[] for _ in range(len(nb_samples_list))]
    time_save = []
    for nb_samples in nb_samples_list:
        print("")
        print(nb_samples)
        t_optimized = 0
        pos_with_t_optimized = [i for i, x in enumerate(T_pos[a]) if x == t_optimized]
        pos_with_t_optimized_kept, nb_sample_same_T = np.unique(
            np.random.choice(pos_with_t_optimized, size=nb_samples),
            return_counts=True)
        time_init = time.time()
        for r in range(10):
            print(".", end="")
            L = 0
            for b in range(len(pos_with_t_optimized_kept)):
                index_i = [0] * len(T_pos[a])
                index_k = [0] * len(T_pos[a])
                for t in range(len(T_pos[a])):
                    if t == pos_with_t_optimized_kept[b]:
                        index_i[t], index_k[t] = np.s_[:], np.s_[:]
                    else:
                        index_i[t] = np.random.choice(len(p[T_pos[a][t]]),
                                                      size=nb_sample_same_T[b],
                                                      p=p[T_pos[a][t]])
                        index_k[t] = np.zeros(nb_sample_same_T[b], dtype=int)
                        for i in range(nb_sample_same_T[b]):
                            index_k[t][i] = np.random.choice(len(q[T_pos[a][t]]),
                                                             size=1,
                                                             p=T[T_pos[a][t]][index_i[t][i], :] /
                                                               p[T_pos[a][t]][index_i[t][i]])[0]
                if pos_with_t_optimized_kept[b] == 0:
                    L += loss_fun[a](np.expand_dims(X1[a](*index_i).T, 2),
                                     np.expand_dims(X2[a](*index_k).T, 1)
                                     ).sum(axis=0) * param["alpha"][a]

                else:
                    L += loss_fun[a](np.expand_dims(X1[a](*index_i), 2),
                                     np.expand_dims(X2[a](*index_k), 1)
                                     ).sum(axis=0) * param["alpha"][a]
            L = L / nb_samples
            grad[grad_index].append(L)
        time_save.append((time.time() - time_init) / 10)
        grad_index += 1
    return grad, time_save


def make_datasets(dataset_path, cluster, dimension=10, D=(2, 5), std=0.15, rdm_seed=42, nb_dataset=2):
    N = np.sum(cluster)

    distance, closest = [0] * nb_dataset, [0] * nb_dataset
    pq = [np.random.rand(N), np.random.rand(N)]
    for nb_dataset_i in range(nb_dataset):
        np.random.seed(rdm_seed + nb_dataset_i * 123)
        for c in range(len(cluster)):
            mean = np.random.rand(dimension)
            cov = np.diag(np.random.rand(dimension) * std)
            correlation = np.random.rand(dimension * dimension).reshape(dimension, dimension) * std / 2
            correlation[np.arange(dimension), np.arange(dimension)] = 0
            correlation = (correlation + correlation.T) / 2
            correlation = np.eye(dimension) + correlation
            cov = cov @ correlation @ cov
            if c == 0:
                gaussians = np.random.multivariate_normal(mean, cov, cluster[c])
            else:
                gaussians = np.concatenate((gaussians, np.random.multivariate_normal(mean, cov, cluster[c])))
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=D[1] + 2)
        neigh.fit(gaussians)
        distance[nb_dataset_i], closest[nb_dataset_i] = neigh.kneighbors(gaussians)

    for d in range(D[0], D[1]):
        print(d)
        dict_dataset = {}
        for nb_dataset_i in range(nb_dataset):
            distance_d, closest_d = distance[nb_dataset_i][:, :d], closest[nb_dataset_i][:, :d]
            size_tensor = "N," * d
            size_tensor = size_tensor[:-1]
            # TODO: This is VERY ugly
            C = eval("np.zeros((" + size_tensor + "))")

            # TODO: and also this
            def f(value, matrix, *x):
                # print(x)
                matrix[x] += value
                return matrix

            for n in range(N):
                for perm in list(itertools.permutations(closest_d[n, :])):
                    C = f(1, C, *perm)
                    # save_value = closest_d[n, -1]
                    # closest_d[n, 1:] = closest_d[n, :-1]
                    # closest_d[n, 0] = save_value
            dict_dataset["C" + str(nb_dataset_i + 1)] = C

        dict_dataset["cluster"] = cluster
        dict_dataset["D"] = D
        dict_dataset["std"] = std
        dict_dataset["rdm_seed"] = rdm_seed
        dict_dataset["p"] = pq[0] / np.sum(pq[0])
        dict_dataset["q"] = pq[1] / np.sum(pq[1])
        dataset_name = "hypergraphs_" + str(d) + "_" + str(N)
        if args["save"]:
            with open(dataset_path + dataset_name + ".pickle", "wb") as handle:
                pickle.dump(dict_dataset, handle)


def store_gradient(path_pickle, out_pickle, nb_point, D, outer_iteration, nb_samples_list):
    for i in outer_iteration:
        print("outer iteration:", i)
        np.random.rand(1234 + i * 123)
        T, param, T_pos, C1, C2 = find_T(nb_iter=i,
                                         dataset_path=path_pickle + "hypergraphs_" + str(D) + "_" + str(nb_point) +
                                                      ".pickle",
                                         nb_samples=200)
        grad, time_save = diff_grad(T=T, param=param, T_pos=T_pos, X1=C1, X2=C2, nb_samples_list=nb_samples_list)
        dict_grad_time = {"grad": grad, "time_save": time_save}
        if args["save"]:
            with open(out_pickle + "_" + str(i) + "_" + str(D) + "_" + str(nb_point) + ".pickle", "wb") as handle:
                pickle.dump(dict_grad_time, handle)


# def full_grad(pickle_path):
#     # T, param, T_pos, X1, X2
#     p, q, loss_fun = param["p"], param["q"], param["loss_fun"]
#     grad = []
#     a = 0
#     L = 0
#     t_optimized = 1
#     pos_with_t_optimized = [i for i, x in enumerate(T_pos[a]) if x == t_optimized]
#     #     print(pos_with_t_optimized)
#     G = 0
#     for b in pos_with_t_optimized:
#         if b == 1:
#             for i in range(len(p[T_pos[a][0]])):
#                 print(".", end="")
#                 for k in range(len(q[T_pos[a][2]])):
#                     for l in range(len(p[T_pos[a][0]])):
#                         for n in range(len(q[T_pos[a][2]])):
#                             G += loss_fun[a](np.expand_dims(X1[a](i, np.s_[:], k), 0),
#                                              np.expand_dims(X2[a](l, np.s_[:], n), 1)) \
#                                  * T[T_pos[a][0]][i, l] \
#                                  * T[T_pos[a][2]][k, n]
#         #                             print(G.shape)
#
#         elif b == 2:
#             for i in range(len(p[T_pos[a][0]])):
#                 print(".", end="")
#                 for j in range(len(q[T_pos[a][1]])):
#                     for l in range(len(p[T_pos[a][0]])):
#                         for m in range(len(q[T_pos[a][1]])):
#                             G += loss_fun[a](np.expand_dims(X1[a](i, j, np.s_[:]), 0),
#                                              np.expand_dims(X2[a](l, m, np.s_[:]), 1)) \
#                                  * T[T_pos[a][1]][i, l] \
#                                  * T[T_pos[a][2]][j, m]
#         else:
#             pass
#     return G


def display_grad(pickle_path, D, outer_iteration, nb_point, nb_samples_list):
    nb_samples_list = np.array(nb_samples_list)
    marker = ["", "", "|", "^", "s"]
    color = ["", "", "r", "b", "g"]
    for outer_iteration_i in outer_iteration:
        plt.figure(outer_iteration_i)
        for d in range(D[0], D[1]):
            if outer_iteration_i == outer_iteration[0]:
                print("nb element in the tensor L of size N^d" + str(d) + " : ", nb_point ** (2*(d-1)))
            with open(pickle_path + "_" + str(outer_iteration_i) + "_" + str(d) + "_" + str(nb_point) +
                    ".pickle", "rb") as handle:
                dict_dataset = pickle.load(handle)
            grad, time = dict_dataset["grad"], dict_dataset["time_save"]

            best_grad_approx = 0
            for r in range(10):
                best_grad_approx += grad[-1][r]
            best_grad_approx = best_grad_approx / 10
            list_mean_diff, list_std_diff = np.zeros(len(grad)), np.zeros(len(grad))
            keep_diff_zero = []
            for i in range(len(grad)):
                diff = []
                for r in range(10):
                    diff.append(np.linalg.norm(grad[i][r] - best_grad_approx)/np.linalg.norm(best_grad_approx))
                list_mean_diff[i] = np.mean(diff)
                list_std_diff[i] = np.std(diff)
                # print(nb_samples_list[i], " |Time: ", time[i],
                #       " |Diff: ", np.mean(diff), np.std(diff))
                if list_mean_diff[i] != 1:
                    keep_diff_zero.append(i)
            # print(keep_diff_zero)
            keep_diff_zero = np.array(keep_diff_zero, dtype=int)
            plt.plot(nb_samples_list[keep_diff_zero], list_mean_diff[keep_diff_zero],
                     # color=color[d],
                     label=str(d) + "-order tensor", marker=marker[d], markersize=6)
            min_std_list = list_mean_diff - list_std_diff
            # min_std_list[min_std_list < 10**-2] = list_mean_diff[min_std_list < 10**-2]
            plt.fill_between(nb_samples_list[keep_diff_zero],
                             list_mean_diff[keep_diff_zero] + list_std_diff[keep_diff_zero],
                             min_std_list[keep_diff_zero],
                             alpha=0.2)
            # plt.errorbar(nb_samples_list, list_mean_diff, yerr=list_std_diff,
            #              color=color[d],
            #              label='Standard deviation error bar')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of samples $M$")
        plt.ylabel("$\\frac{\\ norm{\\nabla \\mathcal{E} - \\widehat{\\nabla \\mathcal{E}}}_F}{\\ norm{\\nabla \\mathcal{E}}_F}$")
        plt.legend(loc=0, fontsize=10)
        tikzplotlib.save("./gradient_experiment/fig/" + str(nb_point) + "/fig_" + str(outer_iteration_i) + ".tex")
        if args["save"]:
            plt.savefig("./gradient_experiment/fig/" + str(nb_point) + "/fig_" + str(outer_iteration_i) + ".pdf")
        plt.show()

        # auie


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OTT')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cluster', type=str, default="10,20,25")
    parser.add_argument('--outer_iteration', type=str, default="5,10,20,50,100,200,500,1000")
    parser.add_argument('--nb_samples', type=str, default="1,5,10,50,100,500,1000,5000,10000,50000,100000")
    parser.add_argument('-p', '--plot', action="store_true")
    parser.add_argument('-d', '--dataset', action="store_true")
    parser.add_argument('-s', '--save', action="store_true")
    parser.add_argument('--D', type=str, default="2,5")
    parser.add_argument('--pickle_path', type=str,
                        default="./gradient_experiment/pickle/")
    # nb_samples_list = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]  # ,50000]#,100000] #,100000]:

    args = parser.parse_args()
    args = vars(args)
    if "," in args["D"]:
        args["D"] = [int(x) for x in args["D"].split(",")]
    else:
        args["D"] = (2, int(args["D"]))
    np.random.seed(args["seed"])
    args["outer_iteration"] = [int(x) for x in args["outer_iteration"].split(",")]
    args["nb_samples"] = [int(x) for x in args["nb_samples"].split(",")]
    cluster = [int(x) for x in args["cluster"].split(",")]
    if args["dataset"]:
        make_datasets(dataset_path="./gradient_experiment/dataset/",
                      cluster=cluster,
                      dimension=10,
                      D=args["D"],
                      std=0.15,
                      rdm_seed=42)
    elif args["plot"]:
        display_grad(args["pickle_path"],
                     outer_iteration=args["outer_iteration"],
                     D=args["D"],
                     nb_point=sum(cluster),
                     nb_samples_list=args["nb_samples"])
    else:
        store_gradient(path_pickle="./gradient_experiment/dataset/",
                       out_pickle="./gradient_experiment/pickle/",
                       nb_point=sum(cluster),
                       D=args["D"][1],
                       outer_iteration=args["outer_iteration"],
                       nb_samples_list=args["nb_samples"])
