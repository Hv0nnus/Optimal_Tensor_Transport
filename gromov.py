# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein transport method
"""

# Code taken from Authors: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
# Modified for this paper.
# License: MIT License

import numpy as np

from ot.bregman import sinkhorn
# from ot.da import sinkhorn_l1l2_gl
from ot.da import sinkhorn_lpl1_mm
# from ot.utils import dist, UndefinedParameter
# from ot.optim import cg
from ot.lp import emd
# from numpy.random import default_rng
# from scipy.stats import rv_discrete
import ot
# import ot.sliced
import time
import scipy.sparse
# import torch

import warnings

warnings.filterwarnings("error")


def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
    if loss_fun == 'square_loss':
        def f1(a):
            return (a ** 2)

        def f2(b):
            return (b ** 2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):
    A = -np.dot(hC1, T).dot(hC2.T)
    tens = constC + A
    # tens -= tens.min()
    return tens


def gwloss(constC, hC1, hC2, T):
    tens = tensor_product(constC, hC1, hC2, T)

    return np.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):
    return 2 * tensor_product(constC, hC1, hC2,
                              T)  # [12] Prop. 2 misses a 2 factor


def update_square_loss(p, lambdas, T, Cs):
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_kl_loss(p, lambdas, T, Cs):
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.exp(np.divide(tmpsum, ppt))


def GW_init_T(N, K, T_is_sparse=False):
    if T_is_sparse:
        T = np.arange(K)
        T = np.tile(T, int(N / K) + 1)
        T = T[:N]
    else:
        T = np.ones((N, K)) / (N * K)
    if N == K:
        if T_is_sparse:
            T = np.arange(N)
        else:
            T = np.eye(N)
    return T


def compute_distance_sparse(C1, C2, loss_fun, T, dim_T):
    if isinstance(C1, np.ndarray):
        s = np.sum(loss_fun(np.squeeze(C1[np.ix_(T[0], T[0])]),
                            np.squeeze(C2[np.ix_(T[1], T[1])]))
                   * T[2][:, np.newaxis] * T[2][np.newaxis, :])
    else:
        s = np.sum(loss_fun(np.squeeze(C1([np.arange(dim_T[0])]))[np.ix_(T[0], T[0])],
                            np.squeeze(C2([np.arange(dim_T[1])]))[np.ix_(T[1], T[1])])
                   * T[2][:, np.newaxis] * T[2][np.newaxis, :])

    return s / np.sum(T[2])


def compute_distance_sampling_both(C1, C2, loss_fun, T, number_sample=None, std=False, std_total=False):
    if T.shape[0] < T.shape[1]:
        raise Exception("T.shape[0] should be higher than T.shape[1].")
    if number_sample is None:
        if T.shape[0] == T.shape[1]:
            number_sample = 1  # T.shape[0]
        else:
            number_sample = 2  # 2 * max(T.shape[0], T.shape[1])
    if std:
        number_sample = max(2, number_sample)

    index_k = np.zeros((T.shape[0], T.shape[0], number_sample), dtype=int)
    index_l = np.zeros((T.shape[0], T.shape[0], number_sample), dtype=int)
    list_value_sample = np.zeros((T.shape[0], T.shape[0], number_sample))

    # TODO can we skip this loop ? ... not sure.
    for i in range(T.shape[0]):
        index_k[i] = np.random.choice(T.shape[1], size=(T.shape[0], number_sample), p=T[i, :] * T.shape[0])
        index_l[i] = np.random.choice(T.shape[1], size=(T.shape[0], number_sample), p=T[i, :] * T.shape[0])
    if isinstance(C1, np.ndarray):
        for n in range(number_sample):
            list_value_sample[:, :, n] = loss_fun(C1,
                                                  (C2[index_k[:, :, n].reshape(-1),
                                                      index_l[:, :, n].T.reshape(-1)]).reshape(T.shape[0], T.shape[0]))
    else:
        for n in range(number_sample):
            list_value_sample[:, :, n] = loss_fun(C1([np.arange(T.shape[0])]).reshape(T.shape[0], T.shape[0]),
                                                  C2([index_k[:, :, n].reshape(-1)],
                                                     index_l[:, :, n].T.reshape(-1)).reshape(T.shape[0], T.shape[0]))
    if std:
        std_value = np.sum(np.std(list_value_sample, axis=2) ** 2) ** 0.5
        print(std_value / (T.shape[0] * T.shape[0]))
        if std_total:
            return np.mean(list_value_sample), std_value / (T.shape[0] * T.shape[0]), np.std(list_value_sample)
        else:
            return np.mean(list_value_sample), std_value / (T.shape[0] * T.shape[0])
    else:
        return np.mean(list_value_sample)


def compute_distance(T, C1, C2, loss):
    # TODO to be optimised
    T = T / np.sum(T)
    s = 0
    if isinstance(C1, np.ndarray):
        for i in range(T.shape[0]):
            for k in range(T.shape[1]):
                s = s + np.sum(loss(C1[i, np.newaxis, :, np.newaxis],
                                    C2[k, np.newaxis, np.newaxis, :]), axis=0) * T[i, k]
    else:
        for i in range(T.shape[0]):
            for k in range(T.shape[1]):
                s = s + np.sum(loss(C1(np.array([[i]])),
                                    C2(np.array([[k]]))), axis=0) * T[i, k]

    return np.sum(s * T)


def compute_L(C1, C2, loss_fun, T):
    if np.max(T.shape) > 100:
        s = 0
        for i in range(T.shape[0]):
            for k in range(T.shape[1]):
                s = s + np.sum(loss_fun(C1[i, np.newaxis, :, np.newaxis],
                                        C2[k, np.newaxis, np.newaxis, :]), axis=0) * T[i, k]
        return s
    else:
        return np.sum(loss_fun(C1[:, :, np.newaxis, np.newaxis], C2[np.newaxis, np.newaxis, :, :])
                      * T[:, np.newaxis, :, np.newaxis], axis=(0, 2))


def compute_distance_sampling_COOT_generalised(p, q,
                                               loss_fun,
                                               X1, X2,
                                               T,
                                               alpha,
                                               T_pos,
                                               M=[None, None],
                                               nb_samples=None):
    time_init = time.time()
    if nb_samples is None:
        maximum = 0
        for t in T:
            maximum = max(np.max(t.shape), maximum)
        nb_samples = [maximum ** len(T_pos[a]) for a in range(len(T_pos))]
    nb_samples = np.array(nb_samples)

    GW_dist = 0
    for a in range(len(T_pos)):
        if len(T_pos[a]) == 1:
            if M[a] is None:
                GW_dist += (T[T_pos[a][0]] * loss_fun[a](X1[a](np.s_[:]), X2[a](np.s_[:])) * alpha[a]).sum()
            else:
                GW_dist += (M[a] * T[T_pos[a][0]]).sum() * alpha[a]
        dimension_OT = len(alpha)
        index_i = np.zeros((len(T_pos[a]), nb_samples[a]), dtype=int)
        index_k = np.zeros(index_i.shape, dtype=int)
        for t in range(len(T_pos[a])):
            t_to_optimized = T_pos[a][t]
            index_i[t] = np.random.choice(len(p[t_to_optimized]), size=nb_samples[a], p=p[t_to_optimized])

            unique, count = np.unique(index_i[t], return_counts=True)
            for i in range(len(unique)):
                index_k[t, np.where(index_i[t] == unique[i])] = np.random.choice(len(q[t_to_optimized]),
                                                                                 size=count[i],
                                                                                 p=T[t_to_optimized][unique[i], :] /
                                                                                   p[t_to_optimized][unique[i]])

        GW_dist += alpha[a] * loss_fun[a](X1[a](*index_i), X2[a](*index_k)).mean()

        return GW_dist


def compute_distance_sampling_COOT_generalised_full(p, q,
                                                    loss_fun,
                                                    X1, X2,
                                                    T,
                                                    alpha,
                                                    T_pos,
                                                    M=[None, None]):
    # Return only the non constant part for the L2 loss

    GW_dist = 0
    for a in range(len(T_pos)):
        if len(T_pos[a]) == 1:
            if M[a] is None:
                GW_dist += (T[T_pos[a][0]] * loss_fun[a](X1[a](np.s_[:]), X2[a](np.s_[:])) * alpha[a]).sum()
            else:
                GW_dist += (M[a] * T[T_pos[a][0]]).sum() * alpha[a]
        elif len(T_pos[a]) == 3:
            all_index = tuple([np.s_[:]] * len(T_pos[a]))
            GW_dist += alpha[a] * np.einsum("ikm,i,k,m", X1[a](*all_index) ** 2,
                                            p[T_pos[a][0]], p[T_pos[a][1]], p[T_pos[a][2]])
            GW_dist += alpha[a] * np.einsum("jln,j,l,n", X2[a](*all_index) ** 2,
                                            q[T_pos[a][0]], q[T_pos[a][1]], q[T_pos[a][2]])

            GW_dist += -2 * alpha[a] * np.einsum("ikm,jln,ij,kl,mn", X1[a](*all_index), X2[a](*all_index),
                                                 T[T_pos[a][0]], T[T_pos[a][1]], T[T_pos[a][2]])
        else:
            assert False  # Not implemented
        return GW_dist


def OTT(p, q,
        loss_fun,
        X1, X2,
        M=[None, None],
        T_pos=[[0], [0, 0]],
        T=[None],
        alpha=[0.5, 0.5],
        nb_iter=500,
        nb_samples=[1, 1],
        nb_samples_t=None,
        epsilon=0,
        KL=0,
        L2=0,
        labels_s=None,
        labels_t=None,
        eta=1,
        log=False,
        sliced=False,
        learning_step=1,
        verbose=False,
        sparse_T=True,
        threshold=1e-20,
        time_print=False,
        sample_t_only_init=False,
        sample_t_init_and_iteration=False):
    """
    This algorithm is more general than OTT, it can solve some kind of Fused-OTT: severals OTT problems together.
    [OTT first problem, OTT second problem,...]
    Args:
        p: Lists of marginals associated with the first dimension of the transport plans
        q: Lists of marginals associated with the second dimension of the transport plans
        loss_fun: List of function that compare elements of the two tensors list of tensors
        X1: List of first tensor
        X2: List of second tensor
        M: List of Squared matrix = loss_fun(X1,X2) if the problem is the usual OT problem, None if this is another OTT problem.
        T_pos: List of transporst plan that sould be used for each OTT problem.
        [[0], [1,2], [0,0,1]] for example, correspond to 3 OTT problems, the first one ([0]) is a simple OT problem associated with the first transport plan
        The second ([1,2]) correspond to a Co-Ot problem with 2 differents OT plans, also differentes from the first one.
        The third ([0,0,1]) list correspond to a OTT problems with two times the first transport plan and the .
        T: List of initials transport plans.
        alpha:
        nb_iter:
        nb_samples:
        nb_samples_t:
        epsilon:
        KL:
        L2:
        labels_s:
        labels_t:
        eta:
        log:
        sliced:
        learning_step:
        verbose:
        sparse_T:
        threshold:
        time_print:
        sample_t_only_init:
        sample_t_init_and_iteration:

    Returns: An approximation of the real Transport Plan of the OTT problem.

    """
    time_init = time.time()
    alpha = np.array(alpha)
    nb_samples = np.array(nb_samples)

    if sample_t_only_init or sample_t_init_and_iteration:
        init_DA = np.ones((len(T_pos), len(T)), dtype=int)
    else:
        init_DA = np.zeros((len(T_pos), len(T)), dtype=int)

    for t in range(len(T)):
        p[t] = np.asarray(p[t], dtype=np.float64)
        q[t] = np.asarray(q[t], dtype=np.float64)
        if labels_s is not None and labels_s[t] is not None:
            labels_s[t] = np.array(labels_s[t], dtype=np.int)
        if labels_t is not None and labels_s[t] is not None:
            labels_t[t] = np.array(labels_t[t], dtype=np.int)

        if sliced:
            assert False  # sliced not implemented
        else:
            if T[t] is None:
                T[t] = np.outer(p[t], q[t])
    continue_loop = 0
    assert len(alpha) == len(X1) == len(X2) == len(loss_fun) == len(nb_samples)

    for iter in range(nb_iter):
        if verbose:
            if iter % 100 == 0:
                print(".", end="")
        if time_print:
            print("Before sample   ", time.time() - time_init)

        t_optimized = iter % len(T)

        if time_print:
            print("Before sum", time.time() - time_init)

        L = 0

        for a in range(len(T_pos)):

            pos_with_t_optimized = [i for i, x in enumerate(T_pos[a]) if x == t_optimized]
            if len(pos_with_t_optimized) == 0:
                # L += 0
                pass
            elif len(T_pos[a]) == 1:
                if M[a] is None:
                    M[a] = loss_fun[a](X1[a](np.s_[:]), X2[a](np.s_[:])) * alpha[a]
                L += M[a]
            elif len(pos_with_t_optimized) >= 1:

                if nb_samples_t is not None and nb_samples_t[a] is not None:

                    pos_with_t_optimized_kept, nb_sample_same_T = np.unique(
                        np.random.choice(pos_with_t_optimized, size=nb_samples_t[a]),
                        return_counts=True)

                    for b in range(len(pos_with_t_optimized_kept)):
                        index_i = [0] * len(T_pos[a])
                        index_k = [0] * len(T_pos[a])
                        for t in range(len(T_pos[a])):
                            pos_t_known = labels_t[T_pos[a][t]] != -1
                            if t == pos_with_t_optimized_kept[b]:
                                index_i[t], index_k[t] = np.s_[:], np.s_[:]
                            elif labels_t[T_pos[a][t]] is None or pos_t_known.sum() == 0 or (
                                    sample_t_only_init and not init_DA[a, t_optimized]):
                                index_i[t] = np.random.choice(len(p[T_pos[a][t]]),
                                                              size=nb_sample_same_T[b],
                                                              p=p[T_pos[a][t]])
                                index_k[t] = np.zeros(nb_sample_same_T[b], dtype=int)
                                for i in range(nb_sample_same_T[b]):
                                    index_k[t][i] = np.random.choice(len(q[T_pos[a][t]]),
                                                                     size=1,
                                                                     p=T[T_pos[a][t]][index_i[t][i], :] /
                                                                       p[T_pos[a][t]][index_i[t][i]])[0]
                            else:

                                index_k[t] = np.random.choice(
                                    np.where(pos_t_known)[0],
                                    size=nb_sample_same_T[b],
                                    p=q[T_pos[a][t]][pos_t_known] /
                                      np.sum(q[T_pos[a][t]][pos_t_known]))

                                index_i[t] = np.zeros(nb_sample_same_T[b], dtype=int)
                                for i in range(nb_sample_same_T[b]):
                                    same_label_index = labels_s[T_pos[a][t]] == labels_t[T_pos[a][t]][index_k[t][i]]
                                    p_ = T[T_pos[a][t]][same_label_index, index_k[t][i]]
                                    if p_.sum() == 0:
                                        p_ = ot.unif(len(p_))
                                    else:
                                        p_ = p_ / p_.sum()
                                    index_i[t][i] = np.random.choice(
                                        np.arange(len(p[T_pos[a][t]]))[same_label_index],
                                        size=1,
                                        p=p_)[0]

                        if pos_with_t_optimized_kept[b] == 0:
                            L += loss_fun[a](np.expand_dims(X1[a](*index_i).T, 2),
                                             np.expand_dims(X2[a](*index_k).T, 1)
                                             ).sum(axis=0) * alpha[a]
                        else:
                            L += loss_fun[a](np.expand_dims(X1[a](*index_i), 2),
                                             np.expand_dims(X2[a](*index_k), 1)
                                             ).sum(axis=0) * alpha[a]

                    # memory_L = L.sum()

                    if init_DA[a, t_optimized]:
                        init_DA[a, t_optimized] = False
                        continue

                pos_with_t_optimized_kept, nb_sample_same_T = np.unique(
                    np.random.choice(pos_with_t_optimized, size=nb_samples[a]),
                    return_counts=True)

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
                                         ).sum(axis=0) * alpha[a]

                    else:
                        L += loss_fun[a](np.expand_dims(X1[a](*index_i), 2),
                                         np.expand_dims(X2[a](*index_k), 1)
                                         ).sum(axis=0) * alpha[a]

            max_L = np.max(L)
            if max_L == 0:
                continue
            L = L / max_L

            if epsilon * KL > 0:
                log_T = np.log(np.clip(T[t_optimized], np.exp(-200), 1))
                log_T[log_T == -200] = -np.inf
                L = L - epsilon * KL * log_T
            if epsilon * L2 > 0:
                L = L - epsilon * L2 * 2 * T[t_optimized]

            if time_print:
                print("Before OT   ", time.time() - time_init)

            if epsilon == 0:
                if labels_s is not None and labels_s[t_optimized] is not None and labels_t is not None and labels_t[
                    t_optimized] is not None:
                    for label in np.unique(labels_t[t_optimized]):
                        if label == -1:
                            continue
                        L[np.ix_(labels_s[t_optimized] != label, labels_t[t_optimized] == label)] = 1000

                new_T = emd(a=p[t_optimized],
                            b=q[t_optimized],
                            M=L)
            else:
                if labels_s is None or labels_s[t_optimized] is None:
                    if L2 * epsilon > 0:
                        try:
                            new_T = ot.smooth.smooth_ot_semi_dual(a=p[t_optimized],
                                                                  b=q[t_optimized],
                                                                  M=L,
                                                                  reg=epsilon,
                                                                  reg_type='l2')
                        except (RuntimeWarning, UserWarning):
                            print("Warning catched: Return last stable T")
                            break
                        # the marginal are sometime not totaly respected...
                        new_T = (p[t_optimized] / new_T.sum(axis=1))[:, np.newaxis] * new_T

                    else:
                        try:
                            new_T = sinkhorn(a=p[t_optimized],
                                             b=q[t_optimized],
                                             M=L,
                                             reg=epsilon)
                        except (RuntimeWarning, UserWarning):
                            print("Warning catched: Return last stable T")
                            break
                else:

                    if labels_t is not None and labels_t[t_optimized] is not None:
                        assert not L2  # Not implemented with L2 reg

                        for label in np.unique(labels_t[t_optimized]):
                            if label == -1:
                                continue
                            # could be optimized if many classes
                            L[np.ix_(labels_s[t_optimized] != label, labels_t[t_optimized] == label)] = np.inf

                    try:
                        new_T = sinkhorn_lpl1_mm(a=p[t_optimized],
                                                 b=q[t_optimized],
                                                 M=L,
                                                 reg=epsilon,
                                                 labels_a=labels_s[t_optimized],
                                                 eta=eta)
                    except (RuntimeWarning, UserWarning):
                        print("Warning catched: Return last stable T")
                        break

        if KL * epsilon <= 0:
            new_T = (1 - learning_step) * T[t_optimized] + learning_step * new_T

        if verbose:
            if iter % 11 == 0:
                print("||T - T|| : ", ((new_T - T[t_optimized]) ** 2).sum(), )
                if L is not None and L.sum() != np.inf:
                    pass
                    # loss = np.sum(L * T[t_optimized])
                    # print("Loss :", loss, "(batched)")
                print("Real GW dist :", compute_distance_sampling_COOT_generalised(p=p, q=q,
                                                                                   loss_fun=loss_fun,
                                                                                   X1=X1, X2=X2,
                                                                                   T=T,
                                                                                   T_pos=T_pos,
                                                                                   alpha=alpha,
                                                                                   nb_samples=None,
                                                                                   M=M))
                print("")

        if time_print:
            print("Before stop ", time.time() - time_init)
        if ((T[t_optimized] - new_T) ** 2).sum() <= threshold:
            continue_loop += 1
            if continue_loop > 10:  # Number max of low modification of T
                print("Stop")
                T[t_optimized] = new_T.copy()
                break
        else:
            continue_loop = 0
        T[t_optimized] = new_T.copy()

    if log:
        log = {}
        log["gw_dist_estimated"] = compute_distance_sampling_COOT_generalised(p=p, q=q,
                                                                              loss_fun=loss_fun,
                                                                              X1=X1, X2=X2,
                                                                              T=T,
                                                                              T_pos=T_pos,
                                                                              alpha=alpha,
                                                                              nb_samples=None,
                                                                              M=M)
        return T, log
    return T


def barycenter_OTT_L2(p, q,
                      loss_fun,
                      X1,
                      X2=None,
                      M=[None, None],
                      T_pos=[[0], [0, 0]],
                      T=[None],
                      alpha=[0.5, 0.5],
                      nb_iter=500,
                      nb_samples=[1, 1],
                      epsilon=0,
                      KL=0,
                      L2=0,
                      labels_s=None,
                      eta=0,
                      verbose=False,
                      threshold=1e-20,
                      sample_t_only_init=False,
                      sample_t_init_and_iteration=False,
                      nb_loop_barycenter=10):
    def make_f(sx):
        def f(*x):
            return sx[x]

        return f

    X1_ = []
    for x1 in X1:
        X1_.append(make_f(x1))

    if X2 is None:
        X2 = []
        for a in range(len(X1)):
            t = T_pos[a]
            size = [len(q[t[j]]) for j in range(len(t))]
            X2.append(np.random.rand(*size))

    for i in range(nb_loop_barycenter):
        X2_ = []
        for x2 in X2:
            X2_.append(make_f(x2))

        T, log = CO_Generalisation_OT(p, q,
                                      loss_fun,
                                      X1_, X2_,
                                      T_pos=T_pos,
                                      T=T,
                                      alpha=alpha,
                                      nb_iter=nb_iter,
                                      nb_samples=nb_samples,
                                      epsilon=epsilon,
                                      KL=KL,
                                      L2=L2,
                                      labels_s=labels_s,
                                      eta=eta,
                                      log=True,
                                      verbose=verbose,
                                      threshold=threshold,
                                      sample_t_only_init=sample_t_only_init,
                                      sample_t_init_and_iteration=sample_t_init_and_iteration)

        for a in range(len(X1)):
            t = T_pos[a]
            temp = X1[a]
            tuple_axes = tuple(list(range(1, len(X1[a].shape))) + [0])
            for i in range(len(X1[a].shape)):
                temp = np.tensordot((T[t[i]] / q[t[i]][np.newaxis, :]).T, temp, axes=1)
                temp = np.transpose(temp, axes=tuple_axes)
            X2[a] = temp
    X2_ = []
    for x2 in X2:
        X2_.append(make_f(x2))
    loss = compute_distance_sampling_COOT_generalised_full(p=p, q=q,
                                                           loss_fun=loss_fun,
                                                           X1=X1_, X2=X2_,
                                                           T=T,
                                                           T_pos=T_pos,
                                                           alpha=alpha,
                                                           M=M)
    return X2, T, loss


def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon, eta=1,
                                max_iter=1000, tol=1e-9, verbose=False, log=False,
                                KL=False,
                                labels_s=None,
                                labels_t=None):
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    T = np.outer(p, q)  # Initialization
    if loss_fun in ["square_loss", "kl_loss"]:
        constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):
        # print(".", end="")

        Tprev = T

        # compute the gradient
        if loss_fun in ["square_loss", "kl_loss"]:
            tens = gwggrad(constC, hC1, hC2, T)
        else:
            tens = 2 * compute_L(C1=C1, C2=C2, loss_fun=loss_fun, T=T)

        if epsilon * KL > 0:
            log_T = np.log(np.clip(T, np.exp(-200), 1))
            log_T[log_T == -200] = -np.inf
            tens = tens - epsilon * KL * log_T

        if epsilon > 0:
            m = np.max(tens)
            if labels_s is not None:
                if labels_t is not None:
                    for label in np.unique(labels_t):
                        if label == -1:
                            continue
                        tens[np.ix_(labels_s != label, labels_t == label)] = np.inf

                try:
                    T = sinkhorn_lpl1_mm(a=p,
                                         b=q,
                                         M=tens / m,
                                         reg=epsilon,
                                         labels_a=labels_s,
                                         eta=eta)

                except:
                    print("The method as not converged. Return last stable T. Nb iter : " + str(cpt))
                    break
            else:
                # print("here")
                try:
                    T = sinkhorn(p, q, tens / m, epsilon)
                except:
                    print("The method as not converged. Return last stable T. Nb iter : " + str(cpt))
                    break
        else:
            if labels_t is not None:
                for label in np.unique(labels_t):
                    if label == -1:
                        continue
                    tens[np.ix_(labels_s != label, labels_t == label)] = 10000
            try:
                T = emd(p, q, tens)
            except:
                print("The method as not converged. Return last stable T. Nb iter : " + str(cpt))
                break

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        # print("log")
        if loss_fun in ["square_loss", "kl_loss"]:
            # print(q.sum(), p.sum())
            # print(p, q)
            # print(T.sum(), T.sum(axis=0), T.sum(axis=1))
            log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        else:
            log['gw_dist'] = np.sum(T * compute_L(C1, C2, loss_fun, T))
        return T, log
    else:
        return T
