#  Code adapted from the github of Co-Optimal Transport paper.

import numpy as np
import ot
from scipy import stats
from scipy.sparse import random

def sinkhorn_scaling(a,b,K,numItermax=1000, stopThr=1e-9, verbose=False,log=False,always_raise=False, **kwargs):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # init data
    Nini = len(a)
    Nfin = len(b)

    if len(b.shape) > 1:
        nbb = b.shape[1]
    else:
        nbb = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if nbb:
        u = np.ones((Nini, nbb)) / Nini
        v = np.ones((Nfin, nbb)) / Nfin
    else:
        u = np.ones(Nini) / Nini
        v = np.ones(Nfin) / Nfin

    # print(reg)
    # print(np.min(K))

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        zero_in_transp=np.any(KtransposeU == 0)
        nan_in_dual= np.any(np.isnan(u)) or np.any(np.isnan(v))
        inf_in_dual=np.any(np.isinf(u)) or np.any(np.isinf(v))
        if zero_in_transp or nan_in_dual or inf_in_dual:
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration in sinkhorn_scaling', cpt)
            #if zero_in_transp:
                #print('Zero in transp : ',KtransposeU)
            #if nan_in_dual:
                #print('Nan in dual')
                #print('u : ',u)
                #print('v : ',v)
                #print('KtransposeU ',KtransposeU)
                #print('K ',K)
                #print('M ',M)

            #    if always_raise:
            #        raise NanInDualError
            #if inf_in_dual:
            #    print('Inf in dual')
            u = uprev
            v = vprev

            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                    np.sum((v - vprev)**2) / np.sum((v)**2)
            else:
                transp = u.reshape(-1, 1) * (K * v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    if log:
        log['u'] = u
        log['v'] = v

    if nbb:  # return only loss
        res = np.zeros((nbb))
        for i in range(nbb):
            res[i] = np.sum(
                u[:, i].reshape((-1, 1)) * K * v[:, i].reshape((1, -1)) * M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def random_gamma_init(p, q, **kwargs):
    """ Returns random coupling matrix with marginal p,q
    """
    rvs = stats.beta(1e-1, 1e-1).rvs
    S = random(len(p), len(q), density=1, data_rvs=rvs)
    return sinkhorn_scaling(p, q, S.A, **kwargs)


def init_matrix_np(X1, X2, v1, v2):
    def f1(a):
        return (a ** 2)

    def f2(b):
        return (b ** 2)

    def h1(a):
        return a

    def h2(b):
        return 2 * b

    constC1 = np.dot(np.dot(f1(X1), v1.reshape(-1, 1)),
                     np.ones(f1(X2).shape[0]).reshape(1, -1))
    constC2 = np.dot(np.ones(f1(X1).shape[0]).reshape(-1, 1),
                     np.dot(v2.reshape(1, -1), f2(X2).T))

    constC = constC1 + constC2
    hX1 = h1(X1)
    hX2 = h2(X2)

    return constC, hX1, hX2


def cot_numpy(X1, X2, w1=None, w2=None, v1=None, v2=None,
              labels_s=None, labels_t=None,
              niter=10, algo='emd', reg=0, algo2='emd',
              eta=0,
              reg2=0, verbose=True, log=False, random_init=False, C_lin=None):
    if reg == 0:
        algo = "emd"
        algo2 = "emd"
    if v1 is None:
        v1 = np.ones(X1.shape[1]) / X1.shape[1]  # is (d,)
    if v2 is None:
        v2 = np.ones(X2.shape[1]) / X2.shape[1]  # is (d',)
    if w1 is None:
        w1 = np.ones(X1.shape[0]) / X1.shape[0]  # is (n',)
    if w2 is None:
        w2 = np.ones(X2.shape[0]) / X2.shape[0]  # is (n,)

    if not random_init:
        Ts = np.ones((X1.shape[0], X2.shape[0])) / (X1.shape[0] * X2.shape[0])  # is (n,n')
        Tv = np.ones((X1.shape[1], X2.shape[1])) / (X1.shape[1] * X2.shape[1])  # is (d,d')
    else:
        Ts = random_gamma_init(w1, w2)
        Tv = random_gamma_init(v1, v2)

    constC_s, hC1_s, hC2_s = init_matrix_np(X1, X2, v1, v2)

    constC_v, hC1_v, hC2_v = init_matrix_np(X1.T, X2.T, w1, w2)
    cost = np.inf

    log_out = {}
    log_out['cost'] = []
    # print("n_iter", niter)
    for i in range(niter):
        # print("i", i)
        Tsold = Ts
        Tvold = Tv
        costold = cost

        M = constC_s - np.dot(hC1_s, Tv).dot(hC2_s.T)
        if C_lin is not None:
            M = M + C_lin
        if algo == 'emd':
            M_temp = M.copy()
            if labels_t[0] is not None:
                for label in np.unique(labels_t[0]):
                    if label == -1:
                        continue
                    M_temp[np.ix_(labels_s[0] != label, labels_t[0] == label)] = np.inf
            Ts = ot.emd(w1, w2, M_temp, numItermax=1e7)
        elif algo == 'sinkhorn':
            Ts = ot.sinkhorn(w1, w2, M, reg)
        elif algo == 'supervised':
            if labels_s[0] is not None:
                M_temp = M.copy()
                M_temp = M_temp / M_temp.sum()
                # print("labels_s", labels_s[0])
                # print("labels_t", labels_t[0])
                if labels_t[0] is not None:
                    for label in np.unique(labels_t[0]):
                        if label == -1:
                            continue
                        # print("labels_s == label", labels_s[0] != label)
                        # print("labels_t == label", labels_t[0] != label)
                        M_temp[np.ix_(labels_s[0] != label, labels_t[0] == label)] = np.inf
                        # print(M_temp)
                    try:
                        Ts = ot.sinkhorn_lpl1_mm(a=w1, b=w2, M=M_temp, reg=reg, labels_a=labels_s[0], eta=eta)
                    except:
                        print("break1")
            else:
                try:
                    Ts = ot.sinkhorn(v1, v2, M, reg2)
                except:
                    print("break2")
        # print("Ts", Ts)
        M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T)

        if algo2 == 'emd':
            M_temp = M.copy()
            if labels_t[1] is not None:
                for label in np.unique(labels_t[1]):
                    if label == -1:
                        continue
                    M_temp[np.ix_(labels_s[1] != label, labels_t[1] == label)] = np.inf
            Tv = ot.emd(v1, v2, M_temp, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Tv = ot.sinkhorn(v1, v2, M, reg2)
        elif algo == 'supervised':
            if labels_s[1] is not None:
                M_temp = M.copy()
                M_temp = M_temp / M_temp.sum()
                if labels_t[1] is not None:
                    for label in np.unique(labels_t[1]):
                        if label == -1:
                            continue

                        M_temp[np.ix_(labels_s[1] != label, labels_t[1] == label)] = np.inf
                # print("M_temp", M_temp)

                try:
                    Tv = ot.sinkhorn_lpl1_mm(a=v1, b=v2, M=M_temp, reg=reg2, labels_a=labels_s[1], eta=eta)
                except:
                    print("break1")
            else:
                try:
                    Tv = ot.sinkhorn(v1, v2, M, reg2)
                except:
                    print("break2")
        delta = np.linalg.norm(Ts - Tsold) + np.linalg.norm(Tv - Tvold)
        cost = np.sum(M * Tv)

        if log:
            log_out['cost'].append(cost)

        if verbose:
            print('Delta: {0}  Loss: {1}'.format(delta, cost))

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            if verbose:
                print('converged at iter ', i)
            break
    if log:
        return Ts, Tv, cost, log_out
    else:
        return Ts, Tv, cost

