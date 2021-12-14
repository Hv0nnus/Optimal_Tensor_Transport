__all__ = ['SPUR', 'SDP_known_k', 'Lambda_SDP']

import numpy as np
from scipy.sparse import tril,csc_matrix
import scs
from scipy.sparse.linalg import eigsh

def SDP_known_k(K, N, n_clusters, verbose=False, eps=1e-3, warm=None, get_res=False):
    def get_ij(k):
        n_examples = N
        j = n_examples - 1 - (np.sqrt(-8*k + 4*n_examples*(n_examples+1)-7)/2 - 1/2).astype(int)
        i = k + j - (n_examples*(n_examples+1))//2 + ((n_examples-j)*((n_examples-j)+1))//2
        
        return i,j

    def get_k(i,j):
        n_examples = N
        k = ((n_examples*(n_examples+1))//2) - ((n_examples-j)*((n_examples-j)+1))//2 + i - j
        
        return k

    K -= np.diag(np.diag(K))
    
    n_variables = (N*(N+1))//2

    # Objective function
    K_sparse = tril(-K,format="coo")
    c = np.zeros((n_variables,))
    c[get_k(K_sparse.row,K_sparse.col)] = K_sparse.data

    # Constraints initialization
    n_zero_cone = N + 1
    n_pos_cone = n_variables
    dim_sdp_cone = N
    n_constraints = n_zero_cone + n_pos_cone + n_variables
    row_A = []
    col_A = []
    data_A = []
    b = np.zeros((n_constraints,))

    # Rows/columns constraints
    for i in range(N):
        row_A += [i]*N
        col_A += get_k(np.array([i]*i+list(range(i,N))),np.array(list(range(i))+[i]*(N-i))).tolist()
        data_A += [1.0]*N
        
        b[i] = 1.0

    # Trace constraint
    I_sparse = tril(np.eye(N),format="coo")

    row_A += [N]*I_sparse.nnz 
    col_A += get_k(I_sparse.row,I_sparse.col).tolist()
    data_A += I_sparse.data.tolist()

    b[N] = n_clusters

    # Positive entries constraints
    ind = np.arange(n_variables)
    row_A += (N+1+ind).tolist()
    col_A += ind.tolist()
    data_A += [-1.0]*n_variables
    
    b[N+1+ind] = 0.0

    # SDP constraint
    k = np.arange(n_variables)
    i,j = get_ij(k)

    row_A += (N+1+n_variables+k).tolist()
    col_A += k.tolist()
    data_A += np.where(i==j,-1,-np.sqrt(2)).tolist()

    b[(N+1+n_variables+k)] = 0.0
    
    A = csc_matrix((data_A,(row_A,col_A)),shape=(n_constraints,n_variables))

    # Pack SCS data
    data = {
        "A": A,
        "b": b,
        "c": c
    }

    if warm is not None:
        data["x"] = warm["x"]
        data["y"] = warm["y"]
        data["s"] = warm["s"]
            

    cones = {
        "f": n_zero_cone,
        "l": n_pos_cone,
        "s": [dim_sdp_cone]
    }

    res = scs.solve(data,cones,verbose=verbose,eps=eps)

    # Unpack the result
    X = None
    if res['info']['status'] == 'Solved':
        X = np.zeros((N,N))
        k = np.arange(n_variables)
        i,j = get_ij(k)

        X[i,j] = res['x']
        X[j,i] = res['x']
        assert np.all(X==X.T)
    elif res['info']['status'] == 'Solved/Inaccurate':
        print("Inaccurate solution, should rerun with more iterations.", flush=True)
        X = np.zeros((N,N))
        k = np.arange(n_variables)
        i,j = get_ij(k)

        X[i,j] = res['x']
        X[j,i] = res['x']
        assert np.all(X==X.T)
    else:
        print("Unfeasible.", flush=True)

    if get_res:
        return X, res
    else:
        return X

def Lambda_SDP(K, N, lbda, verbose=False, eps=1e-3, warm=None, get_res=False):
    def get_ij(k):
        n_examples = N
        j = n_examples - 1 - (np.sqrt(-8*k + 4*n_examples*(n_examples+1)-7)/2 - 1/2).astype(int)
        i = k + j - (n_examples*(n_examples+1))//2 + ((n_examples-j)*((n_examples-j)+1))//2
        
        return i,j

    def get_k(i,j):
        n_examples = N
        k = ((n_examples*(n_examples+1))//2) - ((n_examples-j)*((n_examples-j)+1))//2 + i - j
        
        return k

    K -= np.diag(np.diag(K))

    n_variables = (N*(N+1))//2

    # Objective function
    K_sparse = tril(-K+lbda*np.eye(N),format="coo")
    c = np.zeros((n_variables,))
    c[get_k(K_sparse.row,K_sparse.col)] = K_sparse.data

    # Constraints initialization
    n_zero_cone = N
    n_pos_cone = n_variables
    dim_sdp_cone = N
    n_constraints = n_zero_cone + n_pos_cone + n_variables
    row_A = []
    col_A = []
    data_A = []
    b = np.zeros((n_constraints,))

    # Rows/columns constraints
    for i in range(N):
        row_A += [i]*N
        col_A += get_k(np.array([i]*i+list(range(i,N))),np.array(list(range(i))+[i]*(N-i))).tolist()
        data_A += [1.0]*N
        
        
        b[i] = 1.0

    # Positive entries constraints
    ind = np.arange(n_variables)
    row_A += (N+ind).tolist()
    col_A += ind.tolist()
    data_A += [-1.0]*n_variables
    
    b[N+ind] = 0.0
    
    # SDP constraint
    k = np.arange(n_variables)
    i,j = get_ij(k)

    row_A += (N+n_variables+k).tolist()
    col_A += k.tolist()
    data_A += np.where(i==j,-1,-np.sqrt(2)).tolist()

    b[(N+n_variables+k)] = 0.0
    
    A = csc_matrix((data_A,(row_A,col_A)),shape=(n_constraints,n_variables))

    # Pack SCS data
    data = {
        "A": A,
        "b": b,
        "c": c
    }

    if warm is not None:
        data["x"] = warm["x"]
        data["y"] = warm["y"]
        data["s"] = warm["s"]
    
    cones = {
        "f": n_zero_cone,
        "l": n_pos_cone,
        "s": [dim_sdp_cone]
    }
    
    res = scs.solve(data,cones,verbose=verbose,eps=eps)

    # Unpack the result
    X = None
    if res['info']['status'] == 'Solved':
        X = np.zeros((N,N))
        k = np.arange(n_variables)
        i,j = get_ij(k)

        X[i,j] = res['x']
        X[j,i] = res['x']
        assert np.all(X==X.T)
    elif res['info']['status'] == 'Solved/Inaccurate':
        print("Inaccurate solution, should rerun with more iterations.", flush=True)
        X = np.zeros((N,N))
        k = np.arange(n_variables)
        i,j = get_ij(k)

        X[i,j] = res['x']
        X[j,i] = res['x']
        assert np.all(X==X.T)
    else:
        print("Unfeasible.", flush=True)

    if get_res:
        return X, res
    else:
        return X

def SPUR(K, N, T, n_observ, verbose=False, eps=1e-3, warm=True):
    """


    Parameters
    ----------
    K : NxN numpy array
        kernel matrix

    N : int
        numer of Datapoints

    T : int
        number of lambda steps

    n_observ : int
        number of comaprisons

    verbose : boolean
        whether we should print solver output

    eps : float
        termination criterion for the solver

    Returns
    -------
    X_opt : NxN numpy array
        estimated solution for the semidefinie programming problem

    k_hat : int
        estimated unber of clusters

    """
    Lambda_low = np.sqrt(n_observ*np.log(N)/N)
    Lambda_up = n_observ/N

    # Initial estimates
    X_hat_low = Lambda_SDP(K, N, Lambda_low, verbose=verbose, eps=eps)
    k_hat_low = int(round(np.trace(X_hat_low)))
    print("Lower bound estimated k: {}".format(k_hat_low), flush=True)
    
    X_hat_up = Lambda_SDP(K, N, Lambda_up, verbose=verbose, eps=eps)    
    k_hat_up = int(round(np.trace(X_hat_up)))
    print("Upper bound estimated k: {}".format(k_hat_up), flush=True)

    res = None
    X_list = []
    k_list = []
    theta_list = []
    for k_test in range(max(2,k_hat_up),k_hat_low+T+1):
        if warm:
            X_hat, res = SDP_known_k(K, N, k_test, verbose=verbose, eps=eps, warm=res, get_res=True)
        else:
            X_hat = SDP_known_k(K, N, k_test, verbose=verbose, eps=eps)
            
        # Compute the trace, it should be k_test
        X_trace = np.trace(X_hat)
        # integer approximation of the trace
        k_hat = int(round(X_trace))

        # Compute the criterion
        _, s, _ = np.linalg.svd(X_hat)
        theta = np.sum(s[:(k_hat)]) / X_trace

        print("Constraint k: {} | Estimated k: {} | Criterion: {}".format(k_test, k_hat, theta), flush=True)
        X_list.append(X_hat)
        k_list.append(k_hat)
        theta_list.append(theta)

        # When theta is this large (it is bounded above by 1 anyway), we probably found the correct number of clusters already so we do not need to search further
        if theta > 0.99:
            break

    # choose X and k where theta is max
    theta_opt_index = int(np.argmax(theta_list))
    X_opt = X_list[theta_opt_index]
    k_opt = k_list[theta_opt_index]
    
    return X_opt, k_opt
