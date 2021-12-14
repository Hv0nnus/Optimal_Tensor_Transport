__all__ = ['get_MulK_quadruplets', 'get_AddS_quadruplets', 'get_MulK_triplets', 'get_AddS_triplets']

import numpy as np


def get_MulK_quadruplets(oracle, n_examples):
    """Returns a symmetric similarity matrix representing the similarities
    between all the examples using the MulK quadruplets approach.

    Parameters
    ----------
    oracle : OracleQuadruplets
        An oracle used to query the comparisons. It should exhibit a
        method comparisons_to_ref(i,j) which return all the
        comparisons associated with the pair (i,j) in a sparse matrix
        where in entry (k,l), the value 1 indicates that the
        quadruplet (i,j,k,l) is available, the value -1 indicates that
        the quadruplet (k,l,i,j) is available, and the value 0
        indicates that neither of the quadruplets is available.

    n_examples : int
        The number of examples handled by the oracle.

    Returns
    -------
    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the examples.

    """
    kernel = np.zeros((n_examples, n_examples))

    comps = oracle.get_MulK_comparisons()
    kernel = comps.dot(comps.transpose())
    kernel = kernel.toarray()
    np.fill_diagonal(kernel, 0)

    return kernel


def get_AddS_quadruplets(oracle, n_examples):
    """Returns a symmetric similarity matrix representing the similarities
    between all the examples using the AddS quadruplets approach.

    Parameters
    ----------
    oracle : OracleQuadruplets
        An oracle used to query the comparisons. It should exhibit a
        method comparisons_to_ref(i,j) which return all the
        comparisons associated with the pair (i,j) in a sparse matrix
        where in entry (k,l), the value 1 indicates that the
        quadruplet (i,j,k,l) is available, the value -1 indicates that
        the quadruplet (k,l,i,j) is available, and the value 0
        indicates that neither of the quadruplets is available.

    n_examples : int
        The number of examples handled by the oracle.

    Returns
    -------
    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the examples.

    """
    kernel = np.zeros((n_examples, n_examples))

    comps = oracle.get_AddS_comparisons()
    entries = comps.sum(axis=1).A1
    i, j = oracle._get_ij(np.arange((n_examples * (n_examples - 1)) // 2))
    kernel[i, j] = entries

    kernel += kernel.transpose()

    return kernel


def get_MulK_triplets(oracle, n_examples):
    """Returns a symmetric similarity matrix representing the similarities
    between all the examples using the MulK triplets approach.

    Parameters
    ----------
    oracle : OracleTriplets
        An oracle used to query the comparisons. It should exhibit a
        method get_MulK_comparisons() that returns a scipy csr matrix
        of shape(n_examples,(n_examples choose 2)) containing values
        in {1,-1,0}. Given i!=j,k, j<k, in entry (i,self._get_k(j,k)),
        the value 1 indicates that the triplet (i,j,k) is available,
        the value -1 indicates that the triplet (i,k,j) is available,
        and the value 0 indicates that neither of the triplets is
        available.


    n_examples : int
        The number of examples handled by the oracle.

    Returns
    -------
    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the examples.
zz
    """
    kernel = np.zeros((n_examples, n_examples))

    comps = oracle.get_MulK_comparisons()
    kernel = comps.dot(comps.transpose())

    norms = np.sqrt(comps.getnnz(axis=1))
    norms = norms.reshape(-1, 1) @ norms.reshape(1, -1)
    norms = np.where(norms == 0, 1, norms)  # This is to avoid issues with the true divide when the norm is 0 for i or j

    kernel = kernel.toarray() / norms
    np.fill_diagonal(kernel, 0)

    return kernel


def get_AddS_triplets(oracle, n_examples):
    """Returns a symmetric similarity matrix representing the similarities
    between all the examples using the AddS triplets approach.

    Parameters
    ----------
    oracle : OracleTriplets
        An oracle used to query the comparisons. It should exhibit a
        method get_AddS_comparisons() that returns a scipy csr matrix,
        shape(n_examples**2,n_examples) A scipy csr_matrix containing
        values in {1,-1,0}. Given i!=j,k, in entry (i*n_examples+j,k),
        the value 1 indicates that the triplet (i,j,k) is available,
        the value -1 indicates that the triplet (i,k,j) is available,
        and the value 0 indicates that neither of the triplets is
        available.

    n_examples : int
        The number of examples handled by the oracle.

    Returns
    -------
    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the examples.

    """
    kernel = np.zeros((n_examples, n_examples))

    comps = oracle.get_AddS_comparisons()
    entries = comps.sum(axis=1).A1

    indices = np.arange(n_examples ** 2)
    i = indices // n_examples
    j = indices - i * n_examples

    kernel[i, j] = entries

    kernel += kernel.transpose()

    return kernel
