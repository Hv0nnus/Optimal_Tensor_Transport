__all__ = ['HandlerQuadruplets', 'ListQuadruplets', 'OracleQuadruplets', 'HandlerTriplets', 'ListTriplets',
           'HandlerTriplets']

import numpy as np

from scipy.sparse import csr_matrix, coo_matrix
import time
import random as rd


class HandlerQuadruplets():
    """A data handler that returns quadruplets in diverse forms.

    Parameters
    ----------
    comparisons : scipy coo matrix, shape ((n_examples choose 2), (n_examples choose 2))
        A scipy coo_matrix containing values in {1,-1,0}. Given i<j,
        k<l, in entry (self._get_k(i,j),self._get_k(k,l)), the value 1
        indicates that the quadruplet (i,j,k,l) is available, the
        value -1 indicates that the quadruplet (k,l,i,j) is available,
        and the value 0 indicates that neither of the quadruplets is
        available.

    n_examples : int
        The number of examples handled.

    n_quadruplets : int or float, optional
        The number of unique quadruplets handled.

    Attributes 
    ---------- 
    comparisons : scipy coo matrix, shape ((n_examples choose 2), (n_examples choose 2))
        A scipy coo_matrix containing values in {1,-1,0}. Given i<j,
        k<l, in entry (self._get_k(i,j),self._get_k(k,l)), the value 1
        indicates that the quadruplet (i,j,k,l) is available, the
        value -1 indicates that the quadruplet (k,l,i,j) is available,
        and the value 0 indicates that neither of the quadruplets is
        available.

    n_examples : int
        The number of examples handled.

    n_comparisons : int
        The number of comparisons handled.

    """

    def __init__(self, comparisons, n_quadruplets, n_examples):
        self.comparisons = comparisons
        self.n_quadruplets = n_quadruplets
        self.n_examples = n_examples

    def _get_ij(self, k, n_examples=None):
        """Returns the row and column coordinates given the index of the
        entries of an off-diagonal upper triangular matrix where the
        elements are taken in a row-major order:
        [. 0 1 2 3
         . . 4 5 6
         . . . 7 8
         . . . . 9
         . . . . .]

        Parameters
        ----------
        k : int or numpy array
            The row-major index of the example, between 0 and
            (n_examples choose 2).
       
        n_example : int
            The number of rows and columns in the matrix. If None,
            self.n_examples is used. (Default: None).

        Returns
        -------
        i : int or numpy array, shape(k.shape)
            The row index of the example, between 0 and n_examples.

        j : int or numpy array, shape(k.shape)
            The column index of the example, between i and n_examples.

        Notes
        -----
        The original formulation was taken from the following link:
        https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix

        """
        if n_examples is None:
            n_examples = self.n_examples

        i = n_examples - 2 - (np.sqrt(-8 * k + 4 * n_examples * (n_examples - 1) - 7) / 2 - 1 / 2).astype(int)
        j = k + i + 1 - (n_examples * (n_examples - 1)) // 2 + ((n_examples - i) * ((n_examples - i) - 1)) // 2

        return i, j

    def _get_k(self, i, j, n_examples=None):
        """Given the row and column coordinates, returns the index of entries
        of an off-diagonal upper triangular matrix where the elements
        are taken in a row-major order:
        [. 0 1 2 3
         . . 4 5 6
         . . . 7 8
         . . . . 9
         . . . . .]

        Returns
        -------
        i : int or numpy array
            The row index of the example, between 0 and n_examples.

        j : int or numpy array, shape(i.shape)
            The column index of the example, between i and n_examples.

        n_example : int
            The number of rows and columns in the matrix. If None,
            self.n_examples is used. (Default: None).

        Parameters
        ----------
        k : int or numpy array, shape(i.shape)
            The row-major index of the example, between 0 and (n_examples choose 2).

        Notes
        -----
        The original formulation was taken from the following link:
        https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix

        """
        if n_examples is None:
            n_examples = self.n_examples

        k = ((n_examples * (n_examples - 1)) // 2) - ((n_examples - i) * ((n_examples - i) - 1)) // 2 + j - i - 1

        return k

    def get_MulK_comparisons(self):
        """Get a sparse matrix representing the comparisons in a way that is
        easy to handle for MulK.

        Returns
        -------
        MulK_comparisons : scipy csr matrix, shape(n_examples,n_examples*(n_examples choose 2))
            A scipy csr_matrix containing values in {1,-1,0}. Given
            i<j and k<l, in entry (i,j*(n_examples choose
            2)+self._get_k(k,l)), the value 1 indicates that the
            quadruplet (i,j,k,l) is available, the value -1 indicates
            that the quadruplet (k,l,i,j) is available, and the value
            0 indicates that neither of the quadruplets is available.

        """
        i, j = self._get_ij(self.comparisons.row)

        n_kl = (self.n_examples * (self.n_examples - 1)) // 2

        rows_i = i
        columns_i = j * n_kl + self.comparisons.col

        rows_j = j
        columns_j = i * n_kl + self.comparisons.col

        rows = np.concatenate((rows_i, rows_j))
        columns = np.concatenate((columns_i, columns_j))
        entries = np.concatenate((self.comparisons.data, self.comparisons.data))

        MulK_comparisons = csr_matrix((entries, (rows, columns)), shape=(self.n_examples, self.n_examples * n_kl),
                                      dtype=int)

        return MulK_comparisons

    def get_AddS_comparisons(self):
        """Get a sparse matrix representing the comparisons in a way that is
        easy to handle for AddS.

        Returns
        -------
        AddS_comparisons : scipy csr matrix, shape((n_examples choose 2),(n_examples choose 2))
            A scipy csr_matrix containing values in {1,-1,0}. Given
            i<j and k<l, in entry (self._get_k(i,j),self._get_k(k,l)),
            the value 1 indicates that the quadruplet (i,j,k,l) is
            available, the value -1 indicates that the quadruplet
            (k,l,i,j) is available, and the value 0 indicates that
            neither of the quadruplets is available.

        """
        AddS_comparisons = self.comparisons.tocsr()

        return AddS_comparisons


class ListQuadruplets(HandlerQuadruplets):
    """A data handler that can return quadruplets in several forms given a
    list of quadruplets.
    
    Parameters
    ----------
    list_comparisons : numpy array, shape (n_comparisons, 4)
        A numpy array where each row indicates that the quadruplet
        (row[0],row[1],row[2],row[3]) is available.

    n_examples : int
        The number of examples handled.

    n_comparsions : int
        The number of unique quadruplets handled.

    Attributes 
    ---------- 
    comparisons : scipy coo matrix, shape ((n_examples choose 2), (n_examples choose 2))
        A scipy coo_matrix containing values in {1,-1,0}. Given i<j,
        k<l, in entry (self._get_k(i,j),self._get_k(k,l)), the value 1
        indicates that the quadruplet (i,j,k,l) is available, the
        value -1 indicates that the quadruplet (k,l,i,j) is available,
        and the value 0 indicates that neither of the quadruplets is
        available.

    n_examples : int
        The number of examples handled.

    n_comparisons : int
        The number of comparisons handled.

    """

    def __init__(self, list_comparisons, n_examples, n_comparisons):
        self.n_examples = n_examples

        n_entries = (n_examples * (n_examples - 1)) // 2

        rows = self._get_k(np.amin(list_comparisons[:, :2], axis=1), np.amax(list_comparisons[:, :2], axis=1))
        columns = self._get_k(np.amin(list_comparisons[:, 2:], axis=1), np.amax(list_comparisons[:, 2:], axis=1))
        entries = np.ones((n_comparisons,))

        # We also populate the lower triangular part of the matrix for convenience reasons
        comparisons = coo_matrix(
            (np.concatenate((entries, -entries)), (np.concatenate((rows, columns)), np.concatenate((columns, rows)))),
            shape=(n_entries, n_entries), dtype=int)
        comparisons.eliminate_zeros()

        super(ListQuadruplets, self).__init__(comparisons, n_comparisons, n_examples)


class OracleQuadruplets(HandlerQuadruplets):
    """An oracle that returns passively queried quadruplets given a
    similarity matrix.

    Parameters
    ----------
    similarities : numpy array, shape (n_examples, n_examples)
        A numpy array containing the similarities that should be used
        to generate the quadruplets.

    n_examples : int
        The number of examples handled by the oracle.

    n_quadruplets : int or float, optional
        If n_quadruplets is strictly greater than 1 it represent the number of
        comparisons that should be generated, if it is lower than one
        it represented the overall proportion of quadruplets that
        should be generated. (Default: 0.1).

    proportion_noise : float, optional
        The overall proportion of noise in the quadruplets (flipped
        comparisons). (Default: 0.0).

    seed : int or None
        The seed used to initialize the random number generators. If
        None the current time is used, that is
        int(time.time()). (Default: None).

    Attributes 
    ---------- 
    similarities : numpy array, shape (n_examples, n_examples)
        A numpy array containing the similarities that should be used
        to generate the quadruplets.

    comparisons : scipy coo matrix, shape ((n_examples choose 2), (n_examples choose 2))
        A scipy coo_matrix containing values in {1,-1,0}. Given i<j,
        k<l, in entry (self._get_k(i,j),self._get_k(k,l)), the value 1
        indicates that the quadruplet (i,j,k,l) is available, the
        value -1 indicates that the quadruplet (k,l,i,j) is available,
        and the value 0 indicates that neither of the quadruplets is
        available.

    n_comparisons : int
        The number of comparisons generated by the oracle.

    n_examples : int
        The number of examples handled by the oracle.

    proportion_noise : float
        The overall proportion of noise in the quadruplets (flipped
        comparisons).

    seed : int
        The seed used to initialize the random number generators.

    """

    def __init__(self, similarities, n_examples, n_quadruplets=0.1
                 , proportion_noise=0.0, seed=None):
        self.similarities = similarities

        self.n_examples = n_examples

        self.proportion_noise = proportion_noise

        if seed is not None:
            self.seed = seed
        else:
            self.seed = int(time.time())

        comparisons, n_comparisons = self._get_comparisons(n_examples, n_quadruplets)

        super(OracleQuadruplets, self).__init__(comparisons, n_comparisons, n_examples)

    def _get_comparisons(self, n_examples, n_comparisons):
        """Constructs the comparisons sparse matrix with n_comparisons
        quadruplets with a probability of noise equal to
        proportion_noise. This method is deterministic.

        """

        random_state = np.random.RandomState(self.seed)
        rd_random_state = rd.Random(self.seed + 42)  # To use a different seed than numpy

        n_entries = (n_examples * (n_examples - 1)) // 2

        if n_comparisons <= 1:
            n_comparisons = random_state.binomial((n_entries * (n_entries - 1)) // 2, n_comparisons)

        rows, columns = self._get_ij(
            np.array(rd_random_state.sample(range((n_entries * (n_entries - 1)) // 2), n_comparisons)), n_entries)

        i, j = self._get_ij(rows)

        k, l = self._get_ij(columns)

        noise = np.where(random_state.rand(n_comparisons) < self.proportion_noise, -1, 1)
        entries = np.multiply(np.where(self.similarities[i, j] > self.similarities[k, l], 1, 0) + np.where(
            self.similarities[i, j] < self.similarities[k, l], -1, 0), noise)

        # We also populate the lower triangular part of the matrix for convenience reasons
        comparisons = coo_matrix(
            (np.concatenate((entries, -entries)), (np.concatenate((rows, columns)), np.concatenate((columns, rows)))),
            shape=(n_entries, n_entries), dtype=int)
        comparisons.eliminate_zeros()

        return comparisons, n_comparisons


class HandlerTriplets():
    """A data handler that returns triplets in diverse forms.

    Parameters
    ----------
    comparisons : scipy coo matrix, shape (n_examples, (n_examples choose 2))
        A scipy coo_matrix containing values in {1,-1,0}. Given
        i!=j,k, j<k, in entry (i,self._get_k(j,k)), the value 1
        indicates that the triplet (i,j,k) is available, the value -1
        indicates that the triplet (i,k,j) is available, and the value
        0 indicates that neither of the triplets is available.

    n_examples : int
        The number of examples handled.

    n_triplets : int or float, optional
        The number of unique triplets handled.

    Attributes 
    ---------- 
    comparisons : scipy coo matrix, shape (n_examples, (n_examples choose 2))
        A scipy coo_matrix containing values in {1,-1,0}. Given
        i!=j,k, j<k, in entry (i,self._get_k(j,k)), the value 1
        indicates that the triplet (i,j,k) is available, the value -1
        indicates that the triplet (i,k,j) is available, and the value
        0 indicates that neither of the triplets is available.

    n_examples : int
        The number of examples handled.

    n_comparisons : int
        The number of comparisons handled.

    """

    def __init__(self, comparisons, n_triplets, n_examples):
        self.comparisons = comparisons
        self.n_quadruplets = n_triplets
        self.n_examples = n_examples

    def _get_ij(self, k, n_examples=None):
        """Returns the row and column coordinates given the index of the
        entries of an off-diagonal upper triangular matrix where the
        elements are taken in a row-major order:
        [. 0 1 2 3
         . . 4 5 6
         . . . 7 8
         . . . . 9
         . . . . .]

        Parameters
        ----------
        k : int or numpy array
            The row-major index of the example, between 0 and
            (n_examples choose 2).
       
        n_example : int
            The number of rows and columns in the matrix. If None,
            self.n_examples is used. (Default: None).

        Returns
        -------
        i : int or numpy array, shape(k.shape)
            The row index of the example, between 0 and n_examples.

        j : int or numpy array, shape(k.shape)
            The column index of the example, between i and n_examples.

        Notes
        -----
        The original formulation was taken from the following link:
        https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix

        """
        if n_examples is None:
            n_examples = self.n_examples

        i = n_examples - 2 - (np.sqrt(-8 * k + 4 * n_examples * (n_examples - 1) - 7) / 2 - 1 / 2).astype(int)
        j = k + i + 1 - (n_examples * (n_examples - 1)) // 2 + ((n_examples - i) * ((n_examples - i) - 1)) // 2

        return i, j

    def _get_k(self, i, j, n_examples=None):
        """Given the row and column coordinates, returns the index of entries
        of an off-diagonal upper triangular matrix where the elements
        are taken in a row-major order:
        [. 0 1 2 3
         . . 4 5 6
         . . . 7 8
         . . . . 9
         . . . . .]

        Returns
        -------
        i : int or numpy array
            The row index of the example, between 0 and n_examples.

        j : int or numpy array, shape(i.shape)
            The column index of the example, between i and n_examples.

        n_example : int
            The number of rows and columns in the matrix. If None,
            self.n_examples is used. (Default: None).

        Parameters
        ----------
        k : int or numpy array, shape(i.shape)
            The row-major index of the example, between 0 and (n_examples choose 2).

        Notes
        -----
        The original formulation was taken from the following link:
        https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix

        """
        if n_examples is None:
            n_examples = self.n_examples

        k = ((n_examples * (n_examples - 1)) // 2) - ((n_examples - i) * ((n_examples - i) - 1)) // 2 + j - i - 1

        return k

    def get_MulK_comparisons(self):
        """Get a sparse matrix representing the comparisons in a way that is
        easy to handle for MulK.

        Returns
        -------
        MulK_comparisons : scipy csr matrix, shape(n_examples,(n_examples choose 2)) 
            A scipy csr_matrix containing values in {1,-1,0}. Given
            i!=j,k, j<k, in entry (i,self._get_k(j,k)), the value 1
            indicates that the triplet (i,j,k) is available, the value
            -1 indicates that the triplet (i,k,j) is available, and
            the value 0 indicates that neither of the triplets is
            available.

        """
        MulK_comparisons = self.comparisons.tocsr()

        return MulK_comparisons

    def get_AddS_comparisons(self):
        """Get a sparse matrix representing the comparisons in a way that is
        easy to handle for AddS.

        Returns
        -------
        AddS_comparisons : scipy csr matrix, shape(n_examples**2,n_examples)
            A scipy csr_matrix containing values in {1,-1,0}. Given
            i!=j,k, in entry (i*n_examples+j,k), the value 1 indicates
            that the triplet (i,j,k) is available, the value -1
            indicates that the triplet (i,k,j) is available, and the
            value 0 indicates that neither of the triplets is
            available.

        """
        i = self.comparisons.row
        j, k = self._get_ij(self.comparisons.col)

        n_pairs = self.n_examples

        rows_j = i * n_pairs + j
        columns_j = k

        rows_k = i * n_pairs + k
        columns_k = j

        rows = np.concatenate((rows_j, rows_k))
        columns = np.concatenate((columns_j, columns_k))
        entries = np.concatenate((self.comparisons.data, -self.comparisons.data))

        AddS_comparisons = csr_matrix((entries, (rows, columns)), shape=(self.n_examples ** 2, self.n_examples),
                                      dtype=int)

        return AddS_comparisons

    def get_tSTE_comparisons(self):
        """Get a numpy array representing the comparisons in a way that is
        easy to handle for tSTE.

        Returns
        -------
        tSTE_comparisons : numpy array, shape(n_comparisons, 3)
            A numpy array containing indices of the examples such that
            a row (i,j,k) indicates that the triplet (i,j,k) is
            available.

        """
        i = self.comparisons.row
        j, k = self._get_ij(self.comparisons.col)

        tSTE_comparisons = np.zeros((self.n_quadruplets, 3), dtype=int)

        tSTE_comparisons[:, 0] = i
        tSTE_comparisons[:, 1] = (self.comparisons.data == 1) * j + (self.comparisons.data == -1) * k
        tSTE_comparisons[:, 2] = (self.comparisons.data == -1) * j + (self.comparisons.data == 1) * k

        return tSTE_comparisons


class ListTriplets(HandlerTriplets):
    """A data handler that can return triplets in several forms given a
    list of triplets.
    
    Parameters
    ----------
    list_comparisons : numpy array, shape (n_comparisons, 3)
        A numpy array where each row indicates that the triplet
        (row[0],row[1],row[2]) is available.

    n_examples : int
        The number of examples handled.

    n_comparsions : int
        The number of unique triplets handled.

    Attributes 
    ---------- 
    comparisons : scipy coo matrix, shape (n_examples, (n_examples choose 2))
        A scipy coo_matrix containing values in {1,-1,0}. Given
        i!=j,k, j<k, in entry (i,self._get_k(j,k)), the value 1
        indicates that the triplet (i,j,k) is available, the value -1
        indicates that the triplet (i,k,j) is available, and the value
        0 indicates that neither of the triplets is available.


    n_examples : int
        The number of examples handled.

    n_comparisons : int
        The number of comparisons handled.

    """

    def __init__(self, list_comparisons, n_examples, n_comparisons):
        self.n_examples = n_examples

        columns = self._get_k(np.amin(list_comparisons[:, 1:], axis=1), np.amax(list_comparisons[:, 1:], axis=1))
        entries = np.where(list_comparisons[:, 1] < list_comparisons[:, 2], 1, -1)

        comparisons = coo_matrix((entries, (list_comparisons[:, 0], columns)),
                                 shape=(n_examples, (n_examples * (n_examples - 1)) // 2), dtype=int)
        comparisons.eliminate_zeros()

        super(ListTriplets, self).__init__(comparisons, n_comparisons, n_examples)


class OracleTriplets(HandlerTriplets):
    """An oracle that returns passively queried triplets given a
    similarity matrix.

    Parameters
    ----------
    similarities : numpy array, shape (n_examples, n_examples)
        A numpy array containing the similarities that should be used
        to generate the triplets.

    n_examples : int
        The number of examples handled by the oracle.

    n_triplets : int or float, optional
        If n_triplets is strictly greater than 1 it represent the
        number of comparisons that should be generated, if it is lower
        than one it represented the overall proportion of triplets
        that should be generated. (Default: 0.1).

    proportion_noise : float, optional
        The overall proportion of noise in the triplets (flipped
        comparisons). (Default: 0.0).

    seed : int or None
        The seed used to initialize the random number generators. If
        None the current time is used, that is
        int(time.time()). (Default: None).

    Attributes 
    ---------- 
    similarities : numpy array, shape (n_examples, n_examples)
        A numpy array containing the similarities that should be used
        to generate the triplets.

    comparisons : scipy coo matrix, shape (n_examples, (n_examples choose 2))
        A scipy coo_matrix containing values in {1,-1,0}. Given
        i!=j,k, j<k, in entry (i,self._get_k(j,k)), the value 1
        indicates that the triplet (i,j,k) is available, the value -1
        indicates that the triplet (i,k,j) is available, and the value
        0 indicates that neither of the triplets is available.

    n_comparisons : int
        The number of comparisons generated by the oracle.

    n_examples : int
        The number of examples handled by the oracle.

    proportion_noise : float
        The overall proportion of noise in the triplets (flipped
        comparisons).

    seed : int
        The seed used to initialize the random number generators.

    """

    def __init__(self, similarities, n_examples, n_triplets=0.1
                 , proportion_noise=0.0, seed=None):
        self.similarities = similarities

        self.n_examples = n_examples

        self.proportion_noise = proportion_noise

        if seed is not None:
            self.seed = seed
        else:
            self.seed = int(time.time())

        comparisons, n_comparisons = self._get_comparisons(n_examples, n_triplets)

        super(OracleTriplets, self).__init__(comparisons, n_comparisons, n_examples)

    def _get_comparisons(self, n_examples, n_comparisons):
        """Constructs the comparisons sparse matrix with n_comparisons
        triplets with a probability of noise equal to
        proportion_noise. This method is deterministic.

        """
        random_state = np.random.RandomState(self.seed)
        rd_random_state = rd.Random(self.seed + 42)  # To use a different seed than numpy

        n_pairs = ((n_examples - 1) * (n_examples - 2)) // 2

        if n_comparisons <= 1:
            n_comparisons = random_state.binomial(n_examples * n_pairs, n_comparisons)

        indices = np.array(rd_random_state.sample(range(n_examples * n_pairs), n_comparisons))
        i = indices // n_pairs
        j, k = self._get_ij(indices - i * n_pairs, n_examples - 1)
        j = np.where(j < i, j, j + 1)
        k = np.where(k < i, k, k + 1)

        noise = np.where(random_state.rand(n_comparisons) < self.proportion_noise, -1, 1)
        entries = np.multiply(np.where(self.similarities[i, j] > self.similarities[i, k], 1, 0) + np.where(
            self.similarities[i, j] < self.similarities[i, k], -1, 0), noise)

        columns = self._get_k(j, k)

        comparisons = coo_matrix((entries, (i, columns)), shape=(n_examples, (n_examples * (n_examples - 1)) // 2),
                                 dtype=int)
        comparisons.eliminate_zeros()

        return comparisons, n_comparisons
