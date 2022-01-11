import numpy as np
import scipy as sp



def bit_get(x, i):
    """Return the ith bit of the binary representation of a number."""
    return (x >> i) & 1

def hamiltonian_to_oracle(H):
    """Return an appropriate oracle function for an (adjacency matrix)
    hamiltonian.
    
    See `OneSparseDecomposer.__init__()` for exact details of the oracle.

    Paramaters
    ----------
    H : np.ndarray
        Square matrix hamiltonian
    
    Returns
    -------
    function

    Notes
    -----
    The actual function here appears to look at a row `x`, not a column `x`.
    This shouldn't matter though since the Hamiltonian must be Hermitian.

    """
    if len(H.shape) != 2:
        raise ValueError('Hamiltonian must be a matrix')
    if H.shape[0] != H.shape[1]:
        raise ValueError('Hamiltonian must be square')
    
    N = H.shape[0]

    def f(x, i):
        nonzero_count = 0
        for k in range(N):
            if H[x, k] != 0:
                if nonzero_count == i:
                    return k, H[x, k]
                nonzero_count += 1
        return x, 0

    return f



class Decomposer:
    pass


class OneSparseDecomposer:
    """
    https://arxiv.org/pdf/quant-ph/0508139.pdf
    """

    def __init__(self, f, d, n):
        """
        Parameters
        ----------
        f : function
            Oracle for a sparse (adjacency matrix) Hamiltonian `H`.

            Takes an input of `(x: int, i: int)` where `x` is a column in the
            Hamiltonian and `i` is an arbitrarily-assigned label (starting from
            0) for the row :math:`y_i` of one of the nonzero elements in that
            column.

            If `D'` is the number of nonzero elements in column `x`, then `f`
            should return :math:`(y_i, H_{x,y_i})` for :math:`i \le D'`, and
            :math:`(x,0)` for `i > D'`.
        d : int
            Decompose into O(d^2) 1-sparse Hamiltonians.
        n : int
            Number of qubits -> dimension of matrix is 2**n by 2**n.
        
        """
        self.f = f
        self.d = d
        self.n = n
        self.N = 2 ** n

    def zn(self):
        """Number of times we must iterate l -> 2*ceil(log2(l)) (starting at 
        2**n) to obtain 6 or less."""
        l = 2 ** self.n
        zn = 0
        while l > 6:
            l = 2 * np.ceil(np.log2(l)).astype(int)
            zn += 1
        return zn

    def _seq0(self, x, i, j):
        zn = self.zn()
        seq = [x]
        while len(seq) <= zn + 1:
            x = seq[-1]
            xn = self.f(x, i)[0]
            if x < xn:
                seq.append(xn)
            else:
                break
        return seq

    def _seq1(self, seq0, bits):
        idx_len = np.ceil(np.log2(bits)).astype(int)
        seq1 = []
        for x, xn in zip(seq0, seq0[1:]):
            c = bits - 1
            while bit_get(x, c) == bit_get(xn, c):
                c -= 1
            diff_bit = bits - c - 1
            seq1.append((bit_get(x, c) << idx_len) | diff_bit)
        seq1.append(bit_get(seq0[-1], bits - 1) << idx_len)
        return seq1, idx_len + 1

    def get_j(self, x, y):
        for j in range(self.d):
            if self.f(y, j)[0] == x:
                return j

    def get_uid(self, x, i, j):
        """Get a unique identifier - an additional parameter to ensure unique
        edge-labels."""
        seq = self._seq0(x, i, j)
        idx = self.n
        for i in range(self.zn()):
            seq, idx = self._seq1(seq, idx)
        return seq[0]

    def print_edge_colorings(self):
        """Print the unique label (i,j,uid) for each edge in the graph
        represented by the hamiltonian."""
        m = [['(x, x, x)'] * self.N for _ in range(self.N)]
        for x in range(self.N):
            for i in range(self.d):
                y = self.f(x, i)[0]
                j = self.get_j(x, y)
                m[x][y] = str((i, j, self.get_uid(x, i, j))) if x < y else str((j, i, self.get_uid(x, i, j)))
        for x in m:
            print('\t'.join(x))

    def g(self, x, i, j, uid):
        """Black-box function g(x, i, j, uid), which conditionally calls the
        oracle f. Where (i,j,uid) uniquely label an edge.

        Parameters
        ----------
        x : int
            Column of the hamiltonian(?)
        i : int
        j : int
        uid : int

        Returns
        -------
        tuple
            A 2-tuple obtained from conditionally calling the oracle, or `(x,0)`

        """
        fxi = self.f(x, i)
        if fxi[0] == x and i == j and uid == 0:
            return fxi
        if fxi[0] > x and self.f(fxi[0], j)[0] == x and self.get_uid(x, i, j) == uid:
            return fxi
        fxj = self.f(x, j)
        if fxj[0] < x and self.f(fxj[0], i)[0] == x and self.get_uid(fxj[0], i, j) == uid:
            return fxj
        return x, 0

    def decompose_hamiltonian(self):
        """Return a list of appropriate 1-sparse matrices. Excludes any empty
        matrices formed in the decomposition.
        
        Returns
        -------
        list[np.ndarray]
        
        """
        ms = [[[np.zeros((2**self.n, 2**self.n)).astype(int) for _ in range(6)] 
                for _ in range(self.d)] 
                for _ in range(self.d)]

        for x in range(self.N):
            for i in range(self.d):
                for j in range(self.d):
                    for uid in range(6):
                        y, h = self.g(x, i, j, uid)
                        ms[i][j][uid][x][y] = h

        ms = [item for sublist in ms for item in sublist]
        ms = [item for sublist in ms for item in sublist]

        # Exclude empty matrices
        zeros = np.zeros((self.N, self.N)).astype(int)
        ms_nonzero = [m for m in ms if not np.array_equal(m, zeros)]
        return ms_nonzero

    def original_hamiltonian(self):
        """Return the original Hamiltonian produced by the oracle."""
        #TODO: reconstruct directly from f rather than decompose first (as 
        # this assumes decomposition works).
        ms = self.decompose_hamiltonian()
        return np.sum(ms, axis=0)


    # Not used (not sure what these are for)
    def _lt2(self, hs, m, l):
        return np.product(
            sp.linalg.expm(hs[i] * l / 2) * np.product(sp.linalg.expm(hs[j] * l / 2) for j in range(m, step=-1))
            for i in range(m))

    def lt(self, hs, m, l, k):
        if k == 2:
            return self._lt2(hs, m, l)
        pk = np.power(4 - np.power(4, 1.0 / (2 * k - 1)), -1)
        return np.power(self.lt(hs, m, pk * l, 2 * k - 2), 2) * np