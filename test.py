import numpy as np
from qscg import *
from scipy import linalg
import matplotlib.pyplot as plt
import time


## GENERATING RANDOM HAMILTONIANS ##
def random_sparse_hamiltonian(n, d) -> np.ndarray:
    """Return a random d-sparse hamiltonian for n qubits."""
    N = 2**n
    m = np.random.randint(0, 2, size=(N, N)) # 0 or 1
    m = np.tril(m) + np.tril(m, -1).T
    for x in range(N):
        while sum(m[x]) > d:
            r = np.random.randint(0, N)
            while m[x, r % N] == 0:
                r = r + 1
            m[x, r % N] = 0
            m[r % N, x] = 0
    return m

def random_f(n, d):
    m = random_sparse_hamiltonian(n, d)
    print('Original Hamiltonian:')
    print(m)
    print('----')
    return hamiltonian_to_oracle(m)

def get_approx_unitary_func(ms, j):
    """Return an approximate unitary operator (as a function)
    
    :math:`e^{itH} \approx (e^{itH_1/j}...e^{itH_k/j})^j`

    Parameters
    ----------
    ms : list[np.ndarray]
        List of 1-sparse hamiltonians
    j : int
        Error-controlling parameter. Larger `j` means smaller error
    
    Returns
    -------
    function
        Takes a single parameter, `t`, and returns a `np.ndarray`

    """
    def U(t):
        single = np.linalg.multi_dot([linalg.expm(1j * t * m / j) for m in ms])
        return np.linalg.matrix_power(single, j)
    return U


## VISUALISATIONS ##
def show_matrix_difference(diff, max_difference = 0.2):
    """Simple way to visualise the difference between two arrays
    
    Parameters
    ----------
    diff : np.ndarray
        Matrix 1 - Matrix 2
    max_difference : float
        Determines colour scale in the plot
    
    """
    plt.imshow(diff.real, vmin = -max_difference, vmax = max_difference)
    plt.title('Difference in real part')
    plt.colorbar()
    plt.show()

    plt.imshow(diff.imag, vmin = -max_difference, vmax = max_difference)
    plt.title('Difference in complex part')
    plt.colorbar()
    plt.show()
    return



if __name__ == '__main__':
    # Decompose a random sparse hamiltonian
    np.random.seed(1)
    n = 3
    d = 3
    md = OneSparseDecomposer(random_f(n, d), d, n)  # lambda x, i: (x ^ (1 << i), 1), n, n)
    #md.print_edge_colorings()

    ms = md.decompose_hamiltonian()
    print('decomposed hamiltonians:')
    print(*ms, sep='\n\n')



    # Construct an approximate unitary operator
    j = 10
    t = 1
    original = md.original_hamiltonian()
    t0 = time.process_time()
    original_val = linalg.expm(1j * t * original)
    t1 = time.process_time()
    print('-------\n\n')
    print('time to calculate original:', t1-t0)

    print('')

    estimate = get_approx_unitary_func(ms, j)
    t0 = time.process_time()
    estimate_val = estimate(t)
    t1 = time.process_time()
    print('time to calculate estimate:', t1-t0)

    print('')

    print('DIFFERENCE\n\n')
    diff = original_val - estimate_val
    print(diff)
    show_matrix_difference(diff)
