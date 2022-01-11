import numpy as np
from qscg import *


def random_hamiltonian(n, d) -> np.ndarray:
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
    m = random_hamiltonian(n, d)
    print('Original Hamiltonian:')
    print(m)
    print('----')
    return hamiltonian_to_oracle(m)


if __name__ == '__main__':
    np.random.seed(1)
    n = 3
    d = 4
    md = OneSparseDecomposer(random_f(n, d), d, n)  # lambda x, i: (x ^ (1 << i), 1), n, n)
    md.print_edge_colorings()

    ms = md.decompose_hamiltonian()
    print(*ms, sep='\n\n')

    print("\n\n")
    print(md.original_hamiltonian())
