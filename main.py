import numpy as np
import cmath
from scipy.linalg import norm
from scipy.stats import unitary_group

def angle_and_phase(a, b):
    if abs(b) < 1e-15:
        return 0.0, 0.0

    theta = np.atan2(abs(b), abs(a))
    phi = np.pi + np.angle(a) - np.angle(b)
    return theta, phi

def Gevins_zero(U, i, j, theta, phi):
    """
    Применяет комплексное Givens-вращение в плоскости (i,j)
    в каноничной форме:
        [[ c,   -e^{i phi} s],
         [ s,    e^{i phi} c]]
    слева к матрице U.
    """
    N = U.shape[0]
    G = np.eye(N, dtype=complex)

    c = np.cos(theta)
    s = np.sin(theta)
    e = np.exp(1j * phi)

    G[i, i] = c
    G[i, j] = -e * s
    G[j, i] = s
    G[j, j] = e * c

    return G @ U

def givens_block(theta, phi):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.exp(1j * phi)
    return np.array([[c,        -e * s],
                     [s,   e * c     ]], dtype=complex)

def givens_block_dagger(theta, phi):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.exp(-1j * phi)
    return np.array([[c,        s       ],
                     [-e * s,   e * c   ]], dtype=complex)

def decompose(W):
    N = W.shape[0]
    U = W.copy()

    thetas = []
    phi_givens = []

    # Проход по поддиагональным элементам
    for col in range(N - 1):
        for row in range(col + 1, N):
            if abs(U[row, col]) > 1e-12:
                theta, phi = angle_and_phase(U[col, col], U[row, col])
                thetas.append((theta, phi, col, row))
                phi_givens.append(phi)

                # строим G† и умножаем справа: U = U @ G†
                Gd = np.eye(N, dtype=complex)
                Gb = givens_block_dagger(theta, phi)

                i, j = col, row
                Gd[i, i] = Gb[0, 0]
                Gd[i, j] = Gb[0, 1]
                Gd[j, i] = Gb[1, 0]
                Gd[j, j] = Gb[1, 1]

                U = U @ Gd

    phi_diag = [cmath.phase(U[i, i]) for i in range(N)]

    return {
        'thetas': thetas,
        'phi_givens': phi_givens,
        'phi_diag': phi_diag,
    }

def assemble_from_dagger(thetas, phases):
    N = len(phases)
    D = np.diag(np.exp(1j * np.array(phases)))
    W = D.copy()

    for theta, phi, i, j in thetas:
        c = np.cos(theta)
        s = np.sin(theta)
        e = np.exp(-1j * phi)

        G_dag = np.eye(N, dtype=complex)
        G_dag[i, i] = c
        G_dag[i, j] = s
        G_dag[j, i] = -e * s
        G_dag[j, j] = e * c

        W = G_dag @ W

    return W

def fmt(x, ndigits=6):
    return f"{float(x):.{ndigits}f}"

def print_params(params):
    print("Givens-параметры:")
    for k, (theta, phi, i, j) in enumerate(params['thetas'], start=1):
        print(
            f"  #{k:02d}  plane=({i},{j})  "
            f"theta={fmt(theta)} rad  phi={fmt(phi)} rad"
        )

    print("\nДиагональные фазы:")
    for i, phi in enumerate(params['phi_diag']):
        print(f"  d[{i}] = exp(i*{fmt(phi)})")

if __name__ == '__main__':
    W = unitary_group.rvs(4)
    W_unitary_check = W @ W.conj().T

    if np.allclose(W_unitary_check, np.eye(4), atol=1e-12):
        print("=== Исходная матрица ===")
        print("W унитарна")

        params = decompose(W)
        print_params(params)

        Wmesh = assemble_wmesh(params['thetas'], params['phi_diag'])
        error = norm(W - Wmesh)
        print(f"\nОшибка восстановления: {error:.4e}")

        np.random.seed(0)

        W0 = unitary_group.rvs(4)
        sigma = 1e-4
        noise = sigma * (np.random.randn(4, 4) + 1j * np.random.randn(4, 4))
        W_noisy = W0 + noise

        print("\n=== Шумная матрица ===")
        params = decompose(W_noisy)
        print_params(params)

        Wmesh = assemble_wmesh(params['thetas'], params['phi_diag'])
        error = norm(W_noisy - Wmesh)
        print(f"\nОшибка восстановления для шумной матрицы: {error:.4e}")

        print("\n=== Шумная, но снова унитарная матрица ===")
        np.random.seed(0)

        W0 = unitary_group.rvs(4)
        sigmas = [1e-8, 1e-6, 1e-4, 1e-2]

        for sigma in sigmas:
            noise = sigma * (np.random.randn(4, 4) + 1j*np.random.randn(4, 4))
            A = W0 + noise

            Q, R = np.linalg.qr(A)
            D = np.diag(np.exp(1j * np.angle(np.diag(R))))
            W_test = Q @ D

            params = decompose(W_test)
            Wmesh = assemble_wmesh(params['thetas'], params['phi_diag'])

            rec_error = norm(W_test - Wmesh)
            unitary_error = norm(W_test @ W_test.conj().T - np.eye(4))

            print(f"sigma={sigma:.0e} | rec_error={rec_error:.4e} | unitary_error={unitary_error:.4e}")
    else:
        raise ValueError("А не унитарная")
