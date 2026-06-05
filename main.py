import numpy as np
import cmath
from scipy.linalg import norm
from scipy.stats import unitary_group

def angle_and_phase(a,b):
    theta = np.atan2(abs(b), abs(a))
    phi = np.angle(b) - np.angle(a)
    return theta, phi

def Gevins_zero(U, i, j, theta, phi):
    #U21,U31,U41 || U32,U42, ||U43
    
    ################################################################
    ### Применяет вращение Гивенса к матрице U в плоскости (i,j).###
    ################################################################

    G = np.eye(U.shape[0], dtype=complex)

    G[i,i] = np.cos(theta) * np.exp(1j*phi)
    G[i,j] = -np.sin(theta) * np.exp(1j*phi)
    G[j,i] = np.sin(theta) * np.exp(-1j*phi)
    G[j,j] = np.cos(theta) * np.exp(-1j*phi)
    return G @ U

def decompose(W):
    params = {'thetas': [], 'phi_givens': [], 'phi_diag': []}

    w_SHAPE = W.shape[0]
    U = W.copy()  # работаем с копией

    for col in range(w_SHAPE-1):
        for row in range(col+1, w_SHAPE):
            if abs(U[row, col]) > 1e-12:
                theta, phi = angle_and_phase(U[col, col], U[row, col])
                params['thetas'].append((theta, phi, col, row))  # см. п.2
                params['phi_givens'].append(phi)
                U = Gevins_zero(U, col, row, theta, phi)

    diag_phases = [cmath.phase(U[i, i]) for i in range(w_SHAPE)]
    params['phi_diag'] = diag_phases

    return params

def assemble_wmesh(theta, phases):
    N = len(phases)
    phases = np.array(phases)          # ← добавляем
    D = np.diag(np.exp(1j * phases))
    W = D.copy()
    for theta, phi, lr, tr in reversed(theta):
        T_H = np.eye(N, dtype=complex)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        T_H[lr, lr] = cos_theta
        T_H[lr, tr] = sin_theta
        T_H[tr, lr] = -np.exp(-1j*phi) * sin_theta
        T_H[tr, tr] = np.exp(-1j*phi) * cos_theta
        W = T_H @ W
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
