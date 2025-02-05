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
    # Параметризация в терминах углов и фаз
    params = {'thetas': [], 'phi_givens': [], 'phi_diag': []}

    w_SHAPE = W.shape[0]
    # Последовательное обнуление элементов
    # (1,2), (1,3), (1,4), (2,3), (2,4), (3,4))
    for col in range(w_SHAPE-1):
        for row in range(col+1, w_SHAPE):
            if abs(W[row, col]) > 1e-12:
                theta, phi = angle_and_phase(W[col, col], W[row, col])
                params['thetas'].append(theta)
                params['phi_givens'].append(phi)
                U = Gevins_zero(W, col, row, theta, phi)
    
    # Извлечение диагональных фаз
    diag_phases = [cmath.phase(U[i,i]) for i in range(w_SHAPE)]
    params['phi_diag'] = diag_phases
    
    return params

def assemble_wmesh(theta, phases):
    N = len(phases)
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

if __name__ == '__main__':
    # Создаем унитарную матрицу 4x4
    W = unitary_group.rvs(4)
    # Проверяем, что матрица унитарна
    W_unitary_check = np.dot(W, W.conj().T) # = свойство унитарной матрицы, когда UxU* = I
    W_unitary_check[W_unitary_check<1e-12] = 0
    if int(np.real(W_unitary_check.sum())) == 4:
        print("W унитарна")
        # можно добавить следующую строчку, чтобы убрать машинный 0
        # W[W<1e-12] = 0

        params = decompose(W)
        print("Углы θ:", params['thetas'])
        print("Фазы вращений φ:", params['phi_givens'])
        print("Диагональные фазы φ:", params['phi_diag'])

        Wmesh = assemble_wmesh(params['thetas'], params['phi_diag'])
        error = norm(W - Wmesh)
        print(f"Ошибка восстановления: {error:.4e}")

        ### test random noise
        np.random.seed(0)
        random_matrix = np.random.randn(4,4) + 1j*np.random.randn(4,4)
        W = W+random_matrix

        params = decompose(W)
        print("Углы θ:", params['thetas'])
        print("Фазы вращений φ:", params['phis_givens'])
        print("Диагональные фазы φ:", params['phis_diag'])

        Wmesh = assemble_wmesh(params['thetas'], params['phis_diag'])
        error = norm(W - Wmesh)
        print(f"Ошибка восстановления c доп шумом: {error:.4e}")

    else:
        raise ValueError('А не унитарная')
