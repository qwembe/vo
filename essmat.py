import numpy as np
from numpy.linalg import matrix_rank, svd, det
from scipy.linalg import null_space

"""
def createEssenceMatrix(cordsCameraA,cordsCameraB):
создает матрицу эссенции из двух наборов координат
они могут быть как 2д (преобразованные ф-ей transformCoords) так и локальные
возвращает матрицу эссенции если точки не копланарны и их колличество равно 9-ти
"""


def createEssenceMatrix(cordsCameraA, cordsCameraB):
    dot = 9
    semiEss = [[]] * dot
    for i in range(dot):
        kron_product = np.kron(cordsCameraA[i], cordsCameraB[i]).T
        semiEss[i] = kron_product
    semiEss = np.array(semiEss)
    rank = matrix_rank(semiEss)
    if (rank != 8):
        # print("dots dont correspond")
        # print("rank = ", rank)
        # assert ("dots dont correspond! ")
        raise IOError("dots dont correspond! rank =",rank)
    else:
        E = null_space(semiEss).reshape(3, 3)
        return E


"""
"Плохая" матрица поворота - это такая матрица, определитель которой равен -1
"""

def check_true_rot(R):
    return det(R)


"""
Разложение матрицы ессенции на 4 возможных варианта
"""


def e_decomposition(E_hat):
    Z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]])

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]).T

    u, s_, vt = svd(E_hat)
    E_norm = E_hat / s_[0]
    u, s, vt = svd(E_norm)

    B = u @ Z @ u.T
    x = np.array([B[1][2], B[2][0], B[0][1]])

    X = np.array([
        x,
        x,
        -x,
        -x
    ])

    # Occasionally rotation may be twisted, so it is beter to multiplay it on it's det in case of wrong rotation matrix
    Rt = np.array([
        (u @ W.T @ vt) * check_true_rot(u @ W.T @ vt),  # !
        (u @ W @ vt) * check_true_rot(u @ W @ vt),  # !
        (u @ W.T @ vt) * check_true_rot(u @ W.T @ vt),  # !
        (u @ W @ vt) * check_true_rot(u @ W @ vt),  # !
    ])

    return zip(Rt, X)
