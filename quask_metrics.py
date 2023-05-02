
"""
Tools for analysing the performance of kernels produced in the TENT project against metrics from <Power of data in quantum machine learning>
The metrics used assess potential quantum advantage in learning tasks.
"""

import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm

def calculate_approximate_dimension(k):
    """
    NOTE: Copied from QuASK
    Calculate the approximate dimension (d), which is equation S111 in the Supplementary
    of the "The power of data in quantum machine learning"
    (https://www.nature.com/articles/s41467-021-22539-9).

    Args:
        k: Kernel gram matrix.

    Returns:
        approximate dimension of the given kernel (float).
    """
    u, t, udagger = la.svd(k, full_matrices=True)

    N = len(t)

    d = 0
    for i in range(N):
        d += 1 / (N - i) * sum(t[i:])
    return d

def calculate_geometric_difference(k_1, k_2, normalization_lambda=0.5e-7): #originally 0.001
    """
    NOTE: Copied from QuASK
    Calculate the geometric difference g(K_1 || K_2), which is equation F9 in
    "The power of data in quantum machine learning" (https://arxiv.org/abs/2011.01938)
    and characterize the separation between classical and quantum kernels.

    Args:
        k_1: Quantum kernel Gram matrix
        k_2: Classical kernel Gram matrix
        normalization_lambda: normalization factor, must be close to zero

    Returns:
        geometric difference between the two kernel functions (float).
    """
    n = k_2.shape[0]
    assert k_2.shape == (n, n)
    assert k_1.shape == (n, n)
    # √K1
    k_1_sqrt = np.real(sqrtm(k_1))
    # √K2
    k_2_sqrt = np.real(sqrtm(k_2))
    # √(K2 + lambda I)^-2
    kc_inv = la.inv(k_2 + normalization_lambda * np.eye(n))
    kc_inv = kc_inv @ kc_inv
    # Equation F9
    f9_body = k_1_sqrt.dot(k_2_sqrt.dot(kc_inv.dot(k_2_sqrt.dot(k_1_sqrt))))
    f9 = np.sqrt(la.norm(f9_body, np.inf))
    return f9

def calculate_model_complexity(k, y, normalization_lambda=0.5e-7): #originally 0.001
    """
    NOTE: Copied from QuASK
    Calculate the model complexity s(K), which is equation F1 in
    "The power of data in quantum machine learning" (https://arxiv.org/abs/2011.01938).

    Args:
        k: Kernel gram matrix
        y: Labels
        normalization_lambda: Normalization factor

    Returns:
        model complexity of the given kernel
    """
    n = k.shape[0]
    k_inv = la.inv(k + normalization_lambda * np.eye(n))
    k_body = k_inv @ k @ k_inv
    model_complexity = y.T @ k_body @ y
    return model_complexity