from __future__ import print_function
import numpy as np
from scipy.optimize import curve_fit, minimize
from sklearn.decomposition import FactorAnalysis
from copy import deepcopy
import random

def invert(A, d):
    k = A.shape[1]
    A = np.array(A)
    d_vec = np.array(d)
    d_inv = np.array(1 / d_vec[:, 0])

    inv_d_squared = np.dot(np.atleast_2d(d_inv).T, np.atleast_2d(d_inv))
    M = np.diag(d_inv) - inv_d_squared * np.dot(
        np.dot(A, np.linalg.inv(np.eye(k, k) + np.dot(A.T, np.dot(np.diag(d_inv), A)))), A.T)

    return M


def Estep(Y, A, mus, sigmas, lamb):
    N, D = Y.shape
    D, K = A.shape
    EX = np.zeros([N, D])
    EXZ = np.zeros([N, D, K])
    EX2 = np.zeros([N, D])
    EZ = np.zeros([N, K])
    EZZT = np.zeros([N, K, K])
    for i in range(N):
        Y_i = Y[i, :]
        Y_is_zero = np.abs(Y_i) < 1e-6
        dim = K + D
        zero_ind=np.array([np.abs(Y_i[j]) < 1e-6 for j in range(D)])
        Y_i_notzero = Y_i[~Y_is_zero]

        mu_x = np.zeros([dim, 1])
        mu_x[K:dim, :] = mus
        A_exten = np.matrix(np.zeros([dim, K]))
        A_exten[:K, :] = np.eye(K)
        A_exten[K:, :] = A
        mu_x = np.atleast_2d(mu_x)
        Y_i_notzero = np.atleast_2d(Y_i_notzero)

        if len(Y_i_notzero) == 1:
            Y_i_notzero = Y_i_notzero.transpose()

        mu_diff = np.matrix(np.atleast_2d(Y_i_notzero - mus[~zero_ind]))

        zero_indices_exten = np.array([True for a in range(K)] + list(zero_ind))
        A_exten_0 = A_exten[zero_indices_exten, :]
        A_exten_non0 = A_exten[~zero_indices_exten, :]
        sigma_11 = A_exten_0 * A_exten_0.T
        sigma_11[K:, K:] = sigma_11[K:, K:] + np.diag(sigmas[zero_ind][:, 0] ** 2)
        D_exten = np.array([0 for i in range(K)] + list(sigmas[zero_ind][:, 0] ** 2))

        if len(Y_i_notzero) == 0:
            sigma_x = A_exten * A_exten.T
            sigma_x[K:, K:] = sigma_x[K:, K:] + np.diag(sigmas[zero_ind][:, 0] ** 2)
            mu_c = np.array(mu_x)
            sigma_c = np.array(sigma_x)
            sigma_22_inv = np.array([[]])
        else:
            sigma_22_inv = np.matrix(invert(A[~zero_ind, :], sigmas[~zero_ind] ** 2))
            mu_0 = mu_x[zero_indices_exten, :] + A_exten_0 * (A_exten_non0.T * (sigma_22_inv * mu_diff))
            sigma_0 = sigma_11 - A_exten_0 * (A_exten_non0.T * sigma_22_inv * A_exten_non0) * A_exten_0.T
            mu_c = np.array(mu_0)
            sigma_c = np.array(sigma_0)
        A_0 = np.matrix(A[zero_ind, :])
        A_non0 = np.matrix(A[~zero_ind, :])
        sigmas_0 = sigmas[zero_ind]

        E_xz = sigma_c[K:, :][:, :K]
        E_00_prime_inv = np.matrix(invert(A_0, sigmas_0 ** 2 + 1 / (2. * lamb)))
        E_plusplus_inv = sigma_22_inv
        E_0non0 = A_0 * A_non0.T

        if (E_plusplus_inv.shape[0] == 0) or (E_plusplus_inv.shape[1] == 0):
            inv_matrix = (1 / (2. * lamb)) * E_00_prime_inv

        elif (A_0.shape[0] < A_0.shape[1]):
            inv_matrix = np.linalg.inv(
                2. * lamb * (np.linalg.inv(E_00_prime_inv) - E_0non0 * E_plusplus_inv * E_0non0.T))

        else:
            b_inv = np.linalg.inv((np.matrix(A_0).T * E_00_prime_inv) * np.matrix(A_0))
            innermost_inverse = np.matrix(-E_plusplus_inv) - (
                    np.matrix(-E_plusplus_inv) * np.matrix(A_non0)) * np.linalg.inv(
                b_inv + np.matrix(A_non0).T * (np.matrix(-E_plusplus_inv) * np.matrix(A_non0))) * (
                                        np.matrix(-E_plusplus_inv) * np.matrix(A_non0)).T
            inv_matrix = (1 / (2. * lamb)) * (
                    E_00_prime_inv - (E_00_prime_inv * A_0) * (A_non0.T * innermost_inverse * A_non0) * (
                    A_0.T * E_00_prime_inv))

        dim = len(sigma_c)
        M = np.zeros([dim, dim])
        M[:K, :K] = np.eye(K)
        M[K:, :K] = -2 * lamb * np.dot(inv_matrix, E_xz)
        M[K:, K:] = inv_matrix
        matrix = np.array(M)
        if (Y_is_zero).sum() < D:
            magical_matrix = 2 * lamb * (np.dot(np.diag(D_exten), matrix) + A_exten_0 * (
                    np.eye(K) - A_exten_non0.T * sigma_22_inv * A_exten_non0) * (A_exten_0.T * matrix))
        else:
            magical_matrix = 2 * lamb * (
                    np.dot(np.diag(D_exten), matrix) + A_exten_0 * (A_exten_0.T * matrix))
        magical_matrix[:, :K] = 0

        if (Y_is_zero).sum() < D:
            sigma_xz = np.array(
                sigma_c - np.dot(np.array(magical_matrix), np.diag(D_exten)) - (magical_matrix * A_exten_0) * (
                        (np.eye(K) - A_exten_non0.T * sigma_22_inv * A_exten_non0) * A_exten_0.T))
        else:
            sigma_xz = np.array(sigma_c - np.dot(np.array(magical_matrix), np.diag(D_exten)) - (
                    magical_matrix * A_exten_0) * A_exten_0.T)

        mu_xz = np.array(np.matrix(np.eye(dim) - magical_matrix) * np.matrix(mu_c))

        EZ[i, :] = mu_xz[:K, 0]
        EX[i, Y_is_zero] = mu_xz[K:, 0]
        EX2[i, Y_is_zero] = mu_xz[K:, 0] ** 2 + np.diag(sigma_xz[K:, K:])
        EZZT[i, :, :] = np.dot(np.atleast_2d(mu_xz[:K, :]), np.atleast_2d(mu_xz[:K, :].transpose())) + sigma_xz[:K, :K]
        EXZ[i, Y_is_zero, :] = np.dot(mu_xz[K:], mu_xz[:K].transpose()) + sigma_xz[K:, :K]
    return EZ, EZZT, EX, EXZ, EX2


def Mstep(Y, EZ, EZZT, EX, EXZ, EX2, old_lamb, singleSigma=False):
    N, D = Y.shape
    N, K = EZ.shape
    Y_is_zero = np.abs(Y) < 1e-6

    B = np.eye(K + 1)
    for k1 in range(K):
        for k2 in range(K):
            B[k1][k2] = sum(EZZT[:, k1, k2])
        B[K, :K] = EZ.sum(axis=0)
        B[:K, K] = EZ.sum(axis=0)

    B[K, K] = N

    tiled_EZ = np.tile(np.resize(EZ, [N, 1, K]), [1, D, 1])
    tiled_Y = np.tile(np.resize(Y, [N, D, 1]), [1, 1, K])
    tiled_Y_is_zero = np.tile(np.resize(Y_is_zero, [N, D, 1]), [1, 1, K])

    c = np.zeros([K + 1, D])
    c[K, :] += (Y_is_zero * EX + (1 - Y_is_zero) * Y).sum(axis=0)
    c[:K, :] = (tiled_Y_is_zero * EXZ + (1 - tiled_Y_is_zero) * tiled_Y * tiled_EZ).sum(axis=0).transpose()

    solution = np.dot(np.linalg.inv(B), c)
    A = solution[:K, :].transpose()
    mus = np.atleast_2d(solution[K, :]).transpose()

    EM = np.zeros([N, D])
    EM2 = np.zeros([N, D])

    tiled_mus = np.tile(mus.transpose(), [N, 1])
    tiled_A = np.tile(np.resize(A, [1, D, K]), [N, 1, 1])

    EXM = (tiled_A * EXZ).sum(axis=2) + tiled_mus * EX
    test_sum = (tiled_A * tiled_EZ).sum(axis=2)
    A_product = np.tile(np.reshape(A, [1, D, K]), [K, 1, 1]) * (np.tile(np.reshape(A, [1, D, K]), [K, 1, 1]).T)

    for i in range(N):
        EM[i, :] = (np.dot(A, EZ[i, :].transpose()) + mus.transpose())
        EZZT_tiled = np.tile(np.reshape(EZZT[i, :, :], [K, 1, K]), [1, D, 1])
        ezzt_sum = (EZZT_tiled * A_product).sum(axis=2).sum(axis=0)
        EM2[i, :] = ezzt_sum + 2 * test_sum[i, :] * tiled_mus[i, :] + tiled_mus[i, :] ** 2

    sigmas = (Y_is_zero * (EX2 - 2 * EXM + EM2) + (1 - Y_is_zero) * (Y ** 2 - 2 * Y * EM + EM2)).sum(axis=0)
    sigmas = np.atleast_2d(np.sqrt(sigmas / N)).transpose()

    if singleSigma:
        sigmas = np.mean(sigmas) * np.ones(sigmas.shape)

    lamb = minimize(lambda x: lam(x, Y, EX2), old_lamb, jac=True, bounds=[[1e-8, np.inf]])
    lamb = lamb.x[0]

    return A, mus, sigmas, lamb


def lam(x, Y, EX2):
    y_squared = Y ** 2
    Y_is_zero = np.abs(Y) < 1e-6
    exp_Y_squared = np.exp(-x * y_squared)
    log_exp_Y = np.nan_to_num(np.log(1 - exp_Y_squared))
    exp_ratio = np.nan_to_num(exp_Y_squared / (1 - exp_Y_squared))
    obj = sum(sum(Y_is_zero * (-EX2 * x) + (1 - Y_is_zero) * log_exp_Y))
    grad = sum(sum(Y_is_zero * (-EX2) + (1 - Y_is_zero) * y_squared * exp_ratio))
    if type(obj) is np.float64:
        obj = -np.array([obj])
    if type(grad) is np.float64:
        grad = -np.array([grad])

    return obj, grad


def exp_lam(x, lamb):
    return np.exp(-lamb * (x ** 2))


def initializing(Y, K, singleSigma=False):
    N, D = Y.shape
    model = FactorAnalysis(n_components=K)
    zeroedY = deepcopy(Y)
    mus = np.zeros([D, 1])

    for j in range(D):
        mus[j] = zeroedY[:, j].mean()
        zeroedY[:, j] = zeroedY[:, j] - mus[j]

    model.fit(zeroedY)

    A = model.components_.transpose()
    sigmas = np.atleast_2d(np.sqrt(model.noise_variance_)).transpose()
    if singleSigma:
        sigmas = np.mean(sigmas) * np.ones(sigmas.shape)

    means = []
    ps = []
    for j in range(D):
        non_zero_idxs = np.abs(Y[:, j]) > 1e-6
        means.append(Y[non_zero_idxs, j].mean())
        ps.append(1 - non_zero_idxs.mean())

        lamb, pcov = curve_fit(exp_lam, means, ps, p0=.05)
        lamb = lamb[0]

    return A, mus, sigmas, lamb


def fitModel(Y, K, singleSigma=False):
    Y = deepcopy(Y)

    A, mus, sigmas, lamb = initializing(Y, K, singleSigma=singleSigma)
    max_iter = 100
    param_change_thresh = 1e-2
    n_iter = 0

    while n_iter < max_iter:

        EZ, EZZT, EX, EXZ, EX2 = Estep(Y, A, mus, sigmas, lamb)
        new_A, new_mus, new_sigmas, new_lamb = Mstep(Y, EZ, EZZT, EX, EXZ, EX2, lamb,
                                                           singleSigma=singleSigma)

        paramsNotChanging = True
        max_param_change = 0

        for new, old in [[new_A, A], [new_mus, mus], [new_sigmas, sigmas], [new_lamb, lamb]]:
            rel_param_change = np.mean(np.abs(new - old)) / np.mean(np.abs(new))

            if rel_param_change > max_param_change:
                max_param_change = rel_param_change

            if rel_param_change > param_change_thresh:
                paramsNotChanging = False
                break

        A = new_A
        mus = new_mus
        sigmas = new_sigmas
        lamb = new_lamb

        if paramsNotChanging:
            break
        n_iter += 1

    EZ, EZZT, EX, EXZ, EX2 = Estep(Y, A, mus, sigmas, lamb)
    params = {'A': A, 'mus': mus, 'sigmas': sigmas, 'lambda': lamb}

    return EZ, params, A, mus, sigmas, lamb


def Simulation(n, d, k, sigma, lamb):
    mu = 3
    Z = np.random.multivariate_normal(mean=np.zeros([k, ]), cov=np.eye(k), size=n).transpose()
    A = np.random.random([d, k])
    mu = np.array([(np.random.uniform()) * mu for i in range(d)])
    sigmas = np.array([(np.random.uniform()) * sigma for i in range(d)])
    noise = np.zeros([d, n])

    for j in range(d):
        noise[j, :] = mu[j] + np.random.normal(loc=0, scale=sigmas[j], size=n)

    X = (np.dot(A, Z) + noise).transpose()
    Y = deepcopy(X)
    Y[Y < 0] = 0
    rand_matrix = np.random.random(Y.shape)

    cutoff = np.exp(-lamb * (Y ** 2))
    zero_mask = rand_matrix < cutoff
    Y[zero_mask] = 0

    return X, Y, Z.transpose(), A, mu, sigmas

if __name__ == '__main__':
    n = 100
    d = 10
    k = 3
    sigma = .1
    n_clusters = 1
    lamb_true = .1

    X, Y, Z, A_true, mu_true, sigmas_true = Simulation(n, d, k, sigma, lamb_true)

    Z, params, A, mu, sigma, lamb = fitModel(Y, k)
    err_A = 0
    err_mu = 0
    err_sigma = 0
    err_lambda = 0
    for i in range(0, A.shape[0]):
        err_mu = err_mu + (mu[i] - mu_true[i]) * (mu[i] - mu_true[i])
        err_sigma = err_sigma + (sigma[i] - sigmas_true[i]) * (sigma[i] - sigmas_true[i])
        for j in range(0, A.shape[1]):
            err_A = err_A + (A[i][j] - A_true[i][j]) * (A[i][j] - A_true[i][j])
    err_lambda = (lamb - lamb_true) * (lamb - lamb_true)
    error = {'error of A': np.sqrt(err_A) / (A.shape[0] + A.shape[1]), 'error of mus': np.sqrt(err_mu) / A.shape[0],
             'error of sigmas': np.sqrt(err_sigma) / A.shape[0], 'error of lmabda': np.sqrt(err_lambda)}
    print(error)
