import numpy as np
import scipy.stats as st


# X ~ TV
def fit_icm(X, W, rank, at = 10, av = 0.2, MAX_ITER=1000):

    [n,m] = X.shape

    # T
    if W is False:
        bt = 1.0
        At = at * np.ones((n, rank))
        Bt = bt * np.ones((n, rank))
        Ct = At/Bt
        T = np.random.gamma(at, bt / at, (n, rank))
    else:
        T = np.copy(W)

    # V
    bv = 1.0
    Av = av * np.ones((rank, m))
    Bv = bv * np.ones((rank, m))
    Cv = Av/Bv
    V = np.random.gamma(av, bv / av, (rank, m))

    M = np.ones(X.shape)
    KL = np.zeros(MAX_ITER)

    T_trans = T.transpose()
    for it in range(MAX_ITER):
        TV = np.dot(T,V)
        if W is False:
            V_trans = V.transpose()
        V = (Av + V * (np.dot(T_trans, (X / TV)))) / (Cv + np.dot(T_trans, M))
        if W is False:
            T = (At + T * (np.dot((X / TV), V_trans))) / (Ct + np.dot(M, V_trans))

        KL[it] = st.entropy(X.flat[:], np.dot(T, V).flat[:])

    # Normalize W:
    s = T.sum(axis=0)
    T = T/s

    return [T, V, KL]


def fit_nmf(x, rank, max_iter=1000):

    # For numerical stability
    X = np.copy(x)
    X[X == 0] = 1

    # get dimensions
    [n, m] = X.shape

    # initialize W and H
    T = np.random.dirichlet(np.ones(n), rank).transpose()
    V = np.random.dirichlet(np.ones(rank), m).transpose()
    KL = np.zeros(max_iter)

    for it in range(max_iter):

        # update H
        Zw = np.tile(T.sum(axis = 0, keepdims = True).transpose(), m)
        numerator = np.dot(T.transpose(), np.divide(X, np.dot(T, V)))
        V = np.multiply(V, np.divide(numerator, Zw))

        # update W
        Zh = np.tile(V.sum(axis = 1, keepdims = True), n).transpose()
        numerator = np.dot(np.divide(X, np.dot(T, V)), V.transpose())
        T = np.multiply(T, np.divide(numerator, Zh))

        # calculate D(V||WH)
        KL[it] = st.entropy(X.flat[:], np.dot(T, V).flat[:])

    # Normalize W:
    s = T.sum(axis=0)
    T = T/s
    V = V * s[:, np.newaxis]

    return T, V, KL