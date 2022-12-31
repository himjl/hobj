import numpy as np

def backwardfilter(x, xold, sigsq, sigsqold):
    '''
    backwardfilter is a helper function that implements the backward filter
    smoothing algorithm to estimate the learning state at trial k, given all
    the data, as the Gaussian random variable with mean x{k|K} (xnew) and
    SIG^2{k|K} (signewsq).

    :param x: (1 x (N(+1?))
    :param xold: (N+1). dtype = list
    :param sigsq: (N+1). dtype = list
    :param sigsqold: (N+1). dtype = list
    :return:
    '''

    T = x.shape[1] # total number of posterior mode estimates (K + 1)

    # Initial conditions: use values of posterior mode and posterior variance
    xnew = list(np.zeros(T-1)) + [x[0, -1]]
    signewsq = list(np.zeros(T-1)) + [sigsq[-1]]
    xnew = np.array(xnew)[None, :]
    signewsq = np.array(signewsq)[None, :]

    A = np.zeros((1, T-1))
    for i in range(T-1, 1, -1):
        A[0, i - 1] = np.true_divide(sigsq[i - 1], sigsqold[i])

        xnew_update = x[0, i - 1] + A[0, i - 1] * (xnew[0, i] - xold[i])
        xnew[0, i - 1] = xnew_update
        signewsq[0, i - 1] = sigsq[i - 1] + A[0, i - 1] * A[0, i - 1] * (signewsq[0, i] - sigsqold[i])

    return xnew, signewsq, A
