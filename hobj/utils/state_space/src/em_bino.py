import numpy as np

def em_bino(I, xnew, signewsq, A, startflag):
    '''
    em_bino is a helper function that computes sigma_eps squared (estimated
    learning process variance).
    :param I: (2 x ntrials)
    :param xnew: (1 x ntrials(+1?))
    :param signewsq: (1 x ntrials(+1?))
    :param A: (1 x ntrials)
    :param startflag: ()
    :return:
    '''

    '''
    print I.shape
    print xnew.shape
    print signewsq.shape
    print A.shape
    print startflag
    raise Exception
    '''

    M = xnew.shape[1]
    xnewt = xnew[0, 2:M]
    xnewtm1 = xnew[0, 1:M-1]
    signewsqt = signewsq[0, 2:M]
    A = A[0, 1:]

    covcalc = signewsqt * A

    term1 = np.sum(np.square(xnewt)) + np.sum(signewsqt)
    term2 = np.sum(covcalc) + np.sum(xnewt * xnewtm1)

    if startflag == 1:
        term3 = 1.5 * xnew[0, 1] * xnew[0, 1] + 2 * signewsq[0, 1]
        term4 = np.square(xnew[0, -1]) + signewsq[0, -1]
    elif startflag == 0:
        term3 = 2 * xnew[0, 1] * xnew[0, 1] + 2 * signewsq[0, 1]
        term4 = np.square(xnew[0, -1]) + signewsq[0, -1]
    elif startflag == 2:
        term3 = 1 * xnew[0, 1]*xnew[0, 1] + 2*signewsq[0, 1]
        term4 = np.square(xnew[0, -1]) + signewsq[0, -1]
        M = M - 1
    else:
        raise ValueError('startflag of %d not supported'%startflag)
    newsigsq = (2 * (term1 - term2) + term3 - term4) / float(M)

    return newsigsq