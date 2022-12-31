import numpy as np


def newtonsolve(mu,  xold, sigoldsq, N, Nmax):

    '''

    newtonsolve is a helper function that implements Newton's Method in order
    to recursively estimate the posterior mode (x).  Once the subsequent estimates
    sufficiently converge, the function returns the last estimate.  If, having
    never met this convergence condition, the function goes through all of the
    recursions, then a special flag (timefail) - indicating the convergence
    failure - is returned along with the last posterior mode estimate.

    :param mu: float
    :param xold: float
    :param sigoldsq: float
    :param N: float
    :param Nmax: float
    :return:
    '''

    it = xold + sigoldsq * (N - Nmax * np.exp(mu) * np.exp(xold) / (1 + np.exp(mu) * np.exp(xold)))

    it = [it]

    g = []
    gprime = []
    niter = 100

    for i in range(niter):
        gupdate = xold + sigoldsq*(N - Nmax*np.exp(mu)*np.exp(it[i])/(1+np.exp(mu)*np.exp(it[i]))) - it[i]
        g.append(gupdate)

        gprime_update = -Nmax*sigoldsq*np.exp(mu)*np.exp(it[i])/np.square(1+np.exp(mu)*np.exp(it[i])) - 1
        gprime.append(gprime_update)

        it_update = it[i] - g[i]/gprime[i]
        it.append(it_update)

        x = it[-1]

        if np.abs(x - it[-2]) < 1e-8:
            timefail = 0
            return x, timefail

    # This tries a new initial condition if first Newtons doesn't work
    if i == (niter - 1):
        it = [-1]
        g = []
        gprime = []

        for i in range(niter):
            gupdate = xold + sigoldsq*(N - Nmax*np.exp(mu)*np.exp(it[i])/(1+np.exp(mu)*np.exp(it[i]))) - it[i]
            g.append(gupdate)

            gprime_update = -Nmax*sigoldsq*np.exp(mu)*np.exp(it[i])/np.square(1+np.exp(mu)*np.exp(it[i]))- 1
            gprime.append(gprime_update)

            it_update = it[i] - g[i] / gprime[i]
            it.append(it_update)

            x = it[-1]

            if np.abs(x - it[i]) < 1e-8:
                timefail = 0
                return x, timefail

    # This tries a new initial condition if second Newtons doesn't work
    if i == (niter - 1):
        it = [1]
        g = []
        gprime = []

        for i in range(niter):
            gupdate = xold + sigoldsq*(N - Nmax*np.exp(mu)*np.exp(it[i])/(1+np.exp(mu)*np.exp(it[i]))) - it[i]
            g.append(gupdate)

            gprime_update = -Nmax*sigoldsq*np.exp(mu)*np.exp(it[i])/np.square(1+np.exp(mu)*np.exp(it[i]))- 1
            gprime.append(gprime_update)

            it_update = it[i] - g[i] / gprime[i]
            it.append(it_update)

            x = it[-1]

            if np.abs(x - it[i]) < 1e-8:
                timefail = 0
                return x, timefail

    timefail = 1
    return x, timefail
