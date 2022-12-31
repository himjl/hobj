import numpy as np
from PySS.src.newtonsolve import newtonsolve


def forwardfilter(I, sigE, xguess, sigsqguess, mu):
    '''
    forwardfilter is a helper function that implements the forward recursive
    filtering algorithm to estimate the learning state (hidden process) at
    trial k as the Gaussian random variable with mean x{k|k} (xhat) and
    SIG^2{k|k} (sigsq).

    :param I: [2xN]. dtype = int. [num_successes_v_time, total_possible]
    :param sigE: (). dtype = float. default variance of learning state process
    :param xguess: ().
    :param sigsqguess: ().
    :param mu: ().
    '''

    '''
    :param xhatold: x{k|k-1}, one-step prediction (equation A.6)*
    :param sigsqold: SIG^2{k|k-1}, one-step prediction variance (equation A.7)*
    :param xhat         x{k|k}, posterior mode (equation A.8)*
    :param sigsq        SIG^2{k|k}, posterior variance (equation A.9)*
    :param p            p{k|k}, observation model probability (equation 2.2)*
    :param N            vector of number correct at each trial
    :param Nmax         total number that could be correct at each trial
    :param K            total number of trials
    :param number_fail  saves the time steps if Newton's Method fails
    '''

    K = I.shape[1] # total number of trials
    N = I[0, :] # vector of number correct at each trial
    Nmax = I[1, :] # total number that could be correct at each trial

    # Initial conditions: use values from previous iteration
    xhat = [xguess]
    sigsq = [sigsqguess]
    number_fail = []

    xhatold = [0]
    sigsqold = [0]

    for k in range(2, K+1+1):
        # for each trial, compute estimates of the one-step prediction, the
        # posterior mode (using Newton's Method), and the posterior variance
        # (estimates from subject's POV)

        # Compute the one-step prediction estimate of mean and variance
        xhatold.append(xhat[k-1-1])
        sigsqold.append(sigsq[k-1-1] + np.square(sigE))

        # Use Newton's Method to compute the nonlinear posterior mode estimate
        xhat_next, flagfail = newtonsolve(mu, xhatold[k-1], sigsqold[k-1], N[k-1-1], Nmax[k-1-1])

        xhat.append(xhat_next)

        # if Newton's Method fails, number_fail saves the time step
        if flagfail > 0:
            number_fail.append(k-1)

        # Compute the posterior variance estimate
        denom = np.true_divide(-1, sigsqold[k-1]) - Nmax[k-1-1]*np.exp(mu)*np.exp(xhat[k-1])/np.square(1 + np.exp(mu) * np.exp(xhat[k-1]))
        sigsq.append(np.true_divide(-1, denom))

    if len(number_fail) > 0:
        print ('Newton convergence failed at times', number_fail)

    # Compute the observation model probability estimate
    p = np.true_divide(np.exp(mu) * np.exp(xhat), 1 + np.exp(mu) * np.exp(xhat))

    xhat = np.array(xhat)[None, :]
    return p, xhat, sigsq, xhatold, sigsqold
