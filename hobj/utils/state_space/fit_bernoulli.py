# Adapted from code written by Anne Smith in 2003 from the Brown Lab at MIT
import numpy as np
from tqdm import tqdm
from PySS.src.forwardfilter import forwardfilter
from PySS.src.backwardfilter import backwardfilter
from PySS.src.em_bino import em_bino
from PySS.src.pdistnv2 import pdistnv2

def fit(Responses,
                  MaxResponse = 1,
                  chance_prior_type = 'fixed',
                  disable = True
                  ):
    '''

    :param Responses: sequence of 0s or 1s. dtype = list, np.array. shape = (1, N)
    :param MaxResponse: number of possible correct per trial. Can be a vector.
    :param chance_prior_type one of 'fixed', 'update', 'none', : dtype: str
    :return: pmid, p025, p075
    '''

    if chance_prior_type not in ['fixed', 'none', 'update']:
        raise ValueError('Specify chance_prior_type of "fixed", "update", or "none". Gave: %s'%(str(chance_prior_type)))

    UpdaterFlagDict = {'fixed': 0, 'update': 1, 'none': 2, }


    BackgroundProb = 0.5
    UpdaterFlag = UpdaterFlagDict[chance_prior_type]

    SigE = 0.5  # default variance of learning state process is sqrt(0.5)

    Responses, MaxResponse = _check_input(Responses, MaxResponse)
    I = np.concatenate([Responses, MaxResponse * np.ones(Responses.shape)], axis=0)

    SigsqGuess = np.square(SigE)

    # set the value of mu from the chance of correct
    mu = np.log(np.true_divide(BackgroundProb, 1 - BackgroundProb))

    # convergence criterion for SIG_EPSILON^2
    CvgceCrit = 1e-5

    ''' Start '''
    xguess = 0.
    NumberSteps = 3000

    newsigsq = []
    xnew1save = []
    for i in tqdm(range(NumberSteps), disable = disable):
        p, x, s, xold, sold = forwardfilter(I, SigE, xguess, SigsqGuess, mu)

        # Compute the backward (smoothing algorithm) estimates of the learning
        # state and its variance: x{k|K} and sigsq{k|K}
        xnew, signewsq, A = backwardfilter(x, xold, s, sold)
        # xnew: (1, ntrials+1)
        # signewsq: (1, ntrials+1)
        # A: (1, ntrials)

        if chance_prior_type == 'fixed':
            # UpdaterFlag == 0
            xnew[0, 0] = 0 # fixes initial value to 50%
            signewsq[0, 0] = np.square(SigE)
        elif chance_prior_type == 'update':
            # Dampens the initial value toward chance
            xnew[0, 0] = 0.5 * xnew[0, 1]  # updates the initial value of the latent process
            signewsq[0, 0] = np.square(SigE)
        elif chance_prior_type == 'none':
            # UpdaterFlag == 2
            # Unconstrained initial value
            xnew[0, 0] = xnew[0, 1]  # x[0] = x[1] means no prior chance probability
            signewsq[0, 0] = signewsq[0, 1]

        # Compute the EM estimate of the learning state process variance
        newsigsq.append(em_bino(I, xnew, signewsq, A, UpdaterFlag))

        xnew1save.append(xnew[0, 0])

        # Check for convergence
        if i > 0:
            a1 = np.abs(newsigsq[i] - newsigsq[i-1])
            a2 = np.abs(xnew1save[i] - xnew1save[i-1])

            if (a1 < CvgceCrit) and (a2 < CvgceCrit) and (UpdaterFlag >= 1):
                if not disable:
                    print ('EM estimates of learning state process variance and start point converged after %d steps'%(i+1))
                break
            elif (a1 < CvgceCrit) and (UpdaterFlag == 0):
                if not disable:
                    print ('EM estimates of learning state process variance converged after %d steps'%(i+1))
                break

        SigE = np.sqrt(newsigsq[i])
        xguess = xnew[0, 0]
        SigsqGuess = signewsq[0, 0]

    if i == (NumberSteps-1):
        print( 'Failed to converge after %d steps; convergence criterion was %f'%(i+1, CvgceCrit))

    # Use sampling to convert from state to a probability

    # Can used smoothed or filtered estimates - here used smoothed

    if chance_prior_type == 'fixed':
        # UpdaterFlag == 0
        xnew[0, 0] = 0  # fixes initial value to 50%

    p025, p975, pmid, _, pcert = pdistnv2(xnew, signewsq, mu, BackgroundProb);

    # Remove the prior
    p025 = p025[0, :-1]
    p975 = p975[0, :-1]
    pmid = pmid[0, :-1]

    return pmid, p025, p975

# Helper functions
def _check_input(Responses, MaxResponse):

    Responses = np.array(Responses)
    if isinstance(MaxResponse, (np.ndarray, list)):
        MaxResponse = np.array(MaxResponse)
        if (MaxResponse.shape) == 1:
            MaxResponse = MaxResponse[None, :]
        if MaxResponse.shape[0] !=1:
            MaxResponse = np.transpose(MaxResponse)
        assert MaxResponse.shape[0] == 1



    if len(Responses.shape) == 1:
        Responses = Responses[None, :]

    if Responses.shape[0] != 1:
        Responses = np.transpose(Responses)

    assert Responses.shape[0] == 1
    return Responses, MaxResponse
