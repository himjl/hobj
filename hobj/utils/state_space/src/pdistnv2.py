import numpy as np

def pdistnv2(x, s, mu, background_prob):

    '''
    pdist is a helper function that calculates the confidence limits of the EM
    estimate of the learning state process.  For each trial, the function
    constructs the probability density for a correct response.  It then builds
    the cumulative density function from this and computes the p values of
    the confidence limits

    :param x: (1 x ntrials)
    :param s:
    :param mu: ()
    :param background_prob: ()
    :return:
    '''

    num_samps = 10000
    p025 = np.zeros(x.shape)
    p975 = np.zeros(x.shape)
    pmid = np.zeros(x.shape)
    pmodnew = np.zeros(x.shape) # not computed
    pcert = np.zeros(x.shape)

    for ov in range(1, x.shape[1]+1):

        xx = x[0, ov-1]
        ss = s[0, ov-1]
        samps = xx + np.sqrt(ss)*np.random.randn(num_samps)
        pr_samps = np.true_divide(np.exp(mu + samps), 1 + np.exp(mu + samps))

        order_pr_samps = np.sort(pr_samps)

        p025[0, ov - 1] = float(order_pr_samps[int(np.floor(0.025 * num_samps))])
        p975[0, ov - 1] = float(order_pr_samps[int(np.ceil(0.975 * num_samps))])
        pmid[0, ov - 1] = float(order_pr_samps[int(np.round(0.5 * num_samps))])
        pcert[0, ov - 1] = np.mean(pr_samps > background_prob)

    return p025, p975, pmid, pmodnew, pcert
