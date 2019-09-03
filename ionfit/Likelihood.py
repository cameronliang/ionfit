"""
I can optimize the code by removing some of the if statements here. 
like h1 - in prior. and which prior, 4 or 3 parameters.  
"""
import numpy as np

from IonizationFitting import ion_model

np.seterr(all='ignore')  # ignore floating points warnings.


def tophat_prior(model_x, x_left, x_right):
    if model_x >= x_left and model_x < x_right:
        return 0
    else:
        return -np.inf


def photo_model_lnprior(alpha, config):
    lognH, logZ, logNHI = alpha

    total_prior = (tophat_prior(lognH, config.min_lognH, config.max_lognH) +
                   tophat_prior(logZ, config.min_logZ,  config.max_logZ) +
                   tophat_prior(logNHI, config.min_logNHI, config.max_logNHI) +
                   obs_data.log_pdf['h1'](logNHI))
    return total_prior


def photo_model_aUV_lnprior(alpha, config):
    lognH, logZ, aUV, logNHI = alpha

    total_prior = (tophat_prior(lognH, config.min_lognH, config.max_lognH) +
                   tophat_prior(logZ, config.min_logZ,   config.max_logZ) +
                   tophat_prior(aUV, config.min_aUV,   config.max_aUV) +
                   tophat_prior(logNHI, config.min_logNHI,  config.max_logNHI) +
                   config.log_pdf['h1'](logNHI)) 	
    return total_prior


def photo_collision_model_lnprior(alpha, config):
    lognH, logZ, logT, logNHI = alpha

    total_prior = (tophat_prior(lognH, config.min_lognH, config.max_lognH) +
                   tophat_prior(logT, config.min_logT, config.max_logT) +
                   tophat_prior(logZ, config.min_logZ, config.max_logZ) +
                   tophat_prior(logNHI, config.min_logNHI, config.max_logNHI) +
                   obs_data.log_pdf['h1'](logNHI))
    return total_prior


def jv_model_lnprior(alpha, config):
    amp_a, amp_b, lognH, logZ, logNHI = alpha

    total_prior = (tophat_prior(lognH, config.min_lognH, config.max_lognH) +
                   tophat_prior(amp_a, config.min_amp_a, config.max_amp_a) +
                   tophat_prior(amp_b, config.min_amp_b, config.max_amp_b) +
                   tophat_prior(logZ, config.min_logZ, config.max_logZ) +
                   tophat_prior(logNHI, config.min_logNHI, config.max_logNHI) +
                   obs_data.log_pdf['h1'](logNHI))
    return total_prior


class Posterior(object):
    """
    Define the natural log of the posterior distribution

    Parameters:
    -----------
    config:
        Parameters object defined by the config file

    Returns:
    -----------
    lnprob: function
        The posterior distribution as a function of the
        input parameters given the spectral data
    """

    def __init__(self, config):
        self.config = config

    def lnlike(self, alpha):
        """
        Likelihood assumes flux follow a Gaussian

        Returns
        --------
        ln_likelihood: float
            Natural log of the likelihood
        """

        ln_likelihood = 0
        for ion_name in config.ion_names:
            if ion_name != 'h1':
                model_logN = ion_model.model_prediction(alpha, ion_name)
                config.log_pdf[ion_name](model_logN)
                ln_likelihood += config.log_pdf[ion_name](model_logN)
        return ln_likelihood

    def lnprior(self, alpha):
        """
        Natural Log of the priors
        """
        if config.nparams == 4:
            # return photo_model_aUV_lnprior(alpha,config)
            return photo_collision_model_lnprior(alpha, config)
        elif config.nparams == 5:
            return jv_model_lnprior(alpha, config)
        elif config.nparams == 3:
            return photo_model_lnprior(alpha, config)
        else:
            print('No models with %d parameters' % config.nparams)

    def __call__(self, alpha):
        """
        Posterior distribution

        Returns
        ---------
        lnprob: float
            Natural log of posterior probability
        """
        lp = self.lnprior(alpha)
        if np.isinf(lp):
            return -np.inf
        else:
            return np.atleast_1d(lp + self.lnlike(alpha))[0]
