import numpy as np
import sys
import os
import time

from Config import DefineParams
from Model import DefineIonizationModel
np.seterr(all='ignore') # ignore floating points warnings.

def tophat_prior(model_x, x_left,x_right):
    if model_x >= x_left and model_x < x_right:
        return 0
    else:
        return -np.inf


def photo_model_lnprior(alpha,config):
    lognH,logZ,logNHI= alpha

    total_prior = (tophat_prior(lognH,config.min_lognH,config.max_lognH)    +
                  tophat_prior(logZ,config.min_logZ,  config.max_logZ)      +
                  tophat_prior(logNHI,config.min_logNHI, config.max_logNHI) + 
                  config.log_pdf['h1'](logNHI)) 	
    return total_prior


def photo_model_aUV_lnprior(alpha,config):
    lognH,logZ,aUV,logNHI= alpha

    total_prior = (tophat_prior(lognH,config.min_lognH,config.max_lognH)    +
                  tophat_prior(logZ,config.min_logZ,  config.max_logZ)      +
                  tophat_prior(aUV,config.min_aUV,  config.max_aUV)      +
                  tophat_prior(logNHI,config.min_logNHI, config.max_logNHI) + 
                  config.log_pdf['h1'](logNHI)) 	
    return total_prior


def photo_collision_model_lnprior(alpha,config):
    lognH,logZ,logT,logNHI = alpha

    total_prior = (tophat_prior(lognH,config.min_lognH,config.max_lognH)    +
                  tophat_prior(logT,config.min_logT,  config.max_logT)      +
                  tophat_prior(logZ,config.min_logZ,  config.max_logZ)      +
                  config.log_pdf['h1'](logNHI) + 
                  tophat_prior(logNHI,config.min_logNHI, config.max_logNHI))

    return total_prior


def lnlike(alpha):
    """
    Likelihood assumes flux follow a Gaussian
    
    Returns
    --------
    ln_likelihood: float
        Natural log of the likelihood
    """

    ln_likelihood = 0
    for ion_name in config_params.ion_names:
        if ion_name != 'h1':
            model_logN = ion_model.model_prediction(alpha,ion_name)
            config_params.log_pdf[ion_name](model_logN)
            ln_likelihood += config_params.log_pdf[ion_name](model_logN)
    return ln_likelihood


def lnprior(alpha):
    """
    Natural Log of the priors   
    """
    return photo_collision_model_lnprior(alpha,config_params)
    #if config_params.nparams == 4:            
    #    return photo_collision_model_lnprior(alpha,config_params)
    #elif config_params.nparams == 3:
    #        return photo_model_lnprior(alpha,config_params)
    #else:
    #    print('No models with %d parameters' % config_params.nparams)


def lnprob(alpha):
    """
    Posterior distribution

    Returns
    ---------
    lnprob: float
        Natural log of posterior probability
    """
    lp = lnprior(alpha)
    
    if np.isinf(lp):
        return -np.inf
    else:
        return np.atleast_1d(lp + lnlike(alpha))[0]


def _initialize_walkers(config_params):
    p0 = np.zeros((config_params.nparams,config_params.nwalkers))
    for i in xrange(config_params.nparams):
        p0[i] = np.random.uniform(config_params.priors[i][0],
                                config_params.priors[i][1],
                                size=config_params.nwalkers)
    
    return np.transpose(p0)

def ionfit_mcmc(config_params):
    t1 = time.time()
    import kombine

    config_params.print_config_params()

    if config_params.nconstraints < config_params.nparams:
        print('Number of constraints %d < number of parameters %d' % (config_params.nconstraints,config_params.nparams))
        print('Exiting program...')
        exit()

    # Set up the sampler
    sampler = kombine.Sampler(config_params.nwalkers, 
                              config_params.nparams, 
                              lnprob, processes=config_params.nthreads)

    p0 = _initialize_walkers(config_params)

    
    # First do a rough burn in based on accetance rate.
    p_post_q = sampler.burnin(p0)
    p_post_q = sampler.run_mcmc(config_params.nsteps)

    np.save(config_params.chain_fname + '.npy',sampler.chain)
    
    dt = time.time() -t1
    print('Finished in %f seconds' % dt)

if __name__ == '__main__':

    config_fname = sys.argv[1]
    config_params = DefineParams(config_fname)
    ion_model = DefineIonizationModel(config_params)
    ionfit_mcmc(config_params)


