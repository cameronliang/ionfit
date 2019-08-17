import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys
import os

from Config import DefineParams
from Model import DefineIonizationModel
from Utilities import triage, model_labels, printline


def write_mcmc_stats(config_params_obj, output_fname):
    chain = np.load(config_params_obj.chain_fname + '.npy')
    #burnin = compute_burin_GR(config_params_obj.chain_fname + '_GR.dat')
    burnin = config_params.burnin

    f = open(output_fname, 'w')
    f.write('x_med\tx_mean\tx_std\tx_cfl11\tx_cfl12\t x_cfl21\tx_cfl22\n')

    n_params = np.shape(chain)[-1]
    for i in xrange(n_params):
        x = chain[burnin:, :, i].flatten()
        output_stats = compute_stats(x)
        f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\n' %
                (output_stats[0], output_stats[1],
                 output_stats[2], output_stats[3],
                 output_stats[4], output_stats[5],
                 output_stats[6]))
    f.close()
    print('--> %s' % config_params.chain_short_fname + '.dat')
    return


def compute_stats(x):
    xmed = np.median(x); xm = np.mean(x); xsd = np.std(x)
    xcfl11 = np.percentile(x,16); xcfl12 = np.percentile(x,84)
    xcfl21 = np.percentile(x,2.5); xcfl22 = np.percentile(x,97.5)	
    return xmed, xm, xsd, xcfl11, xcfl12, xcfl21, xcfl22


def corner_plot(config_params, nbins=30, fontsize=None, cfigsize=[6, 6]):
    """
    Make triangle plot for visuaizaliton of the 
    multi-dimensional posterior
    """
    n_params = config_params.nparams
    burned_in_samples = config_params.burnin
    chain = np.load(config_params.chain_fname + '.npy')
    samples = np.array(chain[burned_in_samples:, :, :].reshape((-1,
                       config_params.nparams)))

    output_path = config_params.input_path + '/ionfit_plots/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    plt.figure(1)
    output_name = output_path + '/corner_' + config_params.chain_short_fname + '.pdf'

    plot_param_labels = model_labels(config_params.model)
    weights_of_chains = np.ones_like(samples)
    fig = triage(samples, weights_of_chains,
                 plot_param_labels, figsize=cfigsize, nbins=nbins,
                 figname=output_name, fontsize=fontsize)
    plt.clf()
    print('--> %s' % 'corner_' + config_params.chain_short_fname + '.pdf')


def comparison_plot(config_params):
    burnin = config_params.burnin

    median_logNs = np.zeros(len(config_params.ion_names))
    logN21s = np.zeros(len(config_params.ion_names))
    logN22s = np.zeros(len(config_params.ion_names))
    model_logNs = np.zeros(len(config_params.ion_names))
    model_logNs21 = np.zeros(len(config_params.ion_names))
    model_logNs22 = np.zeros(len(config_params.ion_names))
    chain = np.load(config_params.chain_fname + '.npy')
    samples = chain[burnin:, :, :].reshape((-1, config_params.nparams))

    bestfit_params = np.median(samples, 0)
    bestfit_params21 = np.percentile(samples, 2.5, axis=0)
    bestfit_params22 = np.percentile(samples, 97.5, axis=0)

    vp_path = os.path.abspath(os.path.join(config_params.input_path, os.pardir))

    for i, ion in enumerate(config_params.ion_names):
        vpchain = np.load(vp_path + '/' + ion + '.npy')
        vpchain = vpchain[burnin:, :, 0].ravel()

        median_logNs[i] = np.median(vpchain)
        logN21s[i] = np.percentile(vpchain, 2.5)
        logN22s[i] = np.percentile(vpchain, 97.5)

        model_logNs[i] = ion_model.model_prediction(bestfit_params, ion)
        model_logNs21[i] = ion_model.model_prediction(bestfit_params21, ion)
        model_logNs22[i] = ion_model.model_prediction(bestfit_params22, ion)
    x = np.arange(len(config_params.ion_names))

    # Plot observed data (column densities)
    plt.plot(x, median_logNs, 'ms')
    dz21 = 0.434*((10**median_logNs-10**logN21s) / 10**median_logNs)
    dz22 = 0.434*((10**logN22s-10**median_logNs) / 10**median_logNs)
    plt.errorbar(x, median_logNs, yerr=[dz21, dz22], linestyle='-', color='b',
                 ecolor='b', linewidth=1.5, capthick=1.5, label='Data')

    dz21 = 0.434*((10**model_logNs-10**model_logNs21) / 10**model_logNs)
    dz22 = 0.434*((10**model_logNs22-10**model_logNs) / 10**model_logNs)
    plt.plot(x, model_logNs, 'gs', alpha=0.5)
    plt.errorbar(x, model_logNs, yerr=[dz21, dz22], linestyle='-', color='g',
                 ecolor='g', linewidth=1.5, capthick=2.5, label='Model')

    plt.legend(loc='best')
    plt.xlim([min(x)-1, max(x)+1])
    min_y = min(min(model_logNs), min(logN21s))
    max_y = max(max(model_logNs), max(logN22s))
    plt.ylim([min_y-0.3, max_y+0.3])
    plt.xticks(x, config_params.ion_names)

    output_path = config_params.input_path + '/ionfit_plots/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    fname = output_path + config_params.chain_short_fname + '_modelcomp.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()
    print('--> %s' % config_params.chain_short_fname + '_modelcomp.png')


def write_model_summary(config_params):
    mcmc_chain_fname = np.load(config_params.chain_fname + '.npy')
    output_path = config_params.input_path + '/ionfit_data'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    output_summary_fname = output_path + '/' + config_params.chain_short_fname + '.dat'
    write_mcmc_stats(config_params, output_summary_fname)
    printline()

if __name__ == '__main__':

    config_fname = sys.argv[1]
    config_params = DefineParams(config_fname)
    ion_model = DefineIonizationModel(config_params)
    config_params.print_config_params()

    corner_plot(config_params)
    comparison_plot(config_params)
    write_model_summary(config_params)
