import numpy as np
import matplotlib.pyplot as plt


def printline():
    print('--------------------------------------------------------')


def model_labels(model):
    if model == 'photo_collision_thin' or model == 'photo_collision_thick':
        labels = [r'$\log n_{\rm H}$', r'$\log Z/Z_{\odot}$',
                  r'$\log T [\rm{K}]$', r'$\log N_{\rm HI}$']
    elif model == 'photo_thick':
        labels = [r'$\log n_{\rm H}$', r'$\log Z/Z_{\odot}$',
                  r'$\log N_{\rm HI}$']
    elif model == 'photo_thick_aUV':
        labels = [r'$\log n_{\rm H}$', r'$\log Z/Z_{\odot}$',
                  r'\alpha_{\rm UV}', r'$\log N_{\rm HI}$']
    elif model == 'jv_model':
        labels = [r'$a_{\rm Amp}$', r'$b_{\rm Amp}$',
                  r'$\log n_{\rm H}$', r'$\log Z/Z_{\odot}$',
                  r'$\log N_{\rm HI}$']
    return labels


def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level


def triage(par, weights, parnames, nbins=30, hist2d_color=plt.cm.PuBu,
           hist1d_color='#3681f9', figsize=[6, 6], figname=None,
           fontsize=None, labelsize=None):
    """
    Plot the multi-dimensional and marginalized posterior distribution (e.g., `corner` plot)

    Parameters:
    ----------
    par: array
        sampled mcmc chains with shape (n_params,nsteps)
    weights: array
        wights of the chains (nominally=1 if all chains carry equal weights), same shape as par
    parnames: array
        parameter names
    nbins: int
        number of bins in the histograms of PDF
    hist2d_color: str
        matplotlib colormap of the 2D histogram
    hist1d_color: str
        color for the 1D marginalized histogram
    figsize: list
        size of the figure. example: [6,6]
    figname: str
        full path and name of the figure to be written
    fontsize: int
        fontsize of the labels
    labelsize: int
        size of tickmark labels
    """

    import matplotlib.gridspec as gridspec

    import itertools as it
    import scipy.optimize as opt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator, NullLocator
    import matplotlib.ticker as ticker

    import warnings
    # ignore warnings if matplotlib version is older than 1.5.3
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ignore warning for log10 of parameters with values <=0.
    warnings.simplefilter(action='ignore', category=UserWarning)

    npar = np.size(par[1, :])
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(npar, npar, wspace=0.05, hspace=0.05)

    if labelsize is None:
        if npar <= 3:
            labelsize = 7
        else:
            labelsize = 5

    if fontsize is None:
        if npar <= 3:
            fontsize = 11
        else:
            fontsize = 10

    for h, v in it.product(range(npar), range(npar)):
        ax = plt.subplot(gs[h, v])

        x_min, x_max = np.min(par[:, v]), np.max(par[:, v])
        y_min, y_max = np.min(par[:, h]), np.max(par[:, h])

        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.tick_params(axis='both', which='minor', labelsize=labelsize)
        if h < npar-1:
            ax.get_xaxis().set_ticklabels([])
        if v > 0:
            ax.get_yaxis().set_ticklabels([])

        if h > v:
            hvals, xedges, yedges = np.histogram2d(par[:, v], par[:, h],
                                                   weights=weights[:, 0],
                                                   bins=nbins)
            hvals = np.rot90(hvals)
            hvals = np.flipud(hvals)

            Hmasked = np.ma.masked_where(hvals == 0, hvals)
            hvals = hvals / np.sum(hvals)

            X, Y = np.meshgrid(xedges, yedges)

            sig1 = opt.brentq(conf_interval, 0., 1., args=(hvals, 0.683))
            sig2 = opt.brentq(conf_interval, 0., 1., args=(hvals, 0.953))
            sig3 = opt.brentq(conf_interval, 0., 1., args=(hvals, 0.997))
            lvls = [sig3, sig2, sig1]

            ax.pcolor(X, Y, (Hmasked), cmap=hist2d_color, norm=LogNorm())
            ax.contour(hvals, linewidths=(1.0, 0.5, 0.25), colors='lavender',
                       levels=lvls, norm=LogNorm(), extent=[xedges[0],
                       xedges[-1], yedges[0], yedges[-1]])

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

        elif v == h:
            ax.hist(par[:, h], bins=nbins, color=hist1d_color, histtype='step',
                    lw=1.5)
            ax.yaxis.set_ticklabels([])

            hmedian = np.percentile(par[:, h], 50)
            h16 = np.percentile(par[:, h], 16)
            h84 = np.percentile(par[:, h], 84)

            ax.axvline(hmedian, lw=0.8, ls='--', color='k')
            ax.axvline(h16, lw=0.8, ls='--', color='k')
            ax.axvline(h84, lw=0.8, ls='--', color='k')
            ax.set_xlim([x_min, x_max])

            ax.set_title(r'$%.2f^{+%.2f}_{-%.2f}$' % (hmedian, h84-hmedian,
                         hmedian-h16), fontsize=fontsize)

        else:
            ax.axis('off')
        if v == 0:
            ax.set_ylabel(parnames[h], fontsize=fontsize)
            if npar <= 3:
                ax.get_yaxis().set_label_coords(-0.35, 0.5)
            else:
                ax.get_yaxis().set_label_coords(-0.4, 0.5)
            ax.locator_params(nbins=5, axis='y')
            labels = ax.get_yticklabels()
            for label in labels:
                label.set_rotation(20)
        if h == npar-1:
            ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
            ax.set_xlabel(parnames[v], fontsize=fontsize)
            ax.locator_params(nbins=5, axis='x')

            if npar <= 3:
                ax.get_xaxis().set_label_coords(0.5, -0.35)
            else:
                ax.get_xaxis().set_label_coords(0.5, -0.6)
            labels = ax.get_xticklabels()
            for label in labels:
                label.set_rotation(80)

    fig.get_tight_layout()
    if figname:
        plt.savefig(figname, dpi=120, bbox_inches='tight')


if __name__ == '__main__':
    labels =  model_labels('photo_collision_thin')
    plt.plot([1,2,3])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()