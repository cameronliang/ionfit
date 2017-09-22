import numpy as np
import pylab as pl
import corner
import sys,os
from Config import DefineParams
from Model import DefineIonizationModel

def write_mcmc_stats(config_params_obj,output_fname):
	chain = np.load(config_params_obj.chain_fname + '.npy')
	#burnin = compute_burin_GR(config_params_obj.chain_fname + '_GR.dat')
	burnin = 1000

	f = open(output_fname,'w')
	f.write('x_med\tx_mean\tx_std\tx_cfl11\tx_cfl12\t x_cfl21\tx_cfl22\n')
	
	n_params = np.shape(chain)[-1]
	for i in xrange(n_params):
		x            = chain[burnin:,:,i].flatten()
		output_stats = compute_stats(x)
		f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % 
				(output_stats[0],output_stats[1],
				 output_stats[2],output_stats[3],
				 output_stats[4],output_stats[5],
				 output_stats[6]))
		
	f.close()
	print('Written %s' % output_fname)

	return

def compute_stats(x):
	xmed = np.median(x); xm = np.mean(x); xsd = np.std(x)
	xcfl11 = np.percentile(x,16); xcfl12 = np.percentile(x,84)
	xcfl21 = np.percentile(x,2.5); xcfl22 = np.percentile(x,97.5)	
	return xmed,xm,xsd,xcfl11, xcfl12, xcfl21,xcfl22

def corner_plot(config_params):
	#burnin = 200
	chain = np.load(config_params.chain_fname + '.npy')


	samples = chain[500:, :, :].reshape((-1, config_params.nparams))

	if config_params.nparams == 3 or config_params.nparams == 4:
		#fig = corner.corner(samples)
		fig = corner.corner(samples,quantiles=(0.16,0.5, 0.84),bins=30,smooth1d=True,
							truths=([tlognH,tlogZ,tlogT,tlogNHI]),
							labels=[r"$\log n_{\rm H}\,[\rm cm^{-3}]$",
							r"$\log Z\,[\rm Z_{\odot}]$",
							r"$\log T\,[\rm K]$",
							r"$\log N_{\rm HI}\,[\rm cm^{-2}]$"],
							show_titles=True,title_kwargs={"fontsize": 13})
		#fig.suptitle("PDF")
	"""	
	elif config_params.nparams == 3:
		fig = corner.corner(samples,quantiles=[0.16, 0.5, 0.84],
			labels=[r"$\log n_{\rm H}\,[\rm cm^{-3}]$",
					r"$\log Z\,[\rm Z_{\odot}]$",
					r"$\log N_{\rm HI}\,[\rm cm^{-2}]$"],
					show_titles=True,title_kwargs={"fontsize": 15})
					"""
	
	output_path = config_params.input_path + '/ionfit_plots/'
	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	
	fname = output_path + config_params.chain_short_fname + '.png'
	pl.savefig(fname, bbox_inches='tight')
	pl.clf()
	#print(fname)


def comparison_plot(config_params):
	logN_index = 0 # assume the first variable is logN, not b, or z. 
	burnin     = 1000 # assume burnin of 1000 steps for the BVPFIT

	median_logNs = np.zeros(len(config_params.ion_names))
	logN21s      = np.zeros(len(config_params.ion_names))
	logN22s      = np.zeros(len(config_params.ion_names))
	model_logNs  = np.zeros(len(config_params.ion_names))
	model_logNs21 = np.zeros(len(config_params.ion_names))
	model_logNs22 = np.zeros(len(config_params.ion_names))
	chain = np.load(config_params.chain_fname + '.npy')
	samples = chain[1000:, :, :].reshape((-1, config_params.nparams))

	#print np.shape(samples)
	bestfit_params = np.median(samples,0)
	bestfit_params21 = np.percentile(samples,2.5,axis=0)
	bestfit_params22 = np.percentile(samples,97.5,axis=0)

	# path to logN vpfit chains
	vp_path = config_params.input_path[:-9] 
	for i, ion in enumerate(config_params.ion_names):

		chain = np.load(vp_path + ion + '.npy')
		chain = chain[burnin:,:,logN_index].ravel()
		median_logNs[i] = np.median(chain)
		logN21s[i] =  np.percentile(chain,2.5)
		logN22s[i] =  np.percentile(chain,97.5)

		if ion == 'h1':
			model_logNs[i] = bestfit_params[-1]
		else:
			model_logNs[i] = ion_model.model_prediction(bestfit_params,ion)
			model_logNs21[i] = ion_model.model_prediction(bestfit_params21,ion)
			model_logNs22[i] = ion_model.model_prediction(bestfit_params22,ion)


	x = np.arange(len(config_params.ion_names))
	pl.plot(x, median_logNs,'bo',ms=12)
	pl.errorbar(x,median_logNs, yerr = [median_logNs-logN21s,		
				logN22s-median_logNs],linestyle='-',color  ='b', 
				ecolor='b',linewidth=1.5,capthick=1.5,label='Data')
	pl.plot(x, model_logNs,'gs',ms=10,alpha = 0.5)
	pl.errorbar(x,model_logNs, yerr = [model_logNs-model_logNs21,		
				model_logNs22-model_logNs],linestyle='-',color  ='g', 
				ecolor='g',linewidth=1.5,capthick=1.5,label='Model')
	pl.legend(loc='best')
	pl.xlim([min(x) - 1,max(x)+1])
	min_y = min(min(model_logNs),min(logN21s))
	max_y = max(max(model_logNs),max(logN22s))
	pl.ylim([min_y-0.3,max_y+0.3])
	pl.xticks(x,config_params.ion_names)

	output_path = config_params.input_path + '/ionfit_plots/'
	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	fname = output_path + config_params.chain_short_fname + '_modelcomp.png'
	pl.savefig(fname, bbox_inches='tight')
	pl.clf()
	print(fname)
	print('\n')


def write_model_summary(config_params):

	mcmc_chain_fname = np.load(config_params.chain_fname + '.npy') 
	output_path = config_params.input_path + '/ionfit_data'
	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	output_summary_fname = output_path +  '/' + config_params.chain_short_fname + '.dat'
	write_mcmc_stats(config_params,output_summary_fname)

if __name__ == '__main__':


	config_fname = sys.argv[1]
	
	tlognH  = float(sys.argv[2])
	tlogT   = float(sys.argv[3])
	tlogZ   = float(sys.argv[4])
	tlogNHI = float(sys.argv[5])
	
	config_params = DefineParams(config_fname)
	ion_model = DefineIonizationModel(config_params)
	config_params.print_config_params()
	corner_plot(config_params)
	comparison_plot(config_params)
	write_model_summary(config_params)