"""
This is an updated version of reading the config file
"""
import numpy as np
import sys
import os
from scipy.interpolate import interp1d


class DefineParams:


    def __init__(self,config_fname):

        self.config_fname = config_fname

        self.priors = []
        # Read and filter empty lines
        all_lines = filter(None,(line.rstrip() for line in open(config_fname)))

        # Remove commented lines
        self.lines = []
        for line in all_lines:
            if not line.startswith('#') and not line.startswith('!'): 
                self.lines.append(line)

        # Paths and fname strings
        for line in self.lines:
            line = filter(None,line.split(' '))
            if 'input' in line:
                self.input_path = line[1]
            elif 'output' in line or 'chain' in line:
                self.chain_short_fname = line[1]
            elif 'mcmc_params' in line or 'mcmc' in line:
                self.nwalkers = int(line[1])
                self.nsteps   = int(line[2])
                self.nthreads = int(line[3])
                # Default  
                self.mcmc_sampler    = 'kombine'

            elif 'ions' in line:
                self.ion_names = line[1:] # list of ion names
                self.n_metal_ions = len(self.ion_names)-1
                self.log_pdf = {}
                for ion_name in self.ion_names:
                    full_path_to_pdf = self.input_path+'/logN_'+ion_name+ '.dat'
                    logN, log_pdf = np.loadtxt(full_path_to_pdf,unpack=True)
                    f = interp1d(logN,log_pdf,kind='linear',bounds_error= False,fill_value=-np.inf)
                    self.log_pdf[ion_name] = f
            elif 'model' in line:
                self.model = line[1]
                self.model_redshift   = float(line[2])
                
                self.nparams = 4
                if self.model == 'photo_fixed_logT_thin':
                    self.nparams = 3

            elif 'lognH' in line:
                self.min_lognH,self.max_lognH  = float(line[1]),float(line[2])
                self.priors.append([float(line[1]),float(line[2])])
            elif 'logZ' in line:
                self.min_logZ,self.max_logZ    = float(line[1]),float(line[2])
                self.priors.append([float(line[1]),float(line[2])])
            elif 'logT' in line:
                self.min_logT,self.max_logT    = float(line[1]),float(line[2])
                self.priors.append([float(line[1]),float(line[2])])
            elif 'logNHI' in line:
                self.min_logNHI,self.max_logNHI = float(line[1]),float(line[2])
                self.priors.append([float(line[1]),float(line[2])])


        self.mcmc_outputpath   = self.input_path + '/ionization_fit'
        if not os.path.isdir(self.mcmc_outputpath):
		    os.mkdir(self.mcmc_outputpath)
        self.chain_fname = self.mcmc_outputpath + '/' + self.chain_short_fname
        self.priors = np.array(self.priors)

    def print_config_params(self):
        print('Ionization model = %s'   % self.model)
        print('Number Params    = %s' % self.nparams)
        print('Model redshift   = %.5f\n' % self.model_redshift)
        print('Input Path       = %s' % self.input_path)
        print('Chain name       = %s' % self.chain_short_fname)
        print 'Ions included    =', self.ion_names


if __name__ == '__main__':

    config_fname = sys.argv[1]
    config = DefineParams(config_fname)
    config.print_config_params()
