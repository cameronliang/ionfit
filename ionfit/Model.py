"""
This is the cleaned up version of the code Utilities.py
it works - but has not been tested for bugs.

make dictionary to convert from 'c3' to 'C', specie.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d,\
							  RegularGridInterpolator

from read_solarabund import SpecieMetalFraction, MetalFraction,\
							NumberFraction

################################################################################
# utils
################################################################################

def helper(ion):
	if ion == 'h1':
		return 'H'
	elif ion == 'c2' or ion == 'c3' or ion == 'c4':
		return 'C'
	elif ion == 'ne8':
		return 'Ne'
	elif ion == 'n5':
		return 'N'
	elif ion == 's2' or ion == 's3' or ion == 's4':
		return 'Si'
	elif ion == 'o1' or ion == 'o6':
		return 'O'
	elif ion == 'mg1' or ion == 'mg2': 
		return 'Mg'
	elif ion == 'fe2':
		return 'Fe'


def ion_lists():
	ions = np.array(['h1','c2','c3', 'c4',
					 's2','s3','s4', 'o1',
					 'o6','ne8','n5','mg2','fe2'])	
	return ions 

def Cloudy_InputParamers():
	lognH = np.arange(-6,0.2,0.2)
	logNHI = np.arange(15,19.2,0.2)
	logT = np.arange(4.0,7.2,0.2)
	return lognH, logNHI,logT
	
def Cloudy_InputParamers_redshift(): 
	low_redshift = np.arange(0,3.0,0.2)
	high_redshift = np.arange(3.0,7.5,0.5)
	redshift = np.sort(np.concatenate((low_redshift,high_redshift)))

	lognH = np.arange(-7.0, 0.2, 0.2)
	logT  = np.arange(3.5,7.1,0.2)
	return lognH,logT, redshift

def GenericModelInterp(gal_z,ion_name,model_choice):

	input_path = '/data/jwliang/projects_data/cloudy_models'
	if model_choice == 'photo_collision_thin' or model_choice == 'photo_fix_logT_thin':
	
		# load the CLOUDY input parameters
		clognH,clogT,redshift = Cloudy_InputParamers_redshift()
		
		# Load the ionization fraction grid
		path = input_path + '/photo_collision_thin/CombinedGrid/cubes/'
		ind = int(np.where(abs(redshift-gal_z) < 0.1)[0]) # Use the closest z
		ion = np.load(path + ion_name + '.npy')[ind,:,:]
		
		# Interpolate the function
		f = np.vectorize(RectBivariateSpline(clognH,clogT,ion))

	elif model_choice == 'photo_collision_noUVB':
		clognH,clogNHI,clogT = Cloudy_InputParamers()
		path = input_path + '/photo_collision_rahmati/f0.0/cubes/'
		ion = np.load(path + ion_name + '.npy')
		f = np.vectorize(RectBivariateSpline(clognH,clogT,ion))

	elif model_choice == 'photo_collision_rahmati':
		#will change name from optically_thick_rahmati to photo_collision_rahmati after the models are finished 
		clognH,clogNHI,clogT = Cloudy_InputParamers()
		path = input_path + '/' + model_choice + '/CombinedGrid/cubes/'
		cgamma_ratios = np.load(path + '/uvb_fraction.npy')
		ion = np.load(path + ion_name + '.npy')
		f = RegularGridInterpolator((clognH,clogT,cgamma_ratios),ion)

	elif model_choice == 'photo_collision_thick':
		clognH  = np.arange(-6,0.2,0.2)
		clogNHI = np.arange(15,19.2,0.2)
		clogT   = np.arange(3.8,6.2,0.2)
		path = input_path + '/' + model_choice + '/CombinedGrid/cubes/'
		ion =  np.load(path + ion_name + '.npy') 
		f = RegularGridInterpolator((clognH,clogNHI,clogT),ion)

	return f

def GetAllIonFunctions(gal_z,model_choice):
	ions_names = ion_lists()
	f = [] 
	for ion_name in ions_names: 
		f.append(GenericModelInterp(gal_z,ion_name,model_choice)) 
	f = np.array(f)
	
	# Make the dictionary between functions and ionization state. 
	dict_intepfunc = {}
	for i in range(len(ions_names)): dict_intepfunc[ions_names[i]] = f[i]
	return dict_intepfunc
	
################################################################################
# Physics related Utils 
################################################################################

def ComputeGammaRatio(lognH):
	"""
	ratio = Gamma/Gamma_UVB; 
	eqn 14. from Rahmati 2013.
	"""
	nH_ssh = 5.1*1e-4; # value taken from table 2 Rahmati+ 2013
	
	nH = 10**lognH
	ratio = 0.98*(1+(nH/nH_ssh)**1.64)**-2.28 + 0.02*(1+nH/nH_ssh)**-0.84
	return ratio

def logZfrac(logZ,specie):
	#logZ is in solar units already  
	logNx_NH = NumberFraction(specie) # number density ratio in the sun 
	return logZ + logNx_NH


################################################################################

class DefineIonizationModel:
	"""
	A Ionization model class 
	"""

	def __init__(self,config_params):
		self.config_params = config_params
		self.logf_ion = GetAllIonFunctions(config_params.model_redshift,config_params.model) 

	def model_prediction(self,alpha,ion_name):
		"""
		Calculate column density given a specific ion, and the model 
		parameters in a photo-ionization model
		"""
		specie = helper(ion_name)

		if self.config_params.model == 'photo_collision_thin':
			lognH,logZ,logT,logNHI = alpha
			if -6 < lognH < 0 and 10 < logNHI <= 22 and 3.8 <= logT < 7:
				logN = (self.logf_ion[ion_name](lognH,logT) - self.logf_ion['h1'](lognH,logT) + logZfrac(logZ,specie) + logNHI)[0][0]
			else:
				logN = -np.inf
			
		elif self.config_params.model == 'photo_collision_noUVB':
			lognH,logZ,logT,logNHI = alpha
			logN = (self.logf_ion[ion_name](lognH,logT) - self.logf_ion['h1'](lognH,logT) + 
					logZfrac(logZ,specie) + logNHI)[0][0]
		
		elif self.config_params.model == 'photo_collision_rahmati':
			lognH,logZ,logT,logNHI = alpha
			if lognH < -6: 
				lognH = -6.0 # this is because the models were not run below -6. 
			elif lognH > 0:
				lognH = 0.
			
			gamma_ratio = ComputeGammaRatio(lognH)
			logN = (self.logf_ion[ion_name]((lognH,logT,gamma_ratio)) - self.logf_ion['h1']((lognH,logT,gamma_ratio)) + logZfrac(logZ,specie) + logNHI)[0][0]

		elif self.config_params.model == 'photo_collision_thick':	
			lognH, logZ,logT,logNHI = alpha

			# ranges to protect out of range in interpolated function. 
			if -6. < lognH <= 0. and 10. < logNHI <= 19. and 3.8 <= logT <6.:

				if logNHI <= 15:
					ifrac_alpha = np.array([lognH,15.0,logT])
					logN = (self.logf_ion[ion_name](ifrac_alpha) - self.logf_ion['h1'](ifrac_alpha) + 
							logZfrac(logZ,specie) + logNHI)[0]
				else:
					ifrac_alpha = np.array([lognH,logNHI,logT])
					logN = np.atleast_1d(self.logf_ion[ion_name](ifrac_alpha) - self.logf_ion['h1'](ifrac_alpha) + 
							logZfrac(logZ,specie) + logNHI)[0]
					
			else:
				logN = -np.inf
		
		elif self.config_params.model == 'photo_fix_logT_thin':
			lognH,logZ,logNHI = alpha
			logT = 4.0 # one can fix this to whatever tempature
			#print 'Assumed logT = %f' % logT  
			logN = (self.logf_ion[ion_name](lognH,logT) - self.logf_ion['h1'](lognH,logT) + 
					logZfrac(logZ,specie) + logNHI)
		return logN



if __name__ == '__main__':
	import sys
	from Config import DefineParams
	config_fname = sys.argv[1]
	config_params = DefineParams(config_fname)
	ion_model = DefineIonizationModel(config_params)

	alpha = np.array([-3,0,4.2,17])
	import time 
	t1 = time.time()
	print ion_model.model_prediction(alpha,'c2')
	print ion_model.model_prediction(alpha,'c3')
	print ion_model.model_prediction(alpha,'c4')
	print ion_model.model_prediction(alpha,'s2')
	print ion_model.model_prediction(alpha,'s3')
	print ion_model.model_prediction(alpha,'s4')
	print ion_model.model_prediction(alpha,'o1')
	print ion_model.model_prediction(alpha,'o6')
	print ion_model.model_prediction(alpha,'n5')
	print time.time() - t1
