import numpy as np
import pylab as pl
import os


def getCodeDir():

	return os.path.dirname(os.path.realpath(__file__))

code_dir = getCodeDir()


def read_abnd(fname = code_dir + '/data/abnd.dat'):
	"""
	Source: Asplund et al 2009, ARAA, section 3. Photosphere Abundance
	Read Solar Abudance
	into dictionary 
	"""
	name = np.loadtxt(fname,dtype=str,usecols=[0])
	abundance = np.loadtxt(fname,usecols=[1])
	
	log10_fraction = abundance - 12 # log10 of fraction

	my_dict = {}
	for i in xrange(len(name)):
		my_dict[str(name[i])] = float(log10_fraction[i])
	
	return my_dict

def SpecieMetalFraction():
	"""
	# Compute the fraction of carbon / all metal by number
	return the dictionary
	"""

	fname = code_dir + '/data/abnd.dat'
	name = np.loadtxt(fname,dtype=str,usecols=[0])
	abundance = np.loadtxt(fname,usecols=[1])


	my_dict = {}	

	metal = []
	for i in range(len(name)):
		if name[i] != 'H':
			metal.append(10**(abundance[i]-12))
	log_total_metal = np.log10(np.sum(metal))
	for i in range(len(name)):
		if name[i] != 'H':
			metal_only_fraction = (abundance[i] - 12.) - log_total_metal
			my_dict[str(name[i])] = float(metal_only_fraction)
	return my_dict
	


def MetalFraction():
	"""
	This is used for converting Z ---> [X/H] where X = any elements.

	Output: 
	Log 10 of the ratio of the number of all metal atoms over hydrogen atoms 
	"""
	fname = code_dir + '/data/abnd.dat'
	name = np.loadtxt(fname,dtype=str,usecols=[0])
	abundance = np.loadtxt(fname,usecols=[1])

	log_epsilon = []
	for i in range(len(name)):
		if name[i] != 'H' and name[i] != 'D':
			log_epsilon.append(abundance[i])

	log_epsilon = np.array(log_epsilon)
	log10_metalfraction = np.log10(np.sum(10**(log_epsilon - 12.)))
	return log10_metalfraction

def NumberFraction(specie):
	"""
	Return log10 of N_x/N_H in the sun
	"""
	fname = code_dir + '/data/abnd.dat'
	name = np.loadtxt(fname,dtype=str,usecols=[0])
	abundance = np.loadtxt(fname,usecols=[1])	
	
	logNx_NH = {}
	for i in range(len(name)):
		if name[i] != 'H':
			metal_only_fraction = abundance[i] - 12.
			logNx_NH[str(name[i])] = float(metal_only_fraction)

	return logNx_NH[specie]


if __name__ == '__main__':
	print NumberFraction('C')
