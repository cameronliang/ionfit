"""
Ionization Modeling for MCMC setup.  This also contains stand-alone model
prediction.
"""

import numpy as np
# interpolation on regular grid in arbitrary dimension
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.interpolate import RegularGridInterpolator
from read_solarabund import SpecieMetalFraction, MetalFraction, NumberFraction

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
    elif ion == 'n5' or ion == 'n2' or ion == 'n3':
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
    ions = np.array(['h1', 'c2', 'c3', 'c4', 'n2', 'n3',
                     's2', 's3', 's4', 'o1',
                     'o6', 'ne8', 'n5', 'mg2', 'fe2'])
    return ions


def Cloudy_InputParamers():
    lognH = np.arange(-6.0, 0.2, 0.2)
    logNHI = np.arange(15, 19.2, 0.2)
    logT = np.arange(4.0, 7.2, 0.2)
    return lognH, logNHI, logT


def Cloudy_InputParamers_redshift(): 
    low_redshift = np.arange(0, 3.0, 0.2)
    high_redshift = np.arange(3.0, 7.5, 0.5)
    redshift = np.sort(np.concatenate((low_redshift, high_redshift)))

    lognH = np.arange(-7.0, 0.2, 0.2)
    logT = np.arange(3.5, 7.1, 0.2)
    return lognH, logT, redshift


def GenericModelInterp(gal_z, ion_name, model_choice):

    input_path = '/Users/cameronliang/research/cloudy_models'
    if model_choice == 'photo_collision_thin':

        # load the CLOUDY input parameters
        clognH, clogT, redshift = Cloudy_InputParamers_redshift()

        # Load the ionization fraction grid
        path = input_path + '/photo_collision_thin/CombinedGrid/cubes/'
        ind = int(np.where(abs(redshift-gal_z) < 0.1)[0])  # Use the closest z
        ion = np.load(path + ion_name + '.npy')[ind, :, :]

        # Interpolate the function
        f = np.vectorize(RectBivariateSpline(clognH, clogT, ion))

    elif model_choice == 'photo_collision_noUVB':
        clognH, clogNHI, clogT = Cloudy_InputParamers()
        path = input_path + '/photo_collision_rahmati/f0.0/cubes/'
        ion = np.load(path + ion_name + '.npy')
        f = np.vectorize(RectBivariateSpline(clognH, clogT, ion))

    elif model_choice == 'photo_collision_rahmati':
        # will change name from optically_thick_rahmati
        # to photo_collision_rahmati after the models are finished
        clognH, clogNHI, clogT = Cloudy_InputParamers()
        path = input_path + '/' + model_choice + '/CombinedGrid/cubes/'
        cgamma_ratios = np.load(path + '/uvb_fraction.npy')
        ion = np.load(path + ion_name + '.npy')
        f = RegularGridInterpolator((clognH, clogT, cgamma_ratios), ion)

    elif model_choice == 'jv_model':
        path = input_path + '/' + model_choice + '/combined_grid/cubes/'
        amp_a = np.load(path+'a.npy')
        amp_b = np.load(path+'b.npy')
        clognH = np.load(path+'lognH.npy')
        clogNHI = np.load(path+'logNHI.npy')
        ion = np.load(path + ion_name + '.npy')

        f = RegularGridInterpolator((amp_a, amp_b, clognH, clogNHI), ion)

    elif model_choice == 'photo_collision_thick':
        clognH = np.arange(-6, 0.2, 0.2)
        clogNHI = np.arange(15, 19.2, 0.2)
        clogT = np.arange(3.8, 6.2, 0.2)
        path = input_path + '/' + model_choice + '/CombinedGrid/cubes/'
        ion = np.load(path + ion_name + '.npy')
        f = RegularGridInterpolator((clognH, clogNHI, clogT), ion)

    elif model_choice == 'photo_thick':
        credshift = np.arange(0, 0.4, 0.1)
        clogNHI = np.arange(14, 22.2, 0.2)
        clognH = np.arange(-4.2, 0.2, 0.2)
        path = input_path + '/' + model_choice + '/CombinedGrid/cubes/'

        #ind = int(np.where(abs(redshift-gal_z) < 0.1)[0]) # Use the closest z
        ion = np.load(path + ion_name + '.npy')  # [ind,:,:]
        f_3D = RegularGridInterpolator((credshift, clognH, clogNHI), ion)

        new_ion = np.zeros((len(clognH), len(clogNHI)))
        for i in range(len(clognH)):
            for j in range(len(clogNHI)):
                new_ion[i][j] = f_3D((gal_z, clognH[i], clogNHI[j]))
        f = RectBivariateSpline(clognH, clogNHI, new_ion)

    elif model_choice == 'photo_thick_aUV':
        # c before aUV just means cloudy grid values
        credshift = np.arange(0, 0.4, 0.1)
        caUV = np.arange(-3, 2.0, 0.5)
        clogNHI = np.arange(14, 22, 0.3)
        clognH = np.linspace(-4.4, 0., 12)
        path = input_path + '/' + model_choice + '/grids/CombinedGrid/cubes/'

        ion = np.load(path + ion_name + '.npy')  # 4D array
        f_4D = RegularGridInterpolator((credshift, caUV, clognH, clogNHI), ion)

        new_ion = np.zeros(( len(caUV), len(clognH), len(clogNHI) ))
        for i in range(len(caUV)):
            for j in range(len(clognH)):
                for k in range(len(clogNHI)):
                    new_ion[i][j][k] = f_4D((gal_z, caUV[i], clognH[j], clogNHI[k]))

        f = RegularGridInterpolator((caUV, clognH, clogNHI), new_ion)

    return f


def GetAllIonFunctions(gal_z, model_choice):
    ions_names = ion_lists()
    f = []
    for ion_name in ions_names:
        f.append(GenericModelInterp(gal_z, ion_name, model_choice))
    f = np.array(f)

    # Make the dictionary between functions and ionization state
    dict_intepfunc = {}
    for i in range(len(ions_names)):
        dict_intepfunc[ions_names[i]] = f[i]
    return dict_intepfunc

#############################################################################
# Physics related Utils
#############################################################################


def ComputeGammaRatio(lognH):
    """
    ratio = Gamma/Gamma_UVB
    eqn 14. from Rahmati 2013.
    """
    # value taken from table 2 Rahmati+ 2013
    nH_ssh = 5.1*1.0e-4
    nH = 10**lognH
    ratio = 0.98*(1+(nH/nH_ssh)**1.64)**-2.28 + 0.02*(1+nH/nH_ssh)**-0.84
    return ratio


def logZfrac(logZ, specie):
    # logZ is in solar units already
    logNx_NH = NumberFraction(specie)  # number density ratio in the sun
    return logZ + logNx_NH

###############################################################################


class DefineIonizationModel:
    def __init__(self, config_params):
        self.config_params = config_params
        self.logf_ion = GetAllIonFunctions(config_params.model_redshift,
                                           config_params.model)

    def model_prediction(self, alpha, ion_name):
        """
        Calculate column density given a specific ion, and the model
        parameters in a photo-ionization model
        """
        specie = helper(ion_name)
        if ion_name == 'h1':
            logNHI = alpha[-1]
            return logNHI
        else:
            if self.config_params.model == 'photo_collision_thin':
                lognH, logZ, logT, logNHI = alpha
                if -6 < lognH < 0 and 10 < logNHI <= 22 and 3.8 <= logT < 7:
                    logN = (self.logf_ion[ion_name](lognH, logT) -
                            self.logf_ion['h1'](lognH, logT) +
                            logNHI)[0][0]
                else:
                    logN = -np.inf

            elif self.config_params.model == 'photo_collision_noUVB':
                lognH, logZ, logT, logNHI = alpha
                logN = (self.logf_ion[ion_name](lognH, logT) -
                        self.logf_ion['h1'](lognH, logT) +
                        logZfrac(logZ, specie) + logNHI)[0][0]

            elif self.config_params.model == 'photo_collision_rahmati':
                lognH, logZ, logT, logNHI = alpha
                if lognH < -6:
                    lognH = -6.0  # because the models were not run below -6
                elif lognH > 0:
                    lognH = 0.

                gamma_ratio = ComputeGammaRatio(lognH)
                logN = (self.logf_ion[ion_name]((lognH, logT, gamma_ratio)) -
                        self.logf_ion['h1']((lognH, logT, gamma_ratio)) +
                        logZfrac(logZ, specie) + logNHI)[0][0]

            elif self.config_params.model == 'photo_collision_thick':
                lognH, logZ, logT, logNHI = alpha
                # ranges to protect out of range in interpolated function
                if -6. < lognH <= 0. and 10. < logNHI <= 19. and 3.8 <= logT < 6.:
                    if logNHI <= 15:
                        ifrac_alpha = np.array([lognH, 15.0, logT])
                    else:
                        ifrac_alpha = np.array([lognH, logNHI, logT])
                    logN = (self.logf_ion[ion_name](ifrac_alpha) -
                            self.logf_ion['h1'](ifrac_alpha) +
                            logZfrac(logZ, specie) + logNHI)[0]
                else:
                    logN = -np.inf

            elif self.config_params.model == 'jv_model':
                amp_a, amp_b, lognH, logZ, logNHI = alpha
                print 'enter'
                # ranges to protect out of range in interpolated function
                if (-5.0 < lognH <= 0. and 10. < logNHI <= 19. and -1 <
                   amp_a <= 4 and -1 < amp_b <= 1):
                    if logNHI <= 10:
                        ifrac_alpha = np.array([amp_a, amp_b, lognH, 10.0])
                    else:
                        print 'here???'
                        ifrac_alpha = np.array([amp_a, amp_b, lognH, logNHI])
                    logN = (self.logf_ion[ion_name](ifrac_alpha) -
                            self.logf_ion['h1'](ifrac_alpha) +
                            logZfrac(logZ, specie) + logNHI)[0]
                else:
                    print 'here!!'
                    logN = -np.inf

            elif self.config_params.model == 'photo_fix_logT_thin':
                lognH, logZ, logNHI = alpha
                logT = 4.0  # one can fix this to whatever tempature
                logN = (self.logf_ion[ion_name](lognH, logT) -
                        self.logf_ion['h1'](lognH, logT) +
                        logZfrac(logZ, specie) + logNHI)

            elif self.config_params.model == 'photo_thick':
                lognH, logZ, logNHI = alpha
                if -4.2 < lognH < 0 and 0 < logNHI <= 22:
                    if logNHI < 14:
                        # if < 14, use optically thin for all values of NHI
                        logN = (self.logf_ion[ion_name](lognH, 14.0) -
                                self.logf_ion['h1'](lognH, logNHI) +
                                logZfrac(logZ, specie) + logNHI)[0][0]
                    else:
                        logN = (self.logf_ion[ion_name](lognH, logNHI) -
                                self.logf_ion['h1'](lognH, logNHI) +
                                logZfrac(logZ, specie) + logNHI)[0][0]
                else:
                    logN = -np.inf

            elif self.config_params.model == 'photo_thick_aUV':
                lognH, logZ, aUV, logNHI = alpha
                if -4.2 <= lognH < 0 and -3 <= aUV < 2 and 0 < logNHI <= 22:
                    if logNHI <= 14:
                        # if < 14 use optically thin for all values of NHI
                        logN = (self.logf_ion[ion_name]((aUV, lognH, 14.0)) -
                                self.logf_ion['h1']((aUV, lognH, 14.0)) +
                                logZfrac(logZ, specie) + logNHI)
                    else:
                        logN = (self.logf_ion[ion_name]((aUV, lognH, logNHI)) -
                                self.logf_ion['h1']((aUV, lognH, logNHI)) +
                                logZfrac(logZ, specie) + logNHI)
                else:
                    logN = -np.inf
            return logN

####################################


class DefineIonizationModel_test:
    """
    A Ionization model class
    """
    def __init__(self, model, model_redshift):
        self.model = model
        self.logf_ion = GetAllIonFunctions(model_redshift, model)

    def produce_ion_logn(self, alpha, ion_name):
        if self.model == 'photo_collision_thin':
            # for this model ony.. for now
            lognH, logZ, logT = alpha
            specie = helper(ion_name)
            if ion_name == 'h1':
                logn_ion = self.logf_ion[ion_name](lognH, logT) + lognH
            else:
                logn_ion = self.logf_ion[ion_name](lognH, logT) + \
                            logZfrac(logZ, specie) + lognH
        return logn_ion[0][0]

    def model_prediction(self, alpha, ion_name):
        """
        Calculate column density given a specific ion, and the model
        parameters in a photo-ionization model
        """
        specie = helper(ion_name)
        if ion_name == 'h1':
            logNHI = alpha[-1]
            return logNHI
        else:
            if self.model == 'photo_collision_thin':
                lognH, logZ, logT, logNHI = alpha
                if -6 < lognH < 0 and 10 < logNHI <= 22 and 3.8 <= logT < 7:
                    logN = (self.logf_ion[ion_name](lognH, logT) -
                            self.logf_ion['h1'](lognH, logT) +
                            logNHI)[0][0]
                else:
                    logN = -np.inf

            elif self.model == 'photo_collision_noUVB':
                lognH, logZ, logT, logNHI = alpha
                logN = (self.logf_ion[ion_name](lognH, logT) -
                        self.logf_ion['h1'](lognH, logT) +
                        logZfrac(logZ, specie) + logNHI)[0][0]

            elif self.model == 'photo_collision_rahmati':
                lognH, logZ, logT, logNHI = alpha
                if lognH < -6:
                    lognH = -6.0  # because the models were not run below -6
                elif lognH > 0:
                    lognH = 0.
                gamma_ratio = ComputeGammaRatio(lognH)
                logN = (self.logf_ion[ion_name]((lognH, logT, gamma_ratio)) -
                        self.logf_ion['h1']((lognH, logT, gamma_ratio)) +
                        logZfrac(logZ, specie) + logNHI)[0][0]

            elif self.model == 'photo_collision_thick':
                lognH, logZ, logT, logNHI = alpha
                # ranges to protect out of range in interpolated function
                if -6. < lognH <= 0. and 10. < logNHI <= 19. and 3.8 <= logT < 6.:
                    if logNHI <= 15:
                        ifrac_alpha = np.array([lognH, 15.0, logT])
                    else:
                        ifrac_alpha = np.array([lognH, logNHI, logT])
                    logN = (self.logf_ion[ion_name](ifrac_alpha) -
                            self.logf_ion['h1'](ifrac_alpha) +
                            logZfrac(logZ, specie) + logNHI)[0]
                else:
                    logN = -np.inf

            elif self.model == 'jv_model':
                amp_a, amp_b, lognH, logZ, logNHI = alpha
                # ranges to protect out of range in interpolated function
                if (-6. < lognH <= 0. and 10. < logNHI <= 19. and -1 <
                   amp_a <= 4 and -1 <= amp_b < 4):
                    if logNHI <= 10:
                        ifrac_alpha = np.array([amp_a, amp_b, lognH, 10.0])
                    else:
                        ifrac_alpha = np.array([amp_a, amp_b, lognH, logNHI])
                    logN = (self.logf_ion[ion_name](ifrac_alpha) -
                            self.logf_ion['h1'](ifrac_alpha) +
                            logZfrac(logZ, specie) + logNHI)[0]
                else:
                    logN = -np.inf

            elif self.model == 'photo_fix_logT_thin':
                lognH, logZ, logNHI = alpha
                logT = 4.0  # one can fix this to whatever tempature
                logN = (self.logf_ion[ion_name](lognH, logT) -
                        self.logf_ion['h1'](lognH, logT) +
                        logZfrac(logZ, specie) + logNHI)

            elif self.model == 'photo_thick':
                lognH, logZ, logNHI = alpha
                if -4.2 < lognH < 0 and 0 < logNHI <= 22:
                    if logNHI < 14:
                        # if < 14, use optically thin for all values of NHI
                        logN = (self.logf_ion[ion_name](lognH, 14.0) -
                                self.logf_ion['h1'](lognH, logNHI) +
                                logZfrac(logZ, specie) + logNHI)[0][0]
                    else:
                        logN = (self.logf_ion[ion_name](lognH, logNHI) -
                                self.logf_ion['h1'](lognH, logNHI) +
                                logZfrac(logZ, specie) + logNHI)[0][0]
                else:
                    logN = -np.inf

            elif self.model == 'photo_thick_aUV':
                lognH, logZ, aUV, logNHI = alpha
                if -4.2 <= lognH < 0 and -3 <= aUV < 2 and 0 < logNHI <= 22:
                    if logNHI <= 14:
                        # if < 14 use optically thin for all values of NHI
                        logN = (self.logf_ion[ion_name]((aUV, lognH, 14.0)) -
                                self.logf_ion['h1']((aUV, lognH, 14.0)) +
                                logZfrac(logZ, specie) + logNHI)
                    else:
                        logN = (self.logf_ion[ion_name]((aUV, lognH, logNHI)) -
                                self.logf_ion['h1']((aUV, lognH, logNHI)) +
                                logZfrac(logZ, specie) + logNHI)
                else:
                    logN = -np.inf
            return logN


def AskForParameters(model):
    lognH = float(raw_input("lognH = "))
    logZ = float(raw_input("logZ/Zsun = "))

    if model == 'photo_thick_aUV':
        aUV = float(raw_input("aUV = "))
        logNHI = float(raw_input("logNHI = "))
        alpha = np.array([lognH, logZ, aUV, logNHI])

    elif model == 'photo_thick':
        logNHI = float(raw_input("logNHI = "))
        alpha = np.array([lognH, logZ, logNHI])

    elif model == 'jv_model':
        logNHI = float(raw_input("logNHI = "))
        amp_a = float(raw_input("a = "))
        amp_b = float(raw_input("b = "))
        alpha = np.array([amp_a, amp_b, lognH, logZ, logNHI])

    elif (model == 'photo_collision_thick' or
          model == 'photo_collision_rahmati' or
          model == 'photo_collision_thin'):
        logT = float(raw_input("logT = "))
        logNHI = float(raw_input("logNHI = "))
        alpha = np.array([lognH, logZ, logT, logNHI])
    else:
        print("Your model does not exist. Did you have a typo?")
        print("photo_collision_thin")
        print("photo_collision_thick")
        print("photo_collision_rahmati")
        print("photo_thick")
        print("photo_thick_aUV")
        exit()
    return alpha

if __name__ == '__main__':

    import sys
    model = sys.argv[1]
    if model == 'jv_model':
        model_redshift = 0.0
    else:
        model_redshift = float(sys.argv[2])
    alpha = AskForParameters(model)
    ion_model = DefineIonizationModel_test(model, model_redshift)

    ions = ion_lists()
    for ion in ions:
        logN = ion_model.model_prediction(alpha, ion)
        print("LogN: %s = %.2f" % (ion, logN))
