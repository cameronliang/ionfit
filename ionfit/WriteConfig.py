import numpy as np
import os
import re
import sys


def write_config(input_path, redshift):
    nwalkers = 200
    nsteps = 2000
    nthreads = 16
    model_choice = 'photo_collision_thick'

    files = os.listdir(input_path)

    ions_list = []
    for temp_file in files:
        if re.search('logN', temp_file):
            ion_name = temp_file[5:-4]
            ions_list.append(ion_name)

    f = open(input_path + '/ionfit_config.dat', 'w')
    f.write('input %s\n' % input_path)
    f.write('output chain_z%s\n' % str(redshift))
    f.write('mcmc %d %d %d\n' % (nwalkers, nsteps, nthreads))

    f.write('ions ')
    for ion in ions_list:
        if ion == 'n2' or ion == 'n3' or ion == 'o6' or ion == 'n5':
            pass
        else:
            f.write('%s ' % ion)
    f.write('\n')
    f.write('model %s %s\n' % (model_choice, str(redshift)))
    f.write('lognH -6 0\n')
    f.write('logZ -4 2\n')
    f.write('logT 4 6\n')
    f.write('logNHI 10 23\n')
    f.close()
    return


if __name__ == '__main__':

    input_path = sys.argv[1]
    redshift = format(float(sys.argv[2]), '.6f')
    write_config(input_path, redshift)
