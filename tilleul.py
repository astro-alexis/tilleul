import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
import sys
import pickle
from scipy import interpolate
import argparse

parser = argparse.ArgumentParser(description="quick & dirty telluric removal routing",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--telluricfile", default="tellurics/tellurics.pickle", help="file with telluric spectra for different species")
parser.add_argument("-o", "--obsfile", default="obs.pickle", help="file with observed spectrum")
parser.add_argument("-f", "--firstrun", default=False, help="first run: convert telluric fits to single file")
args = vars(parser.parse_args()) 

# Load telluric spectra, taken from the Viper Github
print(args['firstrun'])
if args['firstrun'] == "True":
    tCH4 = np.squeeze(np.array(fits.open('tellurics/stdAtmos_crires_CH4.fits')[1].data))
    tCO  = np.squeeze(np.array(fits.open('tellurics/stdAtmos_crires_CO.fits')[1].data))
    tCO2 = np.squeeze(np.array(fits.open('tellurics/stdAtmos_crires_CO2.fits')[1].data))
    tH2O = np.squeeze(np.array(fits.open('tellurics/stdAtmos_crires_H2O.fits')[1].data))
    tN2O = np.squeeze(np.array(fits.open('tellurics/stdAtmos_crires_N2O.fits')[1].data))
    data = {'wave' : tCH4['wave']}
    data['CH4'] = tCH4['flux']
    data['CO'] = tCO['flux'] 
    data['CO2'] = tCO2['flux'] 
    data['H2O'] = tH2O['flux'] 
    data['N2O'] = tN2O['flux'] 
    pickle.dump( data, open( "tellurics/tellurics.pickle", "wb" ) )
    print('Telluric files dumped to "tellurics/tellurics.pickle"')
else:
    tell = pickle.load(open(args['telluricfile'], "rb" ))
    print('Telluric synthetic spectrum loaded from ' + args["telluricfile"])

# Load observations
obs = pickle.load( open(args['obsfile'], "rb" ) )
wave, intensity = obs['wavelength'][0,0,:]*10., obs['intensity'][0,0,:]


tellmod = tell['H2O'] * 2500.
tellmodf = interpolate.interp1d(tell['wave'], tellmod, kind='linear', bounds_error=False)(wave)
#kernel = Gaussian1DKernel(stddev=100)
#convoluted = convolve(tCO['flux'], kernel, normalize_kernel=True,
#                      boundary='extend')

#plt.plot(tCO['wave'], tCO['flux'])
#plt.plot(tCO['wave'], convoluted)
plt.plot(wave,intensity)
plt.plot(wave,tellmodf)
plt.show()
