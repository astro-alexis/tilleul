import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares,minimize
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
import pickle
from scipy import interpolate
import argparse
from astropy.modeling.models import Lorentz1D, Gaussian1D
from astropy.convolution import convolve, CustomKernel
from numpy import trapz

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
wave, intensity = obs['wavelength'][:,0,:]*10., obs['intensity'][:,0,:]
par0 = {'k' : [1.,1.,1.,1.,1.]}
par0['gauss_fwhm'], par0['lorentz_fwhm'] =  1., 1.
par0['scaling'] = np.ones((wave.shape[0]))
par0['slope'] = np.zeros((wave.shape[0])) 

#reduce_tell
def reduce_tell(tell, wave):
    nord = wave.shape[0]
    species = np.sort([k  for k in tell.keys() if k!= 'wave'])
    tell2 = {species[0] : np.zeros(wave.shape)}
    for ik in range(len(species)):
        tell2[species[ik]] = np.array(np.squeeze([interpolate.interp1d(tell['wave'], tell[species[ik]], kind='linear', bounds_error=False)(wave[j,:]) for j in range(nord)]))
    return tell2

# gen_tell generated a model telluric spectrum
# combined different species, convolve with a kernel, interpolate on new wavelength solution
def gen_tell(wave, intensity, tell2, par):
    nord = wave.shape[0]
    species = np.sort([k  for k in tell.keys() if k!= 'wave'])
    tellmod = np.ones(intensity.shape)
    for ik in range(len(species)):
#        tellbuff = np.array(np.squeeze([interpolate.interp1d(tell['wave'], tell[species[ik]], kind='linear', bounds_error=False)(wave[j,:]) for j in range(nord)]))
        tellmod *= tell2[species[ik]] ** par['k'][ik]
    kernlen = 5
    kernbase = np.arange(kernlen * 2 + 1) - kernlen 
    kern =  Gaussian1D(1,0,par['gauss_fwhm'])(kernbase)  + Lorentz1D(1.,0,par['lorentz_fwhm'])(kernbase)
    kernel = CustomKernel(kern)
    tellmodconv = np.array(np.squeeze([convolve(tellmod[j], kernel, normalize_kernel=True, boundary='extend') for j in range(nord)]))
    return tellmodconv

def gen_obs(wave,intensity, tellmod, par):
    nord = wave.shape[0]
    stellar = np.ones(wave.shape)
    for iord in range(nord):
        stellar[iord,:] *= (par['scaling'][iord]+ np.arange(len(stellar[iord,:])) * par['slope'][iord])
    obs = tellmod * stellar
    return obs


def minchi2(varpar, wave, intensity, tell2, par):
    nord, nspe = wave.shape[0], len(par['k'])
    scaling = varpar[0:nord]
    slope = varpar[nord:nord+nord]
    k = varpar[nord+nord:nord+nord+nspe]
    gauss_fwhm = varpar[-2]
    lorentz_fwhm = varpar[-1]
    stellar = np.ones(wave.shape)
    for iord in range(nord):
        stellar[iord,:]  = stellar[iord,:] * (scaling[iord] + np.arange(len(stellar[iord,:])) * slope[iord])
    par['gauss_fwhm'] = gauss_fwhm
    par['lorentz_fwhm'] = lorentz_fwhm
    par['k'] = k
    tellmod = gen_tell(wave, intensity, tell2, par)
    obs = tellmod * stellar
    return np.nansum((intensity-obs)**2.)
    
tell2 = reduce_tell(tell,wave)
tellmod = gen_tell(wave, intensity, tell2, par0)
obsg = gen_obs(wave,intensity, tellmod, par0)

startpar = [sc for sc in par0['scaling']]
bmin,bmax = [0.5 for sc in par0['scaling']], [2.5 for sc in par0['scaling']]

for iord in range(len(par0['slope'])):
    startpar.append(par0['slope'][iord])
    bmin.append(- 1./2000)
    bmax.append( 1./2000)
for ispe in range(len(par0['k'])):
    startpar.append(par0['k'][ispe])
    bmin.append(0)
    bmax.append(10)
startpar.append(par0['gauss_fwhm'])
bmin.append(0.2), bmax.append(5)
startpar.append(par0['lorentz_fwhm'])
bmin.append(0.2), bmax.append(5)

res = least_squares(minchi2, startpar , args=(wave, intensity, tell2, par0), bounds=(bmin,bmax), verbose=2, tr_solver='lsmr', max_nfev=200,)
nord = wave.shape[0]
par0['scaling'], par0['slope'], par0['k'], par0['gauss_fwhm'], par0['lorentz_fwhm'] = res.x[0:nord],res.x[nord:nord+nord], res.x[nord+nord:-2], res.x[-2], res.x[-1]
obs2 = gen_obs(wave,intensity, tellmod, par0)

