#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
# Cell magic method always stays at the top of the cell

# Imports from the Python standard library should be at the top
import os
import copy
import pickle
import itertools 

# Do not import * unless you know what you are doing
import numpy as np 
import pandas as pd

import fsps
import sedpy
import lineid_plot
import torch
import torch.nn as nn 

from sedpy.observate import getSED, vac2air, air2vac

import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)

import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column
from astropy.constants import c, L_sun, pc
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.fitting import fit_generic_continuum
from specutils.analysis import equivalent_width

from prospect import models
from prospect.models import priors

from scipy.stats import entropy

# re-defining plotting defaults
from matplotlib import rcParams

from dwarf_models import SDSS_EMLINES, simulate_dwarf_sed, test_single_model,    sigma_clipping_continuum, measure_ew_emission_line, design_model_grid,    generate_dwarf_population, measure_color_ew, plot_models_with_sdss, setup_fsps_spop

rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'axes.titlepad': '15.0'})
rcParams.update({'font.size': 22})


# ### Read in the SDSS catalog

# In[2]:


sdss_cat = Table.read('/Users/runquanguan/Documents/Dwarf_SDSS_8_9_SF_v2.0.fits')

em_flag = (np.isfinite(sdss_cat['M_u']) & np.isfinite(sdss_cat['M_r']) &            np.isfinite(sdss_cat['M_g']) & np.isfinite(sdss_cat['M_i']) &           np.isfinite(sdss_cat['OIII_5007_EQW']) &            np.isfinite(sdss_cat['H_ALPHA_EQW']) &           np.isfinite(sdss_cat['H_BETA_EQW']))

sdss_use = sdss_cat[em_flag]


SDSS_EMLINES = {    'OII_3726': {'cen':3726.032, 'low':3717.0, 'upp':3737.0},    'OII_3729': {'cen':3728.815, 'low':3717.0, 'upp':3737.0},    'NeIII_3869': {'cen':3869.060, 'low':3859.0, 'upp':3879.0},     'H_delta': {'cen':4101.734, 'low':4092.0, 'upp':4111.0},    'H_gamma': {'cen':4340.464, 'low':4330.0, 'upp':4350.0},    'OIII_4363': {'cen':4363.210, 'low':4350.0, 'upp':4378.0},    'H_beta': {'cen':4861.325, 'low':4851.0, 'upp':4871.0},    'OIII_4959': {'cen':4958.911, 'low':4949.0, 'upp':4969.0},    'OIII_5007': {'cen':5006.843, 'low':4997.0, 'upp':5017.0},    'HeI_5876': {'cen':5875.67, 'low':5866.0, 'upp':5886.0},    'OI_6300': {'cen':6300.304, 'low':6290.0, 'upp':6310.0},    'NII_6548': {'cen':6548.040, 'low':6533.0, 'upp':6553.0},    'H_alpha': {'cen':6562.800, 'low':6553.0, 'upp':6573.0},    'NII_6584': {'cen':6583.460, 'low':6573.0, 'upp':6593.0},    'SII_6717': {'cen':6716.440, 'low':6704.0, 'upp':6724.0},    'SII_6731': {'cen':6730.810, 'low':6724.0, 'upp':6744.0},    'ArIII7135': {'cen':7135.8, 'low':7130.0, 'upp':7140.0}
}


# In[3]:


from hyperopt import hp, fmin, rand, tpe, space_eval

space = [hp.choice('tau_mean', [1.6, 2.6, 3.6, 4.6, 5.6]),
         hp.choice('const_mean', [0.1, 0.2, 0.3, 0.4, 0.5]),
         hp.choice('tage_mean', [2.5, 4.5, 6.5, 8.5, 10.5]),
         hp.choice('fburst_mean', [0.4, 0.5, 0.6, 0.7, 0.8]),
         hp.choice('tburst_mean', [3.0, 4.0, 5.0, 6.0, 7.0]),
         hp.choice('logzsol_mean', [-1.2, -1, -0.8, -0.6, -0.4]),
         hp.choice('gas_logz_mean', [-0.9, -0.7, -0.5, -0.3, -0.1]),
         hp.choice('gas_logu_mean', [-3.7, -3.2, -2.7, -2.2, -1.2]),
         hp.choice('tau_sig', [0.1, 0.2, 0.3, 0.4, 0.5]),
         hp.choice('const_sig', [0.1, 0.2, 0.3]),
         hp.choice('tage_sig', [0.1, 0.3, 0.]),
         hp.choice('fburst_sig', [0.1, 0.2, 0.3]),
         hp.choice('tburst_sig', [0.1, 0.2, 0.3, 0.4, 0.5]),
         hp.choice('logzsol_sig', [0.1, 0.2, 0.3, 0.4, 0.5]),
         hp.choice('gas_logz_sig', [0.1, 0.2, 0.3, 0.4, 0.5]),
         hp.choice('gas_logu_sig', [0.1, 0.2, 0.3, 0.4, 0.5]), 
        ]


# In[4]:


def loss(true_set, predict_set, bins_range):
    
    sdss_hist = np.histogram(true_set, bins = bins_range)[0]
    sps_hist = np.histogram(predict_set, bins = bins_range)[0]
    
    sdss_hist_norm = [float(i+1e-4)/sum(sdss_hist) for i in sdss_hist]
    sps_hist_norm = [float(i+1e-4)/sum(sps_hist) for i in sps_hist]
    
    x = torch.tensor([sdss_hist_norm])
    y = torch.tensor([sps_hist_norm])
    
    criterion = nn.KLDivLoss()
    loss = criterion(x.log(),y)   
    
    return loss.item()

    
    


# In[ ]:


def loss_function(args):

    
    tau_mean, const_mean, tage_mean, fburst_mean, tburst_mean, logzsol_mean, gas_logz_mean, gas_logu_mean,\
        tau_sig, const_sig, tage_sig, fburst_sig, tburst_sig, logzsol_sig, gas_logz_sig, gas_logu_sig = args
    
    set_size = 3000

    tau_arr = [float(priors.ClippedNormal(mean=tau_mean, sigma=tau_sig, 
                                          mini=1.0, maxi=8.0).sample()) for _ in range(set_size)]
    const_arr =  [float(priors.ClippedNormal(mean=const_mean, sigma=const_sig, 
                                             mini=0.0, maxi=0.5).sample()) for _ in range(set_size)]
    tage_arr =  [float(priors.ClippedNormal(mean=tage_mean, sigma=tage_sig, 
                                            mini=1.0, maxi=11.0).sample()) for _ in range(set_size)]
    fburst_arr =  [float(priors.ClippedNormal(mean=fburst_mean, sigma=fburst_sig, 
                                              mini=0.0, maxi=0.8).sample()) for _ in range(set_size)]
    tburst_arr =  [float(priors.ClippedNormal(mean=tburst_mean, sigma=tburst_sig, 
                                              mini=0.0, maxi=8.0).sample()) for _ in range(set_size)]
    logzsol_arr =  [float(priors.ClippedNormal(mean=logzsol_mean, sigma=logzsol_sig, 
                                               mini=-1.5, maxi=0.0).sample()) for _ in range(set_size)]
    gas_logz_arr =  [float(priors.ClippedNormal(mean=gas_logz_mean, sigma=gas_logz_sig, 
                                                mini=-1.5, maxi=0.0).sample()) for _ in range(set_size)]
    gas_logu_arr =  [float(priors.ClippedNormal(mean=gas_logu_mean, sigma=gas_logu_sig, 
                                                mini=-4.0, maxi=-1.0).sample()) for _ in range(set_size)]
                 
    # Fix the fburst + const > 1 issue
    for ii in np.arange(len(const_arr)):
        if const_arr[ii] + fburst_arr[ii] >= 0.95:
            f_over = (const_arr[ii] + fburst_arr[ii]) - 0.95
            if fburst_arr[ii] >= (f_over + 0.01):
                fburst_arr[ii] = fburst_arr[ii] - (f_over + 0.01)
            else:
                const_arr[ii] = const_arr[ii] - (f_over + 0.01)

    # Fixed the rest
    dust1_arr = np.full(set_size, 0.1)
    dust2_arr = np.full(set_size, 0.0)
    sf_trunc_arr = np.full(set_size, 0.0)

    # List of model parameters
    dwarf_sample_parameters = [
         {
             'dust1': dust1_arr[ii], 
             'dust2': dust2_arr[ii],
             'logzsol': logzsol_arr[ii], 
             'gas_logz': gas_logz_arr[ii], 
             'gas_logu': gas_logu_arr[ii],
             'const': const_arr[ii], 
             'tau': tau_arr[ii], 
             'tage': tage_arr[ii],
             'sf_trunc': sf_trunc_arr[ii], 
             'fburst': fburst_arr[ii], 
             'tburst': tburst_arr[ii]
         } for ii in np.arange(set_size)
    ]

    # Double check
    for ii, model in enumerate(dwarf_sample_parameters):
        if model['fburst'] + model['const'] >= 0.99:
            print(ii, model['fburst'], model['const'])
            
            
    # Initialize the spop model
    spop_tau = setup_fsps_spop(
        zcontinuous=1, imf_type=2, sfh=1, dust_type=0, 
        dust_index=-1.3, dust1_index=-1.0)

    # Get the SDSS filters
    sdss_bands = fsps.find_filter('SDSS')
    
    dwarf_sample_gaussian = generate_dwarf_population(
        spop_tau, dwarf_sample_parameters, filters=sdss_bands, n_jobs=6)


    # Measure colors and emission line EWs
    # - SDSS_EMLINES is a pre-defined dict of emission lines center wavelength and the 
    # wavelength window for measuring EW.
    # - You can save the results in a numpy array
    dwarf_sample_table = measure_color_ew(
        dwarf_sample_gaussian, em_list=SDSS_EMLINES, output=None)

    bin_size = 200

    ur_loss = loss(dwarf_sample_table['ur_color'],np.asarray(sdss_use['M_u'] - sdss_use['M_r']), np.linspace(0,2.5,bin_size))
    ug_loss = loss(dwarf_sample_table['ug_color'], np.asarray(sdss_use['M_u'] - sdss_use['M_g']), np.linspace(0,1.75,bin_size))
    gr_loss = loss(dwarf_sample_table['gr_color'], np.asarray(sdss_use['M_g'] - sdss_use['M_r']), np.linspace(-0.1,0.8,bin_size))
    gi_loss = loss(dwarf_sample_table['gi_color'], np.asarray(sdss_use['M_g'] - sdss_use['M_i']), np.linspace(-0.2,1.2,bin_size))
    OIII_loss = loss(np.log10(dwarf_sample_table['ew_oiii_5007']),
                     np.log10(-1.0 * (sdss_use['OIII_5007_EQW'])), np.linspace(-1,3,bin_size))
    Ha_loss = loss(np.log10(np.log10(dwarf_sample_table['ew_halpha'])),
                  np.log10(-1.0*sdss_use['H_ALPHA_EQW']), np.linspace(0,3,bin_size))
    Hb_loss = loss(np.log10(dwarf_sample_table['ew_hbeta']),
                  np.log10(-1.0*sdss_use['H_BETA_EQW']), np.linspace(-0.5,2.5,bin_size))

    total_loss = (ur_loss + ug_loss + gr_loss + gi_loss + OIII_loss + Ha_loss + Hb_loss)/7

    return total_loss


# In[6]:


best = fmin(loss_function, space, algo=tpe.suggest, max_evals = 100)

print(space_eval(space, best))


# In[7]:



'''

space_test = [hp.uniform('x',0,9), hp.normal('y',0,1)]

def q(args):
    x,y = args
    return x**2+y**2

best_test = fmin(q,space_test, algo = rand.suggest, max_evals = 100)
print(space_eval(space_test, best_test))

'''


# In[ ]:




