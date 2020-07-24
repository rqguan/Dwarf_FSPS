"""Misc utilities."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import copy
import pickle
import itertools
import urllib.request

# Do not import * unless you know what you are doing
import numpy as np
import pandas as pd

from scipy.stats import sigmaclip

import fsps
import sedpy
from sedpy.observate import getSED, vac2air, air2vac

from extinction import fm07, ccm89, odonnell94, remove

import astropy.units as u
from astropy.table import Table, Column
from astropy.io import fits

from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.analysis import equivalent_width

# re-defining plotting defaults
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams

plt.rc('text', usetex=True)
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
rcParams.update({'font.size': 26})


__all__ = ['get_sdss_spectrum', 'normalize_spectrum_window',
           'deredden_spectrum', 'compare_sdss_fsps_spectrum',
           'compare_sdss_fsps_spectrum', 'setup_fsps_spop',
           'simulate_dwarf_sed', 'test_single_model',
           'sigma_clipping_continuum', 'measure_ew_emission_line',
           'design_model_grid', 'generate_dwarf_population',
           'measure_color_ew', 'plot_models_with_sdss',
           'sigmoid_narrow_filter', 'filters_to_sedpy_format',
           'SDSS_EMLINES']

SDSS_EMLINES = {
    'OII_3726': {'cen':3726.032, 'low':3717.0, 'upp':3737.0},
    'OII_3729': {'cen':3728.815, 'low':3717.0, 'upp':3737.0},
    'NeIII_3869': {'cen':3869.060, 'low':3859.0, 'upp':3879.0},
    'H_delta': {'cen':4101.734, 'low':4092.0, 'upp':4111.0},
    'H_gamma': {'cen':4340.464, 'low':4330.0, 'upp':4350.0},
    'OIII_4363': {'cen':4363.210, 'low':4350.0, 'upp':4378.0},
    'H_beta': {'cen':4861.325, 'low':4851.0, 'upp':4871.0},
    'OIII_4959': {'cen':4958.911, 'low':4949.0, 'upp':4969.0},
    'OIII_5007': {'cen':5006.843, 'low':4997.0, 'upp':5017.0},
    'HeI_5876': {'cen':5875.67, 'low':5866.0, 'upp':5886.0},
    'OI_6300': {'cen':6300.304, 'low':6290.0, 'upp':6310.0},
    'NII_6548': {'cen':6548.040, 'low':6533.0, 'upp':6553.0},
    'H_alpha': {'cen':6562.800, 'low':6553.0, 'upp':6573.0},
    'NII_6584': {'cen':6583.460, 'low':6573.0, 'upp':6593.0},
    'SII_6717': {'cen':6716.440, 'low':6704.0, 'upp':6724.0},
    'SII_6731': {'cen':6730.810, 'low':6724.0, 'upp':6744.0},
    'ArIII7135': {'cen':7135.8, 'low':7130.0, 'upp':7140.0}
}


def get_sdss_spectrum(plate, mjd, fiberid, rest_frame=True, ebv=None):
    """Download SDSS spectrum using Plate-MJD-FiberID."""
    # Generate URL for download
    sdss_url = """
    https://dr15.sdss.org/optical/spectrum/view/data/format=fits/spec=full?plateid=%s&mjd=%s&fiberid=%s
    """ % (str(plate).strip(), str(mjd).strip(), str(fiberid).strip())

    urllib.request.urlretrieve(sdss_url, 'temp.fits')
    sdss_fits = fits.open('temp.fits')
    os.remove('temp.fits')

    sdss_spec = sdss_fits[1].data
    sdss_ews = Table(sdss_fits[3].data)
    wave, flux = (10.0 ** sdss_spec['loglam']), sdss_spec['flux']
    redshift = sdss_fits[2].data['Z'][0]

    # First, deredden the spectrum at the observed frame
    if ebv is not None:
        flux_deredd = deredden_spectrum(
            wave.astype(np.float64),
            flux.astype(np.float64), ebv)
    else:
        print("# The spectrum is not corrected for Galactic extinction!")

    # Put the spectrum back to the rest frame
    if rest_frame:
        if np.isfinite(redshift) and (redshift > 0):
            wave = wave / (1.0 + redshift)
            flux = flux / (1.0 + redshift)
        else:
            print("# No useful redshift !")

    return wave, flux, sdss_ews


def normalize_spectrum_window(wave, flux, low=3.0, upp=3.0,
                              wave_min=5200, wave_max=5400):
    """Nomalize spectrum using continuum within a wavelength window."""
    flag_use = (wave >= wave_min) & (wave <= wave_max)

    flux_clipped, flux_low, flux_upp = sigmaclip(flux[flag_use], low, upp)

    return flux / np.median(flux_clipped)


def deredden_spectrum(wave, flux, ebv, r_v=3.1, unit='aa', model='fm07'):
    """Deredden a spectrum based on a Galactic extinction model."""

    model = model.lower().strip()

    if model == 'fm07':
        mw_extinction = fm07(wave, ebv * r_v, unit=unit)
    elif model == 'ccm89':
        mw_extinction = ccm89(wave, ebv * r_v, r_v, unit=unit)
    elif model == 'odonnell94':
        mw_extinction = ccm89(wave, ebv * r_v, r_v, unit=unit)
    else:
        raise Exception("# Wrong choice of extinction model: fm07/ccm89/odonnell94")

    return remove(mw_extinction, flux)


def compare_sdss_fsps_spectrum(sdss_lam, sdss_flux_norm, fsps_lam,
                               fsps_flux_norm, fsps_cont_norm=None,
                               filter_list=None, window1=[3690, 7200],
                               window2=[4821, 5049], window3=[6520, 6750]):
    """Compare a SDSS spectrum to a FSPS model spectrum."""
    # ---------------------------------------------------------------------------------------------------- #
    # Plot the spectrum
    fig = plt.figure(figsize=(17, 15))

    gs = fig.add_gridspec(2, 7)

    ax1 = fig.add_subplot(gs[0, : ])

    ax1.grid(linestyle='--')
    ax1.axvline(5300.0, linewidth=20.0, alpha=0.1, c='k')

    ax1.step(sdss_lam, np.log10(sdss_flux_norm), label=r'$\mathrm{SDSS}$', alpha=1.0,
             zorder=1, c='darkgray')

    ax1.step(fsps_lam, np.log10(fsps_flux_norm), label=r'$\mathrm{FSPS}$', linewidth=2.0,
             alpha=0.8, zorder=2, c='coral')

    if fsps_cont_norm is not None:
        ax1.step(fsps_lam, np.log10(fsps_cont_norm), label=r'$\mathrm{FSPS\ Star}$', linewidth=1.5,
                 alpha=1.0, zorder=0)

    ax1.legend(loc='best')

    _ = ax1.set_xlim(window1[0], window1[1])
    ymin = np.log10(4E-1)
    ymax = np.log10(
        np.nanmax([np.nanmax(sdss_flux_norm), np.nanmax(fsps_flux_norm)]) * 1.1)
    _ = ax1.set_ylim(ymin, ymax)

    if filter_list is not None:
        for f in filter_list:
            w, t = f['wavelength'], f['response_curve']
            t = t * ymax + ymin
            ax1.fill_between(
                w, ymin, t, lw=2, edgecolor='gray', alpha=0.2, zorder=0)

    ax1.set_yticklabels([])
    _ = ax1.set_xlabel(r'$\mathrm{Wavelength}\ [\AA]$')
    _ = ax1.set_ylabel(r'$\log\ \mathrm{Flux}\ [\mathrm{Normalized}]$')

    # ---------------------------------------------------------------------------------------------------- #
    # Zoom in to the [OIII] + Hbeta region
    ax2 = fig.add_subplot(gs[1, 0:3])

    ax2.grid(linestyle='--')

    hbeta_flag_1 = (sdss_lam >= window2[0]) & (sdss_lam <= window2[1])
    ax2.step(sdss_lam[hbeta_flag_1], np.log10(sdss_flux_norm[hbeta_flag_1]),
             label=r'$\mathrm{SDSS}$', linewidth=2.5, alpha=0.9, c='grey')

    hbeta_flag_2 = (fsps_lam >= window2[0]) & (fsps_lam <= window2[1])
    ax2.step(fsps_lam[hbeta_flag_2], np.log10(fsps_flux_norm[hbeta_flag_2]),
             label=r'$\mathrm{FSPS}$', linewidth=2.0, alpha=1.0, c='coral')

    ax2.text(0.1, 0.85, r'$\mathrm{H}_{\beta}+[\mathrm{OIII}]$', fontsize=30,
             transform=ax2.transAxes)

    ymin = np.log10(
        np.nanmin([np.nanmin(fsps_flux_norm[hbeta_flag_2]),
                   np.nanmin(sdss_flux_norm[hbeta_flag_1])]) * 0.9)
    ymax = np.log10(
        np.nanmax([np.nanmax(fsps_flux_norm[hbeta_flag_2]),
                   np.nanmax(sdss_flux_norm[hbeta_flag_1])]) * 1.2)

    _ = ax2.set_xlim(window2[0], window2[1])
    _ = ax2.set_ylim(ymin, ymax)

    if filter_list is not None:
        for f in filter_list:
            w, t = f['wavelength'], f['response_curve']
            t = t * ymax * 0.8 + ymin
            ax2.fill_between(
                w, ymin, t, lw=2, edgecolor='gray', alpha=0.15, zorder=0)

    ax2.set_yticklabels([])
    _ = ax2.set_xlabel(r'$\mathrm{Wavelength}\ [\AA]$')
    _ = ax2.set_ylabel(r'$\log\ \mathrm{Flux}\ [\mathrm{Normalized}]$')

    # ---------------------------------------------------------------------------------------------------- #
    # Zoom in to the [NII] + halpha region
    ax3 = fig.add_subplot(gs[1, 3:7])

    ax3.grid(linestyle='--')

    halpha_flag_1 = (sdss_lam >= 6520) & (sdss_lam <= 6750)
    ax3.step(sdss_lam[halpha_flag_1], np.log10(sdss_flux_norm[halpha_flag_1]),
             label=r'$\mathrm{SDSS}$', linewidth=2.5, alpha=0.9, c='gray')

    halpha_flag_2 = (fsps_lam >= 6520) & (fsps_lam <= 6750)
    ax3.step(fsps_lam[halpha_flag_2], np.log10(fsps_flux_norm[halpha_flag_2]),
             label=r'$\mathrm{FSPS}$', linewidth=2.0, alpha=1.0, c='coral')

    ax3.text(0.4, 0.85, r'$\mathrm{H}_{\alpha}+[\mathrm{NII}]+[\mathrm{SII}]$', fontsize=30,
             transform=ax3.transAxes)

    ymin = np.log10(
        np.nanmin([np.nanmin(fsps_flux_norm[halpha_flag_2]),
                   np.nanmin(sdss_flux_norm[halpha_flag_1])]) * 0.9)
    ymax = np.log10(
        np.nanmax([np.nanmax(fsps_flux_norm[halpha_flag_2]),
                   np.nanmax(sdss_flux_norm[halpha_flag_1])]) * 1.2)

    _ = ax3.set_xlim(window3[0], window3[1])
    _ = ax3.set_ylim(ymin, ymax)

    if filter_list is not None:
        for f in filter_list:
            w, t = f['wavelength'], f['response_curve']
            t = t * ymax * 0.8 + ymin
            ax3.fill_between(
                w, ymin, t, lw=2, edgecolor='gray', alpha=0.15, zorder=0)

    ax3.set_yticklabels([])
    _ = ax3.set_xlabel(r'$\mathrm{Wavelength}\ [\AA]$')

    return fig


def setup_fsps_spop(zcontinuous=1, imf_type=1, sfh=1, dust_type=0,
                    dust_index=-1.3, dust1_index=-1.0):
    """Setup the fsps.StellarPopulation() object."""

    # The goal is to just call this initialization once
    # Also put all the fixed parameters here.
    spop = fsps.StellarPopulation(
        zcontinuous=zcontinuous, add_neb_emission=1,
        imf_type=imf_type, sfh=sfh, dust_type=dust_type,
        dust_index=dust_index, dust1_index=dust1_index)

    _ = spop.get_spectrum(
        peraa=True, tage=np.random.uniform(10., 13.))

    return spop


def simulate_dwarf_sed(model, spop, wave_min=3700, wave_max=7400, filters=None,
                       spec_no_emline=True, peraa=True):
    """Simulate the spectrum and SED of a SF dwarf galaxy."""
    # Default Spop model without emission lines
    spop.params['add_neb_emission'] = 1
    spop.params['add_neb_continuum'] = 1

    # Stellar metallicity: only useful when zcontinuous = 1
    spop.params['logzsol'] = model['logzsol']

    if model['const'] + model['fburst'] >= 1.0:
        print(
            "# Adjust the model to make sure const + fburst < 1: %3.1f + % 3.1f!" % (model['const'], model['fburst']))
        f_over = model['const'] + model['fburst'] - 1.0
        if model['fburst'] >= (f_over + 0.01):
            model['fburst'] -= f_over
        else:
            model['const'] -= f_over

    # SFH parameters for tau model
    spop.params['tau'] = model['tau']
    spop.params['sf_trunc'] = model['sf_trunc']

    # FIXME
    try:
        spop.params['fburst'] = model['fburst']
        spop.params['const'] = model['const']
    except AssertionError:
        print("Current: ", spop.params['fburst'], spop.params['const'])
        print("New: ", model['fburst'], model['const'])
        spop.params['fburst'] = 0.0
        spop.params['const'] = model['const']

    spop.params['tburst'] = model['tburst']

    # Parameters for dust extinction
    spop.params['dust1'] = model['dust1']
    spop.params['dust2'] = model['dust2']

    # Nebular emission parameters
    spop.params['gas_logz'] = model['gas_logz']
    spop.params['gas_logu'] = model['gas_logu']

    # Spectrum with emission lines and nebular continuum
    wave_rest, spec_em = spop.get_spectrum(peraa=peraa, tage=model['tage'])

    # Get magnitudes
    if filters is not None:
        model['sed'] = spop.get_mags(bands=filters, tage=model['tage'])

    # Only keep part of the spectrum
    flag_use = (wave_rest >= wave_min) & (wave_rest <= wave_max)

    # Save the result to the dict
    model['wave'] = wave_rest[flag_use]
    model['spec_em'] = spec_em[flag_use]

    # Current stellar mass
    model['mstar'] = spop.stellar_mass

    # Emission line luminosities
    model['emline_luminosity'] = spop.emline_luminosity

    # Star formation rate (normalized by the total amount of star formed)
    model['sfr'] = spop.sfr

    if spec_no_emline:
        # Now turn off the emission line
        spop.params['add_neb_emission'] = 0
        spop.params['add_neb_continuum'] = 0

        _, spec_ne= spop.get_spectrum(peraa=peraa, tage=model['tage'])

        model['spec_ne'] = spec_ne[flag_use]

    # Rest these two fractions, otherwise it may cause problem in the next model
    spop.params['fburst'] = 0.
    spop.params['const'] = 0.

    return model


def test_single_model(model, spop, sdss_data, wave_min=3700, wave_max=7400,
                      em_list=SDSS_EMLINES):
    """Simulate the spectrum and SED of a SF dwarf galaxy."""
    # Get SDSS filters
    sdss_bands = fsps.find_filter('SDSS')

    # Generate model spectrum and SED
    model = simulate_dwarf_sed(
        model, spop, wave_min=3700, wave_max=7400, filters=sdss_bands,
        spec_no_emline=True, peraa=True)

    # Measure EWs of key emission lines
    model['ew_halpha'] = measure_ew_emission_line(
        model, em_list['H_alpha'], wave_margin=200, redshift=0.0)

    model['ew_hbeta'] = measure_ew_emission_line(
        model, em_list['H_beta'], wave_margin=200, redshift=0.0)

    model['ew_oiii_5007'] = measure_ew_emission_line(
        model, em_list['OIII_5007'], wave_margin=200, redshift=0.0)

    # Get magnitudes
    model['ur_color'] = model['sed'][0] - model['sed'][2]
    model['ug_color'] = model['sed'][0] - model['sed'][1]
    model['gr_color'] = model['sed'][1] - model['sed'][2]
    model['gi_color'] = model['sed'][1] - model['sed'][3]

    # Print a few important parameters
    print("# Model u-r color : %7.3f" % model['ur_color'])
    print("# Model u-g color : %7.3f" % model['ug_color'])
    print("# Model g-r color : %7.3f" % model['gr_color'])
    print("# Model EW(Halpha) : %7.3f" % model['ew_halpha'])
    print("# Model EW([OIII]) : %7.3f" % model['ew_oiii_5007'])
    print("# Current stellar mass : %8.5f" % model['mstar'])
    print("# log(SFR) : %8.3f" % np.log10(model['sfr']))

    # ---------------------------------------------------------------------------------------------------- #
    # make a summary plot
    fig = plt.figure(figsize=(14, 20))

    # Grids
    gs = fig.add_gridspec(3, 2)

    # ---------------------------------------------------------------------------------------------------- #
    # Plot the spectrum
    ax1 = fig.add_subplot(gs[0, :])

    ax1.grid(linestyle='--')

    flag_use = (model['wave'] >= 3708) & (model['wave'] <= 7225)

    ax1.step(model['wave'][flag_use], model['spec_em'][flag_use], alpha=0.8)
    ax1.step(model['wave'][flag_use], model['spec_ne'][flag_use], alpha=0.8)

    _ = ax1.set_xlabel(r'$\mathrm{Wavelength}\ [\AA]$')

    # ---------------------------------------------------------------------------------------------------- #
    # Color-Color plot: u-g v.s. g-r
    ax2 = fig.add_subplot(gs[1, 0])

    _ = ax2.hist2d(
        sdss_data['M_u'] - sdss_data['M_g'], sdss_data['M_g'] - sdss_data['M_r'],
        range=[[-0.05, 1.79], [-0.05, 0.79]], bins=[40, 35], cmap='viridis', cmin=5, alpha=0.8)

    ax2.scatter(np.asarray(model['ur_color']) - np.asarray(model['gr_color']),
                model['gr_color'], s=125, alpha=0.8, edgecolor='k', facecolor='r')

    ax2.grid(linestyle='--')

    _ = ax2.set_xlabel(r'$u-g\ [\mathrm{mag}]$')
    _ = ax2.set_ylabel(r'$g-r\ [\mathrm{mag}]$')

    # ---------------------------------------------------------------------------------------------------- #
    # (g-r) v.s. EW(Halpha)
    ax3 = fig.add_subplot(gs[1, 1])

    _ = ax3.hist2d(
        sdss_data['M_g'] - sdss_data['M_r'], np.log10(-1.0 * (sdss_data['H_ALPHA_EQW'])),
        range=[[-0.05, 0.79], [0.05, 2.9]], bins=[40, 35], cmap='viridis', cmin=5, alpha=0.8)

    ax3.scatter(model['gr_color'], np.log10(model['ew_halpha']),
                s=125, alpha=0.8, edgecolor='k', facecolor='r')

    ax3.grid(linestyle='--')

    _ = ax3.set_xlabel(r'$g-r\ [\mathrm{mag}]$')
    _ = ax3.set_ylabel(r'$\log\ \mathrm{EW(H}\alpha)\ \AA$')

    # ---------------------------------------------------------------------------------------------------- #
    # (u-r) v.s. EW([OIII])
    ax4 = fig.add_subplot(gs[2, 0])

    _ = ax4.hist2d(
        sdss_data['M_u'] - sdss_data['M_r'], np.log10(-1.0 * (sdss_data['OIII_5007_EQW'])),
        range=[[0.05, 2.49], [-0.9, 2.9]], bins=[40, 35], cmap='viridis', cmin=5, alpha=0.8)

    ax4.scatter(model['ur_color'], np.log10(model['ew_oiii_5007']),
                s=125, alpha=0.8, edgecolor='k', facecolor='r')

    ax4.grid(linestyle='--')

    _ = ax4.set_xlabel(r'$u-r\ [\mathrm{mag}]$')
    _ = ax4.set_ylabel(r'$\log\ \mathrm{EW([OIII]\ 5007)}\ \AA$')

    # ---------------------------------------------------------------------------------------------------- #
    # (g-i) v.s. EW(Hbeta)
    ax5 = fig.add_subplot(gs[2, 1])

    _ = ax5.hist2d(
        sdss_data['M_g'] - sdss_data['M_i'], np.log10(-1.0 * (sdss_data['H_BETA_EQW'])),
        range=[[-0.05, 0.99], [0.01, 1.9]], bins=[40, 35], cmap='viridis', cmin=5, alpha=0.8)

    ax5.scatter(model['gi_color'], np.log10(model['ew_hbeta']),
                s=125, alpha=0.8, edgecolor='k', facecolor='r')

    ax5.grid(linestyle='--')

    _ = ax5.set_xlabel(r'$g-i\ [\mathrm{mag}]$')
    _ = ax5.set_ylabel(r'$\log\ \mathrm{EW(H}\beta)\ \AA$')

    return model, fig


def sigma_clipping_continuum(wave, flux, low=5.0, upp=3.0, degree=3):
    """Fit a simple polynomial continuum after sigma clipping."""

    from scipy.stats import sigmaclip

    _, flux_low, flux_upp = sigmaclip(flux, low, upp)

    mask = (flux >= flux_low) & (flux <= flux_upp)

    return flux / np.poly1d(np.polyfit(wave[mask], flux[mask], degree))(wave)


def measure_ew_emission_line(model, emline, wave_margin=300, redshift=0.0,
                             cont_low=5, cont_upp=3, cont_degree=2):
    """Measure the EW of an emission line after normalization."""
    # Decide the wavelength range
    wave_flag = ((model['wave'] >= emline['cen'] - wave_margin) &
                 (model['wave'] <= emline['cen'] + wave_margin))

    wave_use = model['wave'][wave_flag]
    flux_em = model['spec_em'][wave_flag]
    flux_ne = model['spec_ne'][wave_flag]

    # Normalize the spectrum, so the continuum level is 1.0
    flux_em_norm = sigma_clipping_continuum(
        wave_use, flux_em, low=5, upp=2, degree=cont_degree)

    flux_ne_norm = sigma_clipping_continuum(
        wave_use, flux_ne, low=2, upp=5, degree=cont_degree)

    # Form a Spectrum1D object
    ew_em = equivalent_width(
        Spectrum1D(
            spectral_axis=wave_use * u.AA,
            flux=flux_em_norm * u.Unit('erg cm-2 s-1 AA-1')
        ),
        regions=SpectralRegion(emline['low'] * u.AA * (1.0 + redshift),
                               emline['upp'] * u.AA * (1.0 + redshift)),
        continuum=1
    ).value

    ew_ne = equivalent_width(
        Spectrum1D(
            spectral_axis=wave_use * u.AA,
            flux=flux_ne_norm * u.Unit('erg cm-2 s-1 AA-1')
        ),
        regions=SpectralRegion(emline['low'] * u.AA * (1.0 + redshift),
                               emline['upp'] * u.AA * (1.0 + redshift)),
        continuum=1
    ).value

    return ew_ne - ew_em


def design_model_grid(tage=None, tau=None, const=None, sf_trunc=None,
                      fburst=None, tburst=None, dust1=None, dust2=None,
                      logzsol=None, gas_logz=None, gas_logu=None):
    """Design model grid for SF dwarfs."""

    # Right now, let's assume a delayed tau model with 6 possible parameters

    # Age of the Universe
    if tage is None:
        tage = [6.0, 9.0]

    # Tau for delayed tau model
    if tau is None:
        tau = [0.2, 0.5, 1.0]

    # Constant part of the tau model
    if const is None:
        const = [0.0, 0.5]

    # Time of truncation
    if sf_trunc is None:
        sf_trunc = [0.0]

    # Star burst time
    if tburst is None:
        tburst = [11.0]

    # Fraction of stars formed in the burst
    if fburst is None:
        fburst = [0.0]

    # Dust parameter describing the attenuation of young stellar light
    if dust1 is None:
        dust1 = [0.2]

    # Dust parameter describing the attenuation of old stellar light
    if dust2 is None:
        dust2 = [0.0]

    # Stellar metallicity
    if logzsol is None:
        logzsol = [-1.5, -1.0, -0.3, 0.0]

    # Gas metallicity
    if gas_logz is None:
        gas_logz = [-1.5, -1.0, -0.3, 0.0]

    # Ionization parameters
    if gas_logu is None:
        gas_logu = [-2.5]

    grid = list(itertools.product(
        tage, tau, const, sf_trunc, tburst, fburst,
        dust1, dust2, logzsol, gas_logz, gas_logu))

    # Convert the grid into a list of dicts
    return [{
        'tage': model[0],
        'tau': model[1],
        'const': model[2],
        'sf_trunc': model[3],
        'tburst': model[4],
        'fburst': model[5],
        'dust1': model[6],
        'dust2': model[7],
        'logzsol': model[8],
        'gas_logz': model[9],
        'gas_logu': model[10]
    } for model in grid]


def generate_dwarf_population(spop, model_grid, filters=None, n_jobs=8, output=None):
    """Generate a population of dwarf spectra using pre-designed model grid."""
    if n_jobs > 1:
        print("# Will use multi-processing with %d cores!" % n_jobs)
        from multiprocessing import Pool
        from functools import partial

        simulate_dwarf_pool = partial(
            simulate_dwarf_sed, spop=spop, filters=filters)

        pool = Pool(n_jobs)

        models = pool.map(simulate_dwarf_pool, model_grid)
    else:
        models = [simulate_dwarf_sed(model, spop) for model in model_grid]

    if output is not None:
        pickle.dump(models, open(output, "wb" ) )

    return models


def measure_color_ew(models, em_list=SDSS_EMLINES, output=None):
    """Measure colors and EWs of key emission lines."""
    # Measure the color and EW of emission lines
    # Colors:
    ur_color = [model['sed'][0] - model['sed'][2] for model in models]
    ug_color = [model['sed'][0] - model['sed'][1] for model in models]
    gr_color = [model['sed'][1] - model['sed'][2] for model in models]
    gi_color = [model['sed'][1] - model['sed'][3] for model in models]

    # EW:
    ew_halpha = [measure_ew_emission_line(
        model, SDSS_EMLINES['H_alpha'], wave_margin=200,
        redshift=0.0) for model in models]

    ew_hbeta = [measure_ew_emission_line(
        model, SDSS_EMLINES['H_beta'], wave_margin=200,
        redshift=0.0) for model in models]

    ew_oiii_5007 = [measure_ew_emission_line(
        model, SDSS_EMLINES['OIII_5007'], wave_margin=200,
        redshift=0.0) for model in models]

    # Convert the list of dicts into astropy.table
    model_table = Table.from_pandas(pd.DataFrame(models))

    # Add columns for colors
    model_table.add_column(Column(data=ur_color, name='ur_color'))
    model_table.add_column(Column(data=ug_color, name='ug_color'))
    model_table.add_column(Column(data=gr_color, name='gr_color'))
    model_table.add_column(Column(data=gi_color, name='gi_color'))

    # Add columns for EWs
    model_table.add_column(Column(data=ew_halpha, name='ew_halpha'))
    model_table.add_column(Column(data=ew_hbeta, name='ew_hbeta'))
    model_table.add_column(Column(data=ew_oiii_5007, name='ew_oiii_5007'))

    if output is not None:
        # Save it as numpy array
        np.save(output, model_table.as_array())

    return model_table


def plot_models_with_sdss(models, sdss_data, wave_min=3720, wave_max=7225,
                          size_symbol=95, alpha_spec=0.2, alpha_symbol=0.8,
                          ecolor_symbol='k'):
    """Simulate the spectrum and SED of a SF dwarf galaxy."""
    # ---------------------------------------------------------------------------------------------------- #
    # Wavelength array
    wave_arr = models[0]['wave']

    # make a summary plot
    fig = plt.figure(figsize=(14, 20))

    # Grids
    gs = fig.add_gridspec(3, 2)

    # ---------------------------------------------------------------------------------------------------- #
    ug_color = models['ug_color']
    gr_color = models['gr_color']
    ur_color = models['ur_color']
    gi_color = models['gi_color']

    ew_halpha = models['ew_halpha']
    ew_hbeta = models['ew_hbeta']
    ew_oiii_5007 = models['ew_oiii_5007']

    ur_min, ur_max = np.min(ur_color), np.max(ur_color)

    # ---------------------------------------------------------------------------------------------------- #
    # Plot the spectrum
    ax1 = fig.add_subplot(gs[0, :])

    ax1.grid(linestyle='--')

    flag_use = (wave_arr >= wave_min) & (wave_arr <= wave_max)

    _ = [ax1.step(wave_arr[flag_use], np.log10(model['spec_em'][flag_use]),
                  alpha=alpha_spec, linewidth=1.0,
                  c=plt.get_cmap('coolwarm')((model['ur_color'] - ur_min) / (ur_max - ur_min)))
         for model in models]

    _ = ax1.set_xlabel(r'$\mathrm{Wavelength}\ [\AA]$')

    # ---------------------------------------------------------------------------------------------------- #
    # Color-Color plot: u-g v.s. g-r
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.grid(linestyle='--')

    _ = ax2.hist2d(
        sdss_data['M_u'] - sdss_data['M_g'], sdss_data['M_g'] - sdss_data['M_r'],
        range=[[-0.05, 1.79], [-0.05, 0.79]], bins=[40, 35], cmap='Greys', cmin=1, alpha=0.8)

    ax2.scatter(ug_color, gr_color, s=size_symbol, alpha=alpha_symbol, edgecolor=ecolor_symbol,
                facecolor=plt.get_cmap('coolwarm')((ur_color - ur_min) / (ur_max - ur_min)))

    _ = ax2.set_xlabel(r'$u-g\ [\mathrm{mag}]$')
    _ = ax2.set_ylabel(r'$g-r\ [\mathrm{mag}]$')

    # ---------------------------------------------------------------------------------------------------- #
    # (g-r) v.s. EW(Halpha)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.grid(linestyle='--')

    _ = ax3.hist2d(
        sdss_data['M_g'] - sdss_data['M_r'], np.log10(-1.0 * (sdss_data['H_ALPHA_EQW'])),
        range=[[-0.05, 0.79], [0.05, 2.9]], bins=[40, 35], cmap='Greys', cmin=1, alpha=0.8)

    ax3.scatter(gr_color, np.log10(ew_halpha), s=size_symbol, alpha=alpha_symbol,
                edgecolor=ecolor_symbol,
                facecolor=plt.get_cmap('coolwarm')((ur_color - ur_min) / (ur_max - ur_min)))

    _ = ax3.set_xlabel(r'$g-r\ [\mathrm{mag}]$')
    _ = ax3.set_ylabel(r'$\log\ \mathrm{EW(H}\alpha)\ \AA$')

    # ---------------------------------------------------------------------------------------------------- #
    # (u-r) v.s. EW([OIII])
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.grid(linestyle='--')

    _ = ax4.hist2d(
        sdss_data['M_u'] - sdss_data['M_r'], np.log10(-1.0 * (sdss_data['OIII_5007_EQW'])),
        range=[[0.05, 2.49], [-0.9, 2.9]], bins=[40, 35], cmap='Greys', cmin=1, alpha=0.8)

    ax4.scatter(ur_color, np.log10(ew_oiii_5007), s=size_symbol, alpha=alpha_symbol,
                edgecolor=ecolor_symbol,
                facecolor=plt.get_cmap('coolwarm')((ur_color - ur_min) / (ur_max - ur_min)))

    _ = ax4.set_xlabel(r'$u-r\ [\mathrm{mag}]$')
    _ = ax4.set_ylabel(r'$\log\ \mathrm{EW([OIII]\ 5007)}\ \AA$')

    # ---------------------------------------------------------------------------------------------------- #
    # (g-i) v.s. EW(Hbeta)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.grid(linestyle='--')

    _ = ax5.hist2d(
        sdss_data['M_g'] - sdss_data['M_i'], np.log10(-1.0 * (sdss_data['H_BETA_EQW'])),
        range=[[-0.05, 0.99], [0.01, 1.9]], bins=[40, 35], cmap='Greys', cmin=1, alpha=0.8)

    ax5.scatter(gi_color, np.log10(ew_hbeta), s=size_symbol, alpha=alpha_symbol,
                edgecolor=ecolor_symbol,
                facecolor=plt.get_cmap('coolwarm')((ur_color - ur_min) / (ur_max - ur_min)))

    _ = ax5.set_xlabel(r'$g-i\ [\mathrm{mag}]$')
    _ = ax5.set_ylabel(r'$\log\ \mathrm{EW(H}\beta)\ \AA$')

    return fig

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_narrow_filter(wave_cen, filter_width, peak_response=1.0, wave_margin=10):
    """Design artificial narrow band filter with sigmoid damping."""
    if filter_width < 20:
        raise Exception("The filter should be at least 20 Anstrom wide.")

    half_width = int(filter_width / 2) + wave_margin

    wavelength = np.arange(half_width * 2) + wave_cen - half_width

    half_window = np.arange(half_width) - wave_margin

    response_curve = np.concatenate(
        [_sigmoid(half_window), _sigmoid(half_window)[::-1]]) * peak_response

    return wavelength, response_curve


def filters_to_sedpy_format(name, wave, response):
    """Convert a filter response curve to the Sedpy/Kcorrect format."""
    assert len(wave) == len(response), '''
        Wavelength and response curve should have the same size'''

    par = open("%s.par" % name.lower().strip(), 'w')
    par.write(
        "# %s\n typedef struct {\n  double lambda;\n  double pass;\n } KFILTER;\n\n")
    for w, r in zip(wave, response):
        par.write("KFILTER %9.3f %10.6f\n" % (w, r))
    par.close()
