import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc, rcParams
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

def SetPlotStyle():
    rc('text',usetex=True)
    rc('font',**{'family':'serif','serif':['Computer Modern']})
    plt.rcParams['axes.linewidth']  = 1.
    plt.rcParams['axes.labelsize']  = 15
    plt.rcParams['axes.titlesize']  = 15
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['legend.fontsize']  = 15
    plt.rcParams['legend.frameon']  = False

    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1

def make_plot(exps, cmbs=['TT','EE','BB'],
              lsep=200, ms=1, ms_ul=1.5, elinewidth=0.5, txtsize=12, 
              theorycolor='grey', theory='camb', cutsom_style=True,
              **kwargs):

    if cutsom_style:
        SetPlotStyle()

    if theory == 'camb':
        import camb
        pars = camb.CAMBparams()

        # Some Planck2018 best-fit cosmology
        pars.set_cosmology(H0=kwargs.get('H0', 67.32117), 
                        ombh2=kwargs.get('ombh2',0.02238280), 
                        omch2=kwargs.get('omch2',0.1201075), 
                        mnu=kwargs.get('mnu',0.0543), 
                        omk=kwargs.get('omk', 0),
                        tau=kwargs.get('tau',0.06)
                        )

        pars.WantTensors = True
        pars.InitPower.set_params(As=kwargs.get('As',2.100549e-9), 
                                  ns=kwargs.get('ns',0.9660499), 
                                  r=kwargs.get('r',0.01)
                                  )
        pars.set_for_lmax(kwargs.get('lmax',5000)+500, lens_potential_accuracy=kwargs.get('lens_potential_accuracy',1)));
        results = camb.get_results(pars)
        powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

        dl_th['TT'] = powers['lensed_scalar'][:,0]
        dl_th['EE'] = powers['lensed_scalar'][:,1]
        dl_th['BB'] = powers['lensed_scalar'][:,2]
        dl_th['BB'] = powers['tensor'][:,2] 
 
    plt.figure(figsize=kwargs.get('figsize'))
    axMain = plt.subplot(111)
